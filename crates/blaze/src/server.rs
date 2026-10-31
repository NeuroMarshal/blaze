//! HTTP inference API with streaming, tool calls and structured outputs.
//! The vendored blaze-chat renders chat templates and parses model output.

mod chat;
mod completions;
mod media;
mod embeddings;
mod messages;
mod responses;
mod tokenize;
mod control;

pub use media::{MmConfig, parse_limit};

use crate::sampler::SamplingParams;
use crate::sched::{Finish, GenEvent, GenRequest, Scheduler, ThinkBudget, TokenLogprob};
use crate::structured::{Constraint, Grammars, grammar_of};
use crate::text::Text;
use anyhow::Result;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use blaze_chat::ProcessorError;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use tokio::sync::mpsc::unbounded_channel;

/// Largest request body: a 500k-token prompt is a few MB of JSON, images (inlined as base64) more.
/// Most choices of one request (`n`); the scheduler queues what does not fit its rows at once.
const MAX_N: u64 = 64;
/// Most stop strings of a request and bytes of one (the cutter scans them after every token).
const MAX_STOPS: usize = 64;
const MAX_STOP_BYTES: usize = 1024;
/// Most stop_token_ids and bad_words of a request.
const MAX_STOP_IDS: usize = 1024;
/// Most generations of one completions request (prompts x n): they queue before every other client's.
pub(crate) const MAX_GENERATIONS: usize = 256;
/// Most alternatives per token (`top_logprobs`, completions `logprobs`), as vLLM's default max_logprobs.
const MAX_TOP_LOGPROBS: u64 = 20;

pub struct ServeConfig {
    pub api_keys: Vec<String>,
    pub max_concurrent_requests: usize,
    pub request_timeout: std::time::Duration,
    pub shutdown_timeout: std::time::Duration,
    pub body_limit: usize,
    pub allowed_origins: Vec<String>,
    pub effective: Value,
    pub served_name: String,
    pub host: String,
    pub port: u16,
    pub defaults: SamplingParams,
    /// defaults of chat requests with thinking off (None: `defaults`)
    pub instruct_defaults: Option<SamplingParams>,
    pub default_max_tokens: usize,
    pub tool_parser: Option<String>,
    pub reasoning_parser: Option<String>,
    /// default thinking budget of chat requests (None: unlimited)
    pub thinking_budget: Option<usize>,
    /// longest whitespace run between JSON tokens of structured outputs (0: unbounded)
    pub grammar_whitespace: usize,
}

struct AppState {
    http_slots: Arc<tokio::sync::Semaphore>,
    key_hashes: Vec<blake3::Hash>,
    mm: MmConfig,
    sched: Scheduler,
    text: Arc<Text>,
    formatter: Option<blaze_chat::ChatFormatter>,
    cfg: ServeConfig,
    max_context: usize,
    vocab: usize,
    /// structured outputs (None: the tokenizer could not be read for grammars)
    grammars: Option<Grammars>,
    /// the chat template knows no "developer" role (Qwen3.5/3.6 raise on it): such messages render as "system", the
    /// role OpenAI's developer messages take the place of
    developer_as_system: bool,
    /// `created` of /v1/models: when the server started
    started: u64,
    metrics: Metrics,
}

/// Request counters of /metrics (the scheduler's own are in `Scheduler::stats`).
#[derive(Default)]
struct Metrics {
    http_requests: AtomicU64,
    http_active: AtomicUsize,
    http_errors: AtomicU64,
    http_rejected: AtomicU64,
    http_timeouts: AtomicU64,
    http_latency: crate::telemetry::Histogram,
    requests: AtomicU64,
    prompt_tokens: AtomicU64,
    generation_tokens: AtomicU64,
    /// generations still open on the HTTP side
    open: AtomicUsize,
    /// generations dropped before their end (the client went away or a stop string ended them)
    dropped: AtomicU64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ApiError {
    message: String,
}

impl From<ProcessorError> for ApiError {
    fn from(e: ProcessorError) -> Self {
        Self { message: e.to_string() }
    }
}

/// An error in OpenAI's shape: {"error": {"message", "type", "param", "code"}}.
fn api_error(status: StatusCode, message: impl Into<String>, code: Option<&str>, param: Option<&str>) -> Response {
    let message = message.into();
    let kind = if status == StatusCode::UNAUTHORIZED { "authentication_error" } else if status == StatusCode::TOO_MANY_REQUESTS { "rate_limit_error" } else if status.is_server_error() { "server_error" } else { "invalid_request_error" };
    // every refused or failed request on stderr, one line (a client that only shows "400" leaves the reason here)
    let mut shown: String = message.chars().take(300).collect();
    if shown.len() < message.len() {
        shown.push_str("...");
    }
    crate::logging::event(if status.is_server_error() { crate::logging::Level::Error } else { crate::logging::Level::Warn },"api.error",
        json!({"request_id":crate::logging::request_trace(0).id,"status":status.as_u16(),"param":param,"message":shown}));
    (status, Json(json!({"error": {"message": message, "type": kind, "param": param, "code": code}}))).into_response()
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    api_error(status, message, None, None)
}

fn bad_param(param: &str, message: impl Into<String>) -> Response {
    api_error(StatusCode::BAD_REQUEST, message, None, Some(param))
}

/// The prompt does not fit the context (Hermes and opencode read this code to compact the conversation).
fn too_long(prompt_tokens: usize, max_context: usize) -> Response {
    api_error(
        StatusCode::BAD_REQUEST,
        format!(
            "This model's maximum context length is {max_context} tokens. However, your prompt contains {prompt_tokens} \
             tokens. Please reduce the length of the messages or the prompt."
        ),
        Some("context_length_exceeded"),
        Some("messages"),
    )
}

/// An error event of a stream, in the same shape.
fn stream_error(message: &str) -> String {
    json!({"error": {"message": message, "type": "server_error", "param": null, "code": null}}).to_string()
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn parse_body(bytes: &Bytes) -> std::result::Result<Value, Response> {
    match serde_json::from_slice::<Value>(bytes) {
        Ok(v) if v.is_object() => Ok(v),
        Ok(_) => Err(error(StatusCode::BAD_REQUEST, "the request body must be a JSON object")),
        Err(e) => Err(error(StatusCode::BAD_REQUEST, format!("invalid JSON in the request body: {e}"))),
    }
}

/// The HTTP port, open from the start of the load. Until `serve` hands it the API, every request except /version gets
/// 503 (`/health` and `/ping` too, so that health checks wait for the model) with the stage the load is in; a shutdown
/// signal during the load ends the process.
pub struct Early {
    app: Arc<std::sync::OnceLock<Router>>,
    serving: tokio::task::JoinHandle<std::io::Result<()>>,
    stop: tokio::sync::oneshot::Sender<()>,
    signal: tokio::sync::watch::Receiver<bool>,
    listen: std::net::SocketAddr,
    started: std::time::Instant,
}

pub async fn listen(host: &str, port: u16) -> Result<Early> {
    use anyhow::Context;
    use std::future::IntoFuture;
    use tower::ServiceExt;
    let listener = tokio::net::TcpListener::bind((host, port)).await.with_context(|| format!("listening on {host}:{port}"))?;
    let listen = listener.local_addr()?;
    let app: Arc<std::sync::OnceLock<Router>> = Arc::default();
    let started = std::time::Instant::now();
    let front = Router::new().fallback({
        let app = app.clone();
        move |req: Request| {
            let app = app.clone();
            async move {
                match app.get() {
                    Some(r) => r.clone().oneshot(req).await.unwrap_or_else(|e| match e {}),
                    None => loading(&req, started),
                }
            }
        }
    });
    let (stop, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(
        axum::serve(listener, front.into_make_service_with_connect_info::<std::net::SocketAddr>())
            .with_graceful_shutdown(async { let _ = stop_rx.await; })
            .into_future(),
    );
    let (signal_tx, signal) = tokio::sync::watch::channel(false);
    {
        let app = app.clone();
        tokio::spawn(async move {
            control::shutdown_signal().await;
            if app.get().is_none() {
                crate::logging::event(crate::logging::Level::Warn, "server.interrupted", json!({"message":"stopped during the load"}));
                std::process::exit(130);
            }
            let _ = signal_tx.send(true);
        });
    }
    crate::logging::event(crate::logging::Level::Info, "http.listening", json!({"message":format!(
        "listening on http://{listen}: until the model is ready, requests get 503 with the load's progress (GET /health)")}));
    Ok(Early { app, serving, stop, signal, listen, started })
}

/// The answer of a loading server: 503, the running stages and their progress, Retry-After.
fn loading(req: &Request, started: std::time::Instant) -> Response {
    if req.uri().path() == "/version" {
        return Json(json!({"version": env!("CARGO_PKG_VERSION")})).into_response();
    }
    let what = crate::logging::running_text().unwrap_or_else(|| "starting".into());
    let mut r = (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({"error": {"message": format!("the model is loading: {what}"), "type": "service_unavailable", "param": null,
                              "code": "model_loading"},
                    "status": "loading", "elapsed_s": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
                    "stages": crate::logging::running()})),
    )
        .into_response();
    r.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
    r
}

#[allow(clippy::too_many_arguments)]
pub async fn serve(early: Early, sched: Scheduler, model_dir: &std::path::Path, text: Text, grammars: Option<Grammars>, cfg: ServeConfig,
                   max_context: usize, vocab: usize, mm: MmConfig) -> Result<()> {
    let template = model_dir.join("chat_template.jinja");
    let config = model_dir.join("tokenizer_config.json");
    let formatter = blaze_chat::load_chat_formatter(
        config.exists().then(|| config.to_string_lossy().into_owned()).as_deref(),
        Some(&model_dir.to_string_lossy()),
        None,
        template.exists().then(|| template.to_string_lossy().into_owned()).as_deref(),
    )
    .map_err(|e| eprintln!("blaze: chat completions disabled: {e}"))
    .ok();
    // the template's text: chat_template.jinja, else tokenizer_config.json's chat_template (one or a list of named ones)
    let template_text = std::fs::read_to_string(&template).ok().or_else(|| {
        let c: Value = serde_json::from_str(&std::fs::read_to_string(&config).ok()?).ok()?;
        Some(c.get("chat_template")?.to_string())
    });
    let developer_as_system = template_text.is_some_and(|t| !t.contains("developer"));
    let body_limit = cfg.body_limit;
    let state = Arc::new(AppState {
        http_slots: Arc::new(tokio::sync::Semaphore::new(cfg.max_concurrent_requests)),
        key_hashes: cfg.api_keys.iter().map(|k| blake3::hash(k.as_bytes())).collect(),
        mm,
        sched,
        text: Arc::new(text),
        formatter,
        cfg,
        max_context,
        vocab,
        grammars,
        developer_as_system,
        started: now(),
        metrics: Metrics::default(),
    });
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/ping", get(|| async { "ok" }).post(|| async { "ok" }))
        .route("/version", get(version))
        .route("/metrics", get(metrics))
        .route("/server_info", get(control::info))
        .route("/get_server_info", get(control::info))
        .route("/v1/models", get(models))
        .route("/v1/models/{*id}", get(model))
        .route("/v1/chat/completions", post(chat::chat))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/embeddings", post(embeddings::embeddings))
        .route("/v1/responses", post(responses::create))
        .route("/v1/messages", post(messages::create))
        .route("/v1/messages/count_tokens", post(messages::count_tokens))
        .route("/v1/responses/{id}", get(responses::get_one))
        .route("/tokenize", post(tokenize::tokenize))
        .route("/detokenize", post(tokenize::detokenize))
        .fallback(|| async { api_error(StatusCode::NOT_FOUND, "no such endpoint", Some("not_found"), None) })
        .method_not_allowed_fallback(|| async { error(StatusCode::METHOD_NOT_ALLOWED, "method not allowed") })
        .layer(DefaultBodyLimit::max(body_limit))
        .layer(axum::middleware::from_fn_with_state(state.clone(), control::request))
        .with_state(state.clone());
    let Early { app: slot, mut serving, stop, mut signal, listen, started } = early;
    let _ = slot.set(app);
    let endpoints = ["/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/v1/embeddings"];
    if crate::logging::options().format == crate::logging::Format::Json {
        crate::logging::event(crate::logging::Level::Info,"server.ready",json!({"listen":listen.to_string(),
            "model":state.cfg.served_name,"max_model_len":max_context,"api_auth":!state.key_hashes.is_empty(),
            "load_s":started.elapsed().as_secs_f64(),
            "endpoints":["/v1/chat/completions","/v1/completions","/v1/responses","/v1/messages","/v1/embeddings","/metrics","/server_info"]}));
    } else {
        let mut rows = vec![
            ("API".to_string(), endpoints.join("  ")),
            ("service".to_string(), "/health  /metrics  /server_info  /v1/models  /tokenize  /detokenize  /version".to_string()),
            ("API keys".to_string(), if state.key_hashes.is_empty() { "none: any client".to_string() } else { format!("{} accepted", state.key_hashes.len()) }),
            ("structured outputs".to_string(), if state.grammars.is_some() { "on".into() } else { "off (no grammar compiler)".into() }),
            ("chat template".to_string(), if state.formatter.is_some() { "from the checkpoint".into() } else { "none: /v1/chat/completions disabled".into() }),
        ];
        let stages = {
            let mut f = crate::logging::finished();
            f.sort_by_key(|x| x.id);
            f
        };
        for st in stages.iter().filter(|st| st.depth <= 2) {
            let mut label: String = st.label.chars().take(60).collect();
            if st.label.chars().count() > 60 {
                label.pop();
                label.push('…');
            }
            rows.push((format!("{}{label}", "  ".repeat(st.depth + 1)), format!("{:>8}{}", crate::logging::dur(st.secs), if st.ok { "" } else { "  failed" })));
        }
        rows.insert(5, ("load".to_string(), format!("{:>8} in all", crate::logging::dur(started.elapsed().as_secs_f64()))));
        let host = if listen.ip().is_unspecified() { format!("<this host>:{}", listen.port()) } else { listen.to_string() };
        crate::logging::block(crate::logging::Level::Info, "server",
            &format!("\u{2714} ready in {} · http://{host}/v1 · model {} · context {max_context}",
                     crate::logging::dur(started.elapsed().as_secs_f64()), state.cfg.served_name), &rows);
    }
    tokio::select! {
        result = &mut serving => result??,
        _ = signal.wait_for(|stop| *stop) => {
            crate::logging::event(crate::logging::Level::Info, "server.shutdown", json!({"message":"draining active HTTP requests"}));
            let _ = stop.send(());
            match tokio::time::timeout(state.cfg.shutdown_timeout, &mut serving).await {
                Ok(result) => result??,
                Err(_) => crate::logging::event(crate::logging::Level::Warn,"server.shutdown_timeout",
                    json!({"active_http":state.metrics.http_active.load(Ordering::Relaxed),"drain_seconds":state.cfg.shutdown_timeout.as_secs_f64()})),
            }
        }
    }
    Ok(())
}

async fn version() -> Json<Value> {
    Json(json!({"version": env!("CARGO_PKG_VERSION")}))
}

/// Prometheus text: the counters of the HTTP side and the scheduler's gauges.
async fn metrics(State(st): State<Arc<AppState>>) -> Response {
    let m = &st.metrics;
    let s = &st.sched.stats;
    let rows = [
        ("blaze_http_requests_total", "counter", "HTTP requests excluding liveness probes", m.http_requests.load(Ordering::Relaxed) as f64),
        ("blaze_http_requests_active", "gauge", "Active HTTP requests including streaming bodies", m.http_active.load(Ordering::Relaxed) as f64),
        ("blaze_http_errors_total", "counter", "HTTP server errors", m.http_errors.load(Ordering::Relaxed) as f64),
        ("blaze_http_rejected_total", "counter", "HTTP authentication or admission rejections", m.http_rejected.load(Ordering::Relaxed) as f64),
        ("blaze_http_timeouts_total", "counter", "HTTP requests exceeding their deadline", m.http_timeouts.load(Ordering::Relaxed) as f64),
        ("blaze_requests_outstanding", "gauge", "Generations admitted including intake and preempted requests", s.outstanding.load(Ordering::Relaxed) as f64),
        ("blaze_requests_completed_total", "counter", "Generations retired by the scheduler", s.completed.load(Ordering::Relaxed) as f64),
        ("blaze_prefill_tokens_total", "counter", "Prompt tokens actually computed including recompute", s.prefill_tokens.load(Ordering::Relaxed) as f64),
        ("blaze_cached_tokens_total", "counter", "Prompt tokens restored from the prefix cache", s.cached_tokens.load(Ordering::Relaxed) as f64),
        ("blaze_scheduler_generation_tokens_total", "counter", "Output tokens emitted live by the scheduler", s.generation_tokens.load(Ordering::Relaxed) as f64),
        ("blaze_prefill_seconds_total", "counter", "Time of scheduler steps containing prefill", s.prefill_micros.load(Ordering::Relaxed) as f64 / 1e6),
        ("blaze_decode_seconds_total", "counter", "Time of scheduler steps containing only decode", s.decode_micros.load(Ordering::Relaxed) as f64 / 1e6),
        ("blaze_kv_pages_free", "gauge", "Available KV pages", s.kv_pages_free.load(Ordering::Relaxed) as f64),
        ("blaze_kv_pages_total", "gauge", "KV pool capacity in pages", s.kv_pages_total.load(Ordering::Relaxed) as f64),
        ("blaze_prefix_cache_entries", "gauge", "Reusable prefix snapshots", s.cache_entries.load(Ordering::Relaxed) as f64),
        ("blaze_speculative_rounds_total", "counter", "Speculative verify rounds", s.spec_rounds.load(Ordering::Relaxed) as f64),
        ("blaze_speculative_tokens_total", "counter", "Output tokens emitted by speculative rounds", s.spec_emitted.load(Ordering::Relaxed) as f64),
        ("blaze_requests_total", "counter", "Generations started", m.requests.load(Ordering::Relaxed) as f64),
        ("blaze_prompt_tokens_total", "counter", "Prompt tokens of finished or dropped generations", m.prompt_tokens.load(Ordering::Relaxed) as f64),
        ("blaze_generation_tokens_total", "counter", "Generated tokens", m.generation_tokens.load(Ordering::Relaxed) as f64),
        ("blaze_requests_dropped_total", "counter", "Generations dropped before their end (client gone or stop string)", m.dropped.load(Ordering::Relaxed) as f64),
        ("blaze_requests_cancelled_total", "counter", "Requests the scheduler dropped because their client was gone", s.cancelled.load(Ordering::Relaxed) as f64),
        ("blaze_requests_kv_full_total", "counter", "Requests ended early because the KV pool could not hold them", s.kv_full.load(Ordering::Relaxed) as f64),
        ("blaze_requests_preempted_total", "counter", "Requests that gave their KV pages up and resumed later", s.preempted.load(Ordering::Relaxed) as f64),
        ("blaze_requests_open", "gauge", "Generations open on the HTTP side", m.open.load(Ordering::Relaxed) as f64),
        ("blaze_requests_running", "gauge", "Requests prefilling or decoding", s.running.load(Ordering::Relaxed) as f64),
        ("blaze_requests_waiting", "gauge", "Requests waiting for a free row or KV room", s.waiting.load(Ordering::Relaxed) as f64),
    ];
    let mut out = String::new();
    for (name, kind, help, v) in rows {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}\n"));
    }
    for (h,name,help) in [
        (&s.ttft,"blaze_time_to_first_token_seconds","Time from HTTP arrival to first generated token, including queue"),
        (&s.queue_time,"blaze_queue_time_seconds","Time from HTTP arrival to scheduler admission"),
        (&s.request_time,"blaze_request_duration_seconds","Generation duration including queue and prefill"),
        (&s.inter_token,"blaze_inter_token_seconds","Time between emitted tokens; speculative bursts share a round"),
        (&s.prefill_step,"blaze_prefill_step_seconds","Scheduler step durations containing prefill"),
        (&s.decode_step,"blaze_decode_step_seconds","Decode and speculative step durations"),
        (&m.http_latency,"blaze_http_duration_seconds","HTTP duration through response-body completion or disconnect"),
    ] { h.render(name,help,&mut out); }
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], out).into_response()
}

fn model_card(st: &AppState) -> Value {
    json!({"id": st.cfg.served_name, "object": "model", "created": st.started, "owned_by": "blaze", "root": st.cfg.served_name,
           "parent": null, "max_model_len": st.max_context, "permission": []})
}

async fn models(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [model_card(&st)]}))
}

async fn model(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    if id != st.cfg.served_name {
        return api_error(StatusCode::NOT_FOUND, format!("The model '{id}' does not exist"), Some("model_not_found"), Some("model"));
    }
    Json(model_card(&st)).into_response()
}

// request fields

type Checked<T> = std::result::Result<T, Response>;

/// Numeric fields out of range: what the engines would reject (or silently turn into garbage), and token ids that are
/// not in the vocabulary (the GPU would index outside its tables).
fn check_ranges(st: &AppState, body: &Value) -> Checked<()> {
    crate::service::validate_sampling(body).map_err(|e| bad_param(e.field, e.message))?;
    if let Some(ids)=body.get("stop_token_ids").filter(|v|!v.is_null()) {
        if ids.as_array().is_none_or(|a|a.iter().any(|t|t.as_u64().is_none())) {
            return Err(bad_param("stop_token_ids","stop_token_ids must be an array of nonnegative token IDs"));
        }
    }
    if let Some(bias)=body.get("logit_bias").filter(|v|!v.is_null()) {
        if bias.as_object().is_none_or(|m|m.iter().any(|(k,v)|k.parse::<u32>().is_err() || v.as_f64().is_none_or(|v|!v.is_finite() || !(-100.0..=100.0).contains(&v)))) {
            return Err(bad_param("logit_bias","logit_bias must map token IDs to finite numbers in [-100,100]"));
        }
    }
    let num = |k: &str| body.get(k).and_then(Value::as_f64);
    if num("temperature").is_some_and(|t| t < 0.0) {
        return Err(bad_param("temperature", "temperature must be at least 0"));
    }
    if num("top_p").is_some_and(|p| p <= 0.0 || p > 1.0) {
        return Err(bad_param("top_p", "top_p must be in (0, 1]"));
    }
    for k in ["presence_penalty", "frequency_penalty"] {
        if num(k).is_some_and(|p| !(-2.0..=2.0).contains(&p)) {
            return Err(bad_param(k, format!("{k} must be in [-2, 2]")));
        }
    }
    if num("repetition_penalty").is_some_and(|p| p <= 0.0) {
        return Err(bad_param("repetition_penalty", "repetition_penalty must be positive"));
    }
    if body.get("min_p").and_then(Value::as_f64).is_some_and(|p| !(0.0..=1.0).contains(&p)) {
        return Err(bad_param("min_p", "min_p must be in [0, 1]"));
    }
    let n = body.get("n").and_then(Value::as_u64);
    if body.get("n").is_some_and(|v| !v.is_null()) && n.is_none_or(|n| n == 0 || n > MAX_N) {
        return Err(bad_param("n", format!("n must be an integer between 1 and {MAX_N}")));
    }
    for k in ["max_tokens", "max_completion_tokens"] {
        if body.get(k).is_some_and(|v| !v.is_null() && v.as_u64().is_none_or(|x| x < 1 || x > usize::MAX as u64)) {
            return Err(bad_param(k, format!("{k} must be an integer of at least 1")));
        }
    }
    let requested_max=body.get("max_completion_tokens").filter(|v|!v.is_null()).or(body.get("max_tokens")).and_then(Value::as_u64);
    if requested_max.is_some_and(|max|body.get("min_tokens").and_then(Value::as_u64).is_some_and(|min|min>max)) {
        return Err(bad_param("min_tokens","min_tokens exceeds the requested output token limit"));
    }
    if body.get("top_logprobs").is_some_and(|v| !v.is_null() && v.as_u64().is_none_or(|k| k > MAX_TOP_LOGPROBS)) {
        return Err(bad_param("top_logprobs", format!("top_logprobs must be an integer in 0..={MAX_TOP_LOGPROBS}")));
    }
    if matches!(body.get("logprobs"), Some(Value::Number(k)) if k.as_u64().is_none_or(|k| k > MAX_TOP_LOGPROBS)) {
        return Err(bad_param("logprobs", format!("logprobs is at most {MAX_TOP_LOGPROBS}")));
    }
    if body.get("logprobs").is_some_and(|v|!v.is_null() && !v.is_boolean() && !v.is_number()) {
        return Err(bad_param("logprobs","logprobs must be a boolean or an integer"));
    }
    for key in ["stream", "ignore_eos", "skip_special_tokens", "include_stop_str_in_output", "no_stop_trim", "echo"] {
        if body.get(key).is_some_and(|v|!v.is_null() && !v.is_boolean()) {
            return Err(bad_param(key,format!("{key} must be a boolean")));
        }
    }
    let outside = |mut ids: Box<dyn Iterator<Item = i64> + '_>| ids.find(|&t| t < 0 || t as usize >= st.vocab);
    if let Some(bad) = body.get("stop_token_ids").and_then(Value::as_array).and_then(|a|a.iter().filter_map(Value::as_u64).find(|&id|id>=st.vocab as u64)) {
        return Err(bad_param("stop_token_ids", format!("token id {bad} is outside the vocabulary of {}", st.vocab)));
    }
    if let Some(bad) = body.get("logit_bias").and_then(Value::as_object).and_then(|m| outside(Box::new(m.keys().filter_map(|k| k.parse().ok())))) {
        return Err(bad_param("logit_bias", format!("token id {bad} is outside the vocabulary of {}", st.vocab)));
    }
    let stops = stop_strings(body);
    if stops.len() > MAX_STOPS || stops.iter().any(|s| s.len() > MAX_STOP_BYTES) {
        return Err(bad_param("stop", format!("at most {MAX_STOPS} stop strings of at most {MAX_STOP_BYTES} bytes each")));
    }
    for k in ["stop_token_ids", "bad_words"] {
        if body.get(k).and_then(Value::as_array).is_some_and(|a| a.len() > MAX_STOP_IDS) {
            return Err(bad_param(k, format!("{k} has at most {MAX_STOP_IDS} entries")));
        }
    }
    Ok(())
}

fn n_of(body: &Value) -> usize {
    body.get("n").and_then(Value::as_u64).unwrap_or(1) as usize
}

/// Sampling parameters of a request body: the OpenAI fields, vLLM / SGLang extras (top_k, min_p, repetition_penalty,
/// min_tokens, bad_words), server defaults.
fn sampling_of(body: &Value, d: &SamplingParams, text: &Text) -> Checked<SamplingParams> {
    let pipeline = body.get("samplers").filter(|v| !v.is_null())
        .map(crate::sampler::SamplingPipeline::from_value).transpose()
        .map_err(|e| bad_param("samplers", e.to_string()))?.unwrap_or_else(|| d.samplers.clone());
    crate::service::validate_pipeline_fields(body, &pipeline).map_err(|e| bad_param("samplers", e.to_string()))?;
    let f = |k: &str| body.get(k).and_then(Value::as_f64).map(|x| x as f32);
    let logit_bias = body
        .get("logit_bias")
        .and_then(Value::as_object)
        .map(|m| m.iter().filter_map(|(k, v)| Some((k.parse::<i32>().ok()?, v.as_f64()? as f32))).collect())
        .unwrap_or_default();
    let bad_words = body
        .get("bad_words")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .flat_map(|w| [w.to_string(), format!(" {w}")])
                .filter_map(|w| text.encode(&w).ok())
                .filter(|t| !t.is_empty())
                .map(|t| t.into_iter().map(|x| x as i32).collect())
                .collect()
        })
        .unwrap_or_default();
    // chat: logprobs (bool) + top_logprobs; completions: logprobs (int)
    let logprobs = match body.get("logprobs") {
        Some(Value::Bool(true)) => Some(body.get("top_logprobs").and_then(Value::as_u64).unwrap_or(0) as usize),
        Some(Value::Number(n)) => n.as_u64().map(|x| x as usize),
        _ => None,
    };
    Ok(SamplingParams {
        samplers: pipeline,
        top_nsigma: f("top_nsigma").unwrap_or(d.top_nsigma),
        top_h: f("top_h").unwrap_or(d.top_h),
        temperature: f("temperature").unwrap_or(d.temperature),
        top_p: f("top_p").unwrap_or(d.top_p),
        top_k: body.get("top_k").and_then(Value::as_i64).map(|x| x as i32).unwrap_or(d.top_k),
        min_p: f("min_p").unwrap_or(d.min_p),
        presence_penalty: f("presence_penalty").unwrap_or(d.presence_penalty),
        frequency_penalty: f("frequency_penalty").unwrap_or(d.frequency_penalty),
        seed: body.get("seed").and_then(Value::as_u64).or(d.seed),
        repetition_penalty: f("repetition_penalty").unwrap_or(d.repetition_penalty),
        logit_bias,
        min_tokens: body.get("min_tokens").and_then(Value::as_u64).map_or(d.min_tokens, |x| x as usize),
        bad_words,
        logprobs,
    })
}

/// Stop tokens of a request: the model's EOS (unless ignore_eos) and stop_token_ids.
fn stop_ids_of(body: &Value, text: &Text) -> Vec<i32> {
    let mut ids: Vec<i32> = if body.get("ignore_eos").and_then(Value::as_bool).unwrap_or(false) {
        Vec::new()
    } else {
        text.eos.iter().map(|&e| e as i32).collect()
    };
    if let Some(a) = body.get("stop_token_ids").and_then(Value::as_array) {
        ids.extend(a.iter().filter_map(Value::as_i64).map(|x| x as i32));
    }
    ids
}

fn stop_strings(body: &Value) -> Vec<String> {
    match body.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        _ => Vec::new(),
    }
}

/// What the scheduler is asked for apart from the prompt.
struct Spec {
    params: SamplingParams,
    max_tokens: usize,
    stops: Vec<String>,
    /// vLLM's include_stop_str_in_output (SGLang: no_stop_trim)
    include_stop: bool,
    /// vLLM / SGLang skip_special_tokens: special tokens are left out of the text (default) or written
    skip_special: bool,
    stop_ids: Vec<i32>,
    think: Option<ThinkBudget>,
    /// the prompt's images (their rows in the token list are prefix-cache keys)
    mm: Option<Arc<crate::mm::MmInput>>,
}

impl Spec {
    /// `instruct`: a chat prompt with thinking off (its own defaults, --instruct-sampling)
    fn of(st: &AppState, body: &Value, max_tokens: usize, think: Option<ThinkBudget>, instruct: bool) -> Checked<Self> {
        let defaults = st.cfg.instruct_defaults.as_ref().filter(|_| instruct).unwrap_or(&st.cfg.defaults);
        Ok(Self {
            params: sampling_of(body, defaults, &st.text)?,
            max_tokens,
            stops: stop_strings(body),
            include_stop: ["include_stop_str_in_output", "no_stop_trim"].iter().any(|k| body.get(*k).and_then(Value::as_bool) == Some(true)),
            skip_special: body.get("skip_special_tokens").and_then(Value::as_bool).unwrap_or(true),
            stop_ids: stop_ids_of(body, &st.text),
            think,
            mm: None,
        })
    }
}

/// The request's thinking budget: vLLM's thinking_token_budget, SGLang-style thinking_budget, or the server default;
/// the forced ending follows Qwen's recipe for budgeted thinking.
fn think_budget(st: &AppState, body: &Value, starts_inside: bool) -> Option<ThinkBudget> {
    let budget = ["thinking_token_budget", "thinking_budget"]
        .iter()
        .find_map(|k| body.get(*k).and_then(Value::as_u64))
        .map(|x| x as usize)
        .or(st.cfg.thinking_budget)?;
    let start_id = st.text.token_id("<think>")? as i32;
    let end_id = st.text.token_id("</think>")? as i32;
    let mut force: Vec<i32> = st
        .text
        .encode("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n")
        .ok()?
        .into_iter()
        .map(|t| t as i32)
        .collect();
    force.push(end_id);
    force.extend(st.text.encode("\n\n").ok()?.into_iter().map(|t| t as i32));
    Some(ThinkBudget { budget, start_id, end_id, starts_inside, force })
}

/// The request's grammar as a scheduler constraint (applied after `after` when the output opens with reasoning).
/// `forced`: a grammar of the server's own (tool_choice) used when the request asks for none.
fn constraint_of(st: &AppState, body: &Value, after: Option<i32>, forced: Option<&str>) -> std::result::Result<Option<Constraint>, String> {
    use llguidance::api::TopLevelGrammar;
    let g = match grammar_of(body, st.cfg.grammar_whitespace).map_err(|e| e.to_string())? {
        Some(g) => g,
        None => match forced {
            Some(lark) => TopLevelGrammar::from_lark_or_grammar_list(lark).map_err(|e| e.to_string())?,
            None => return Ok(None),
        },
    };
    let gr = st.grammars.as_ref().ok_or("structured outputs are unavailable for this tokenizer")?;
    let matcher = gr.matcher(g).map_err(|e| e.to_string())?;
    Ok(Some(Constraint { matcher, after }))
}

// generation

/// Cuts stop strings out of a text stream. The tail that may still grow into a stop string is held back, so a stop
/// string that spans chunks never reaches the client in part.
struct StopCutter {
    stops: Vec<String>,
    /// KMP prefix function of every stop string (its longest proper border at each length)
    borders: Vec<Vec<usize>>,
    include: bool,
    held: String,
    stopped: bool,
    /// the stop string that ended the text
    matched: Option<String>,
}

impl StopCutter {
    fn new(stops: Vec<String>, include: bool) -> Self {
        let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let borders = stops.iter().map(|s| borders(s.as_bytes())).collect();
        Self { stops, borders, include, held: String::new(), stopped: false, matched: None }
    }

    /// Text that is safe to send after `delta`.
    fn push(&mut self, delta: &str) -> String {
        if self.stopped {
            return String::new();
        }
        if self.stops.is_empty() {
            return delta.to_string();
        }
        self.held.push_str(delta);
        // the earliest stop string of the held text (it may begin in an earlier chunk)
        let hit = self.stops.iter().enumerate().filter_map(|(j, s)| self.held.find(s.as_str()).map(|i| (i, s.len(), j)))
            .min_by_key(|&(i, l, _)| (i, l));
        if let Some((at, len, j)) = hit {
            self.stopped = true;
            self.matched = Some(self.stops[j].clone());
            let end = if self.include { at + len } else { at };
            let out = self.held[..end].to_string();
            self.held.clear();
            return out;
        }
        let keep = self.stops.iter().zip(&self.borders).map(|(s, b)| overlap(&self.held, s, b)).max().unwrap_or(0);
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        out
    }

    /// The held-back tail when the output ends without the stop string completing.
    fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// KMP prefix function: for each length l of `p`, the longest proper border of p[..l] (index l - 1).
fn borders(p: &[u8]) -> Vec<usize> {
    let mut b = vec![0; p.len()];
    let mut k = 0;
    for i in 1..p.len() {
        while k > 0 && p[i] != p[k] {
            k = b[k - 1];
        }
        if p[i] == p[k] {
            k += 1;
        }
        b[i] = k;
    }
    b
}

/// Bytes of the longest suffix of `text` that is a proper prefix of `stop`, on a character boundary of both (the KMP
/// automaton of `stop` run over the last bytes of `text`: linear in the stop's length).
fn overlap(text: &str, stop: &str, border: &[usize]) -> usize {
    let (t, p) = (text.as_bytes(), stop.as_bytes());
    let mut k = 0;
    for &c in &t[t.len().saturating_sub(p.len())..] {
        while k > 0 && (k == p.len() || c != p[k]) {
            k = border[k - 1];
        }
        if c == p[k] {
            k += 1;
        }
    }
    // a whole match is found by the caller's search; a border that splits a character is no prefix to hold
    while k > 0 && (k == p.len() || !stop.is_char_boundary(k) || !text.is_char_boundary(t.len() - k)) {
        k = border[k - 1];
    }
    k
}

/// Text of the generated tokens, emitted as soon as it is valid UTF-8, with stop strings cut.
struct Detok {
    text: Arc<Text>,
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
    skip_special: bool,
    cutter: StopCutter,
}

impl Detok {
    fn new(text: Arc<Text>, spec: &Spec) -> Self {
        Self { text, ids: Vec::new(), prefix: 0, read: 0, skip_special: spec.skip_special,
               cutter: StopCutter::new(spec.stops.clone(), spec.include_stop) }
    }

    fn decode(&self, ids: &[u32]) -> String {
        let text = if self.skip_special { self.text.decode_skip(ids) } else { self.text.decode(ids) };
        text.unwrap_or_default()
    }

    fn stopped(&self) -> bool {
        self.cutter.stopped
    }

    /// New text after `id` (empty while a character is incomplete or a stop string may be coming).
    fn push(&mut self, id: u32) -> String {
        if self.cutter.stopped {
            return String::new();
        }
        self.ids.push(id);
        let before = self.decode(&self.ids[self.prefix..self.read]);
        let after = self.decode(&self.ids[self.prefix..]);
        if after.len() <= before.len() || after.ends_with('\u{FFFD}') || !after.is_char_boundary(before.len()) {
            return String::new();
        }
        let delta = after[before.len()..].to_string();
        self.prefix = self.read;
        self.read = self.ids.len();
        self.cutter.push(&delta)
    }
}

/// One generation's events as text: a piece is some text with the logprobs of the tokens behind it, or the end.
enum Piece {
    Text(String, Vec<(u32, TokenLogprob)>),
    Finish(Finish),
    Error(String),
}

/// One generation on the scheduler: its events, detokenized.
struct Gen {
    rx: tokio::sync::mpsc::UnboundedReceiver<GenEvent>,
    cancel: Arc<AtomicBool>,
    detok: Detok,
    /// logprobs of the tokens whose text has not been sent yet
    logprobs: Vec<(u32, TokenLogprob)>,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    /// tokens that end the generation: their text is not part of the output unless the stop string is asked for (vLLM)
    stop_ids: Vec<i32>,
    keep_stop: bool,
    /// the end, after the held-back text
    queued: Option<Piece>,
    done: bool,
    st: Arc<AppState>,
}

impl Gen {
    /// The stop string that ended the generation, or null (vLLM's stop_reason).
    fn stop_reason(&self) -> Value {
        self.detok.cutter.matched.clone().map_or(Value::Null, Value::from)
    }

    async fn next(&mut self) -> Option<Piece> {
        if let Some(p) = self.queued.take() {
            return Some(p);
        }
        if self.done {
            return None;
        }
        loop {
            let Some(ev) = self.rx.recv().await else {
                self.done = true;
                return Some(Piece::Error("generation ended unexpectedly".into()));
            };
            match ev {
                GenEvent::Started { prompt_tokens, cached_tokens } => {
                    self.prompt_tokens = prompt_tokens;
                    self.cached_tokens = cached_tokens;
                }
                GenEvent::Embedding(_) => {}
                GenEvent::Token(t, lp) => {
                    self.completion_tokens += 1;
                    if let Some(lp) = lp {
                        self.logprobs.push((t as u32, lp));
                    }
                    if !self.keep_stop && self.stop_ids.contains(&t) {
                        continue;
                    }
                    let d = self.detok.push(t as u32);
                    if self.detok.stopped() {
                        self.cancel.store(true, Ordering::Relaxed);
                    }
                    if !d.is_empty() {
                        return Some(Piece::Text(d, std::mem::take(&mut self.logprobs)));
                    }
                }
                GenEvent::Done(f) => {
                    self.done = true;
                    let f = if self.detok.stopped() { Finish::Stop } else { f };
                    let tail = self.detok.cutter.flush();
                    if tail.is_empty() {
                        return Some(Piece::Finish(f));
                    }
                    self.queued = Some(Piece::Finish(f));
                    return Some(Piece::Text(tail, std::mem::take(&mut self.logprobs)));
                }
                GenEvent::Error(e) => {
                    self.done = true;
                    return Some(Piece::Error(e));
                }
            }
        }
    }
}

impl Drop for Gen {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        let m = &self.st.metrics;
        m.open.fetch_sub(1, Ordering::Relaxed);
        if !self.done {
            m.dropped.fetch_add(1, Ordering::Relaxed);
        }
        m.prompt_tokens.fetch_add(self.prompt_tokens as u64, Ordering::Relaxed);
        m.generation_tokens.fetch_add(self.completion_tokens as u64, Ordering::Relaxed);
    }
}

/// Token counts of one API request, summed over its generations.
struct Totals {
    prompt: usize,
    cached: AtomicUsize,
    completion: AtomicUsize,
    /// the stop string that ended each choice
    stops: std::sync::Mutex<std::collections::HashMap<usize, String>>,
}

impl Totals {
    fn new(prompt: usize) -> Arc<Self> {
        Arc::new(Self { prompt, cached: AtomicUsize::new(0), completion: AtomicUsize::new(0), stops: Default::default() })
    }

    fn note_stop(&self, choice: usize, g: &Gen) {
        if let Some(s) = g.stop_reason().as_str() {
            self.stops.lock().unwrap().insert(choice, s.to_string());
        }
    }

    fn stop_of(&self, choice: usize) -> Value {
        self.stops.lock().unwrap().get(&choice).map_or(Value::Null, |s| json!(s))
    }

    /// Fold in what `g` has produced since `seen` tokens.
    fn note(&self, g: &Gen, seen: &mut usize) {
        self.cached.fetch_max(g.cached_tokens, Ordering::Relaxed);
        self.completion.fetch_add(g.completion_tokens - *seen, Ordering::Relaxed);
        *seen = g.completion_tokens;
    }

    fn usage(&self) -> Value {
        let c = self.completion.load(Ordering::Relaxed);
        json!({"prompt_tokens": self.prompt, "completion_tokens": c, "total_tokens": self.prompt + c,
               "prompt_tokens_details": {"cached_tokens": self.cached.load(Ordering::Relaxed)}})
    }
}

/// Submit one generation. `choice` (0..n) offsets the seed so that the choices of a seeded request differ.
fn submit(st: &Arc<AppState>, tokens: &[i64], spec: &Spec, choice: usize, constraint: Option<Constraint>) -> Checked<Gen> {
    let (tx, rx) = unbounded_channel();
    let cancel = Arc::new(AtomicBool::new(false));
    // Prompt-cache point before the last turn header: the next turn of a chat repeats everything up to it.
    let cache_points = st
        .text
        .token_id("<|im_start|>")
        .and_then(|h| tokens.iter().rposition(|&t| t == h as i64))
        .filter(|&p| p > 0)
        .into_iter()
        .collect();
    let mut params = spec.params.clone();
    params.seed = params.seed.map(|s| s.wrapping_add(choice as u64));
    let mut req = GenRequest::new(tokens.to_vec(), params, tx, cancel.clone());
    req.trace = crate::logging::request_trace(choice);
    req.max_tokens = spec.max_tokens.min(st.max_context.saturating_sub(tokens.len()));
    req.stop_ids = spec.stop_ids.clone();
    req.cache_points = cache_points;
    req.think = spec.think.clone();
    req.constraint = constraint;
    req.mm = spec.mm.clone();
    st.sched.submit(req).map_err(control::submit_error)?;
    st.metrics.requests.fetch_add(1, Ordering::Relaxed);
    st.metrics.open.fetch_add(1, Ordering::Relaxed);
    Ok(Gen {
        rx,
        cancel,
        detok: Detok::new(st.text.clone(), spec),
        logprobs: Vec::new(),
        prompt_tokens: 0,
        cached_tokens: 0,
        completion_tokens: 0,
        stop_ids: spec.stop_ids.clone(),
        keep_stop: spec.include_stop,
        queued: None,
        done: false,
        st: st.clone(),
    })
}

/// The `n` generations of one prompt; each gets its own grammar matcher from `constraint`.
fn submit_n(st: &Arc<AppState>, tokens: &[i64], spec: &Spec, n: usize, first_choice: usize,
            constraint: &dyn Fn() -> std::result::Result<Option<Constraint>, String>) -> Checked<Vec<Gen>> {
    (0..n)
        .map(|i| {
            let c = constraint().map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
            submit(st, tokens, spec, first_choice + i, c)
        })
        .collect()
}

fn sse(s: impl Stream<Item = String> + Send + 'static) -> Response {
    let mut res = Sse::new(s.map(|d| Ok::<_, Infallible>(Event::default().data(d)))).keep_alive(KeepAlive::default()).into_response();
    // proxies must not hold the stream back
    res.headers_mut().insert("x-accel-buffering", HeaderValue::from_static("no"));
    res
}

/// Every prompt token must have a place in the context, with room for one more.
fn check_fits(st: &AppState, prompt_tokens: usize) -> Checked<()> {
    if prompt_tokens + 1 > st.max_context {
        return Err(too_long(prompt_tokens, st.max_context));
    }
    Ok(())
}

/// vLLM's truncate_prompt_tokens: keep the last k prompt tokens (-1: the context's size).
fn truncate_prompt(st: &AppState, body: &Value, tokens: &mut Vec<i64>) {
    let k = match body.get("truncate_prompt_tokens").and_then(Value::as_i64) {
        Some(-1) => st.max_context.saturating_sub(1),
        Some(k) if k > 0 => k as usize,
        _ => return,
    };
    if tokens.len() > k {
        tokens.drain(..tokens.len() - k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get(url: String) -> (u16, String) {
        match ureq::get(&url).call() {
            Ok(r) => (r.status(), r.into_string().unwrap()),
            Err(ureq::Error::Status(code, r)) => (code, r.into_string().unwrap()),
            Err(e) => panic!("{url}: {e}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_port_answers_during_the_load_and_hands_over_to_the_api() {
        let early = listen("127.0.0.1", 0).await.unwrap();
        let base = format!("http://{}", early.listen);
        let stage = crate::logging::stage("test: reading weights").total(10.0, "layers");
        stage.progress(4.0);
        let b = base.clone();
        let (status, body) = tokio::task::spawn_blocking(move || get(format!("{b}/health"))).await.unwrap();
        assert_eq!(status, 503);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["code"], "model_loading");
        assert!(v["error"]["message"].as_str().unwrap().contains("test: reading weights (40% · 4/10 layers"), "{v}");
        assert!(v["stages"].as_array().unwrap().iter().any(|s| s["stage"] == "test: reading weights" && s["progress_pct"] == 40.0), "{v}");
        let b = base.clone();
        let (status, _) = tokio::task::spawn_blocking(move || get(format!("{b}/version"))).await.unwrap();
        assert_eq!(status, 200);
        stage.done("");
        // the API in place: requests reach it through the same port
        let _ = early.app.set(Router::new().route("/health", axum::routing::get(|| async { "ok" })));
        let (status, body) = tokio::task::spawn_blocking(move || get(format!("{base}/health"))).await.unwrap();
        assert_eq!((status, body.as_str()), (200, "ok"));
    }

    fn cut(stops: &[&str], include: bool, chunks: &[&str]) -> (String, bool) {
        let mut c = StopCutter::new(stops.iter().map(|s| s.to_string()).collect(), include);
        let mut out = String::new();
        for ch in chunks {
            out.push_str(&c.push(ch));
        }
        if !c.stopped {
            out.push_str(&c.flush());
        }
        (out, c.stopped)
    }

    #[test]
    fn overlap_matches_the_quadratic_search() {
        let brute = |text: &str, stop: &str| {
            (1..stop.len()).rev().filter(|&k| stop.is_char_boundary(k)).find(|&k| text.ends_with(&stop[..k])).unwrap_or(0)
        };
        let mut x = 7u64;
        let alphabet = ["a", "b", "ab", "é", "日"];
        let mut next = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 33) as usize
        };
        for _ in 0..3000 {
            let (ls, lt) = (1 + next() % 6, next() % 9);
            let stop: String = (0..ls).map(|_| alphabet[next() % alphabet.len()]).collect();
            let text: String = (0..lt).map(|_| alphabet[next() % alphabet.len()]).collect();
            if stop.is_empty() {
                continue;
            }
            assert_eq!(overlap(&text, &stop, &borders(stop.as_bytes())), brute(&text, &stop), "text {text:?} stop {stop:?}");
        }
    }

    #[test]
    fn stop_string_inside_a_chunk() {
        assert_eq!(cut(&["END"], false, &["abc ", "xENDy", "more"]), ("abc x".into(), true));
    }

    #[test]
    fn stop_string_across_chunks_is_held_back() {
        assert_eq!(cut(&["STOP"], false, &["go ST", "O", "P tail"]), ("go ".into(), true));
        assert_eq!(cut(&["STOP"], true, &["go ST", "O", "P tail"]), ("go STOP".into(), true));
    }

    #[test]
    fn near_miss_is_released() {
        assert_eq!(cut(&["STOP"], false, &["a ST", "ONE"]), ("a STONE".into(), false));
        assert_eq!(cut(&["STOP"], false, &["a ST"]), ("a ST".into(), false));
    }

    #[test]
    fn earliest_of_several_and_multibyte() {
        assert_eq!(cut(&["zz", "b"], false, &["aab", "zz"]), ("aa".into(), true));
        assert_eq!(cut(&["日本語"], false, &["x日", "本", "語y"]), ("x".into(), true));
        assert_eq!(cut(&["日本語"], false, &["x日", "本x"]), ("x日本x".into(), false));
    }

    #[test]
    fn no_stops_pass_through() {
        assert_eq!(cut(&[], false, &["a", "b"]), ("ab".into(), false));
    }
}
