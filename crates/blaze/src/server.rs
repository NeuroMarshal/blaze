//! HTTP inference API with streaming, tool calls and structured outputs.
//! The vendored blaze-chat renders chat templates and parses model output.

mod chat;
mod completions;
mod tokenize;

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
const BODY_LIMIT: usize = 512 << 20;
/// Most choices of one request (`n`); the scheduler queues what does not fit its rows at once.
const MAX_N: u64 = 64;
/// Most alternatives per token (`top_logprobs`, completions `logprobs`), as vLLM's default max_logprobs.
const MAX_TOP_LOGPROBS: u64 = 20;

pub struct ServeConfig {
    pub served_name: String,
    pub host: String,
    pub port: u16,
    pub defaults: SamplingParams,
    pub default_max_tokens: usize,
    pub tool_parser: Option<String>,
    pub reasoning_parser: Option<String>,
    /// default thinking budget of chat requests (None: unlimited)
    pub thinking_budget: Option<usize>,
}

struct AppState {
    sched: Scheduler,
    text: Arc<Text>,
    formatter: Option<blaze_chat::ChatFormatter>,
    cfg: ServeConfig,
    max_context: usize,
    vocab: usize,
    /// structured outputs (None: the tokenizer could not be read for grammars)
    grammars: Option<Grammars>,
    /// `created` of /v1/models: when the server started
    started: u64,
    metrics: Metrics,
}

/// Request counters of /metrics (the scheduler's own are in `Scheduler::stats`).
#[derive(Default)]
struct Metrics {
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
    let kind = if status.is_server_error() { "server_error" } else { "invalid_request_error" };
    (status, Json(json!({"error": {"message": message.into(), "type": kind, "param": param, "code": code}}))).into_response()
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

pub async fn serve(sched: Scheduler, model_dir: &std::path::Path, text: Text, cfg: ServeConfig, max_context: usize,
                   vocab: usize) -> Result<()> {
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
    let t0 = std::time::Instant::now();
    let grammars = Grammars::load(&model_dir.join("tokenizer.json"), &text.eos, vocab)
        .map_err(|e| eprintln!("blaze: structured outputs disabled: {e}"))
        .ok();
    if grammars.is_some() {
        eprintln!("blaze: grammar compiler ready in {:.2} s", t0.elapsed().as_secs_f64());
    }
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let state = Arc::new(AppState {
        sched,
        text: Arc::new(text),
        formatter,
        cfg,
        max_context,
        vocab,
        grammars,
        started: now(),
        metrics: Metrics::default(),
    });
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/ping", get(|| async { "ok" }).post(|| async { "ok" }))
        .route("/version", get(version))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/models/{*id}", get(model))
        .route("/v1/chat/completions", post(chat::chat))
        .route("/v1/completions", post(completions::completions))
        .route("/tokenize", post(tokenize::tokenize))
        .route("/detokenize", post(tokenize::detokenize))
        .fallback(|| async { api_error(StatusCode::NOT_FOUND, "no such endpoint", Some("not_found"), None) })
        .method_not_allowed_fallback(|| async { error(StatusCode::METHOD_NOT_ALLOWED, "method not allowed") })
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .layer(axum::middleware::from_fn(cors))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("blaze: serving on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

/// CORS for browser clients: any origin, the headers it asks for; preflights answer here.
async fn cors(req: Request, next: Next) -> Response {
    let ask = req.headers().get(header::ACCESS_CONTROL_REQUEST_HEADERS).cloned();
    let preflight = req.method() == Method::OPTIONS;
    let mut res = if preflight { StatusCode::NO_CONTENT.into_response() } else { next.run(req).await };
    let h = res.headers_mut();
    h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    if preflight {
        h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, OPTIONS"));
        h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, ask.unwrap_or(HeaderValue::from_static("*")));
        h.insert(header::ACCESS_CONTROL_MAX_AGE, HeaderValue::from_static("86400"));
    }
    res
}

async fn version() -> Json<Value> {
    Json(json!({"version": env!("CARGO_PKG_VERSION")}))
}

/// Prometheus text: the counters of the HTTP side and the scheduler's gauges.
async fn metrics(State(st): State<Arc<AppState>>) -> Response {
    let m = &st.metrics;
    let s = &st.sched.stats;
    let rows = [
        ("blaze_requests_total", "counter", "Generations started", m.requests.load(Ordering::Relaxed) as f64),
        ("blaze_prompt_tokens_total", "counter", "Prompt tokens of finished or dropped generations", m.prompt_tokens.load(Ordering::Relaxed) as f64),
        ("blaze_generation_tokens_total", "counter", "Generated tokens", m.generation_tokens.load(Ordering::Relaxed) as f64),
        ("blaze_requests_dropped_total", "counter", "Generations dropped before their end (client gone or stop string)", m.dropped.load(Ordering::Relaxed) as f64),
        ("blaze_requests_cancelled_total", "counter", "Requests the scheduler dropped because their client was gone", s.cancelled.load(Ordering::Relaxed) as f64),
        ("blaze_requests_open", "gauge", "Generations open on the HTTP side", m.open.load(Ordering::Relaxed) as f64),
        ("blaze_requests_running", "gauge", "Requests prefilling or decoding", s.running.load(Ordering::Relaxed) as f64),
        ("blaze_requests_waiting", "gauge", "Requests waiting for a free row or KV room", s.waiting.load(Ordering::Relaxed) as f64),
    ];
    let mut out = String::new();
    for (name, kind, help, v) in rows {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}\n"));
    }
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
        if body.get(k).is_some_and(|v| v.as_i64().is_some_and(|x| x < 1)) {
            return Err(bad_param(k, format!("{k} must be an integer of at least 1")));
        }
    }
    if body.get("top_logprobs").and_then(Value::as_u64).is_some_and(|k| k > MAX_TOP_LOGPROBS) {
        return Err(bad_param("top_logprobs", format!("top_logprobs is at most {MAX_TOP_LOGPROBS}")));
    }
    if matches!(body.get("logprobs"), Some(Value::Number(k)) if k.as_u64().is_none_or(|k| k > MAX_TOP_LOGPROBS)) {
        return Err(bad_param("logprobs", format!("logprobs is at most {MAX_TOP_LOGPROBS}")));
    }
    let outside = |mut ids: Box<dyn Iterator<Item = i64> + '_>| ids.find(|&t| t < 0 || t as usize >= st.vocab);
    if let Some(bad) = body.get("stop_token_ids").and_then(Value::as_array).and_then(|a| outside(Box::new(a.iter().filter_map(Value::as_i64)))) {
        return Err(bad_param("stop_token_ids", format!("token id {bad} is outside the vocabulary of {}", st.vocab)));
    }
    if let Some(bad) = body.get("logit_bias").and_then(Value::as_object).and_then(|m| outside(Box::new(m.keys().filter_map(|k| k.parse().ok())))) {
        return Err(bad_param("logit_bias", format!("token id {bad} is outside the vocabulary of {}", st.vocab)));
    }
    Ok(())
}

fn n_of(body: &Value) -> usize {
    body.get("n").and_then(Value::as_u64).unwrap_or(1) as usize
}

/// Sampling parameters of a request body: the OpenAI fields, vLLM / SGLang extras (top_k, min_p, repetition_penalty,
/// min_tokens, bad_words), server defaults.
fn sampling_of(body: &Value, d: &SamplingParams, text: &Text) -> SamplingParams {
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
    SamplingParams {
        temperature: f("temperature").unwrap_or(d.temperature),
        top_p: f("top_p").unwrap_or(d.top_p),
        top_k: body.get("top_k").and_then(Value::as_i64).map(|x| x as i32).unwrap_or(d.top_k),
        min_p: f("min_p").unwrap_or(d.min_p),
        presence_penalty: f("presence_penalty").unwrap_or(d.presence_penalty),
        frequency_penalty: f("frequency_penalty").unwrap_or(d.frequency_penalty),
        seed: body.get("seed").and_then(Value::as_u64),
        repetition_penalty: f("repetition_penalty").unwrap_or(d.repetition_penalty),
        logit_bias,
        min_tokens: body.get("min_tokens").and_then(Value::as_u64).unwrap_or(0) as usize,
        bad_words,
        logprobs,
    }
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
}

impl Spec {
    fn of(st: &AppState, body: &Value, max_tokens: usize, think: Option<ThinkBudget>) -> Self {
        Self {
            params: sampling_of(body, &st.cfg.defaults, &st.text),
            max_tokens,
            stops: stop_strings(body),
            include_stop: ["include_stop_str_in_output", "no_stop_trim"].iter().any(|k| body.get(*k).and_then(Value::as_bool) == Some(true)),
            skip_special: body.get("skip_special_tokens").and_then(Value::as_bool).unwrap_or(true),
            stop_ids: stop_ids_of(body, &st.text),
            think,
        }
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
    let g = match grammar_of(body).map_err(|e| e.to_string())? {
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
    include: bool,
    held: String,
    stopped: bool,
}

impl StopCutter {
    fn new(stops: Vec<String>, include: bool) -> Self {
        Self { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), include, held: String::new(), stopped: false }
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
        let hit = self.stops.iter().filter_map(|s| self.held.find(s.as_str()).map(|i| (i, s.len()))).min_by_key(|&(i, l)| (i, l));
        if let Some((at, len)) = hit {
            self.stopped = true;
            let end = if self.include { at + len } else { at };
            let out = self.held[..end].to_string();
            self.held.clear();
            return out;
        }
        let keep = self.stops.iter().map(|s| overlap(&self.held, s)).max().unwrap_or(0);
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

/// Bytes of the longest suffix of `text` that is a proper prefix of `stop`.
fn overlap(text: &str, stop: &str) -> usize {
    (1..stop.len()).rev().filter(|&k| stop.is_char_boundary(k)).find(|&k| text.ends_with(&stop[..k])).unwrap_or(0)
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
}

impl Totals {
    fn new(prompt: usize) -> Arc<Self> {
        Arc::new(Self { prompt, cached: AtomicUsize::new(0), completion: AtomicUsize::new(0) })
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
    st.sched
        .submit(GenRequest {
            tokens: tokens.to_vec(),
            params,
            max_tokens: spec.max_tokens,
            stop_ids: spec.stop_ids.clone(),
            cache_points,
            think: spec.think.clone(),
            constraint,
            events: tx,
            cancel: cancel.clone(),
        })
        .map_err(|_| error(StatusCode::SERVICE_UNAVAILABLE, "the scheduler has stopped"))?;
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
