//! OpenAI-compatible HTTP server: /v1/chat/completions (streaming or not, reasoning split out, tool calls parsed),
//! /v1/completions, /v1/models, /health. Chat templates, reasoning and tool-call parsing come from the vendored
//! blaze-chat; generation runs on the scheduler.

use crate::sampler::SamplingParams;
use crate::sched::{Finish, GenEvent, GenRequest, Scheduler, ThinkBudget, TokenLogprob};
use crate::structured::{Constraint, Grammars, grammar_of};
use crate::text::Text;
use anyhow::{Result, anyhow};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use blaze_chat::dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use blaze_chat::dynamo_renderer::{OAIChatLikeRequest, TextInput};
use blaze_chat::{ChatEvent, ChatFinishReason, ChatFormatter, ChatResponseProcessor, DecodedChatEvent, ProcessorError};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc::unbounded_channel;

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
    formatter: Option<ChatFormatter>,
    cfg: ServeConfig,
    max_context: usize,
    /// structured outputs (None: the tokenizer could not be read for grammars)
    grammars: Option<Grammars>,
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

fn error(code: StatusCode, message: impl Into<String>) -> Response {
    let message = message.into();
    (code, Json(json!({"error": {"object": "error", "message": message, "type": "BadRequestError", "code": code.as_u16()}})))
        .into_response()
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
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
    let state = Arc::new(AppState { sched, text: Arc::new(text), formatter, cfg, max_context, grammars });
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("blaze: serving on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn models(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": st.cfg.served_name, "object": "model", "created": now(), "owned_by": "blaze",
                                             "max_model_len": st.max_context}]}))
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

/// Text of the generated tokens, emitted as soon as it is valid UTF-8, with stop strings cut.
struct Detok {
    text: Arc<Text>,
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
    out: String,
    stops: Vec<String>,
    stopped: bool,
}

impl Detok {
    fn new(text: Arc<Text>, stops: Vec<String>) -> Self {
        Self { text, ids: Vec::new(), prefix: 0, read: 0, out: String::new(), stops, stopped: false }
    }

    /// New text after `id` (empty while a character is incomplete).
    fn push(&mut self, id: u32) -> String {
        if self.stopped {
            return String::new();
        }
        self.ids.push(id);
        let before = self.text.decode_skip(&self.ids[self.prefix..self.read]).unwrap_or_default();
        let after = self.text.decode_skip(&self.ids[self.prefix..]).unwrap_or_default();
        if after.len() <= before.len() || after.ends_with('\u{FFFD}') || !after.is_char_boundary(before.len()) {
            return String::new();
        }
        let mut delta = after[before.len()..].to_string();
        self.prefix = self.read;
        self.read = self.ids.len();
        // stop strings: cut at the first one in the whole output
        let start = self.out.len();
        self.out.push_str(&delta);
        if let Some(cut) = self.stops.iter().filter_map(|s| self.out.find(s.as_str())).min() {
            self.stopped = true;
            self.out.truncate(cut);
            delta = if cut > start { self.out[start..].to_string() } else { String::new() };
        }
        delta
    }
}

/// One generation's events as text deltas: (delta, finish, prompt tokens, cached tokens, completion tokens), and the
/// logprobs of its tokens when asked for.
struct Gen {
    logprobs: Vec<(u32, TokenLogprob)>,
    rx: tokio::sync::mpsc::UnboundedReceiver<GenEvent>,
    cancel: Arc<AtomicBool>,
    detok: Detok,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
}

enum Piece {
    Text(String),
    Finish(Finish),
    Error(String),
}

impl Gen {
    async fn next(&mut self) -> Option<Piece> {
        loop {
            match self.rx.recv().await? {
                GenEvent::Started { prompt_tokens, cached_tokens } => {
                    self.prompt_tokens = prompt_tokens;
                    self.cached_tokens = cached_tokens;
                }
                GenEvent::Token(t, lp) => {
                    self.completion_tokens += 1;
                    if let Some(lp) = lp {
                        self.logprobs.push((t as u32, lp));
                    }
                    let d = self.detok.push(t as u32);
                    if self.detok.stopped {
                        self.cancel.store(true, Ordering::Relaxed);
                    }
                    if !d.is_empty() {
                        return Some(Piece::Text(d));
                    }
                }
                GenEvent::Done(f) => {
                    let f = if self.detok.stopped { Finish::Stop } else { f };
                    return Some(Piece::Finish(f));
                }
                GenEvent::Error(e) => return Some(Piece::Error(e)),
            }
        }
    }
}

impl Drop for Gen {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
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
fn constraint_of(st: &AppState, body: &Value, after: Option<i32>) -> std::result::Result<Option<Constraint>, String> {
    let Some(g) = grammar_of(body).map_err(|e| e.to_string())? else { return Ok(None) };
    let gr = st.grammars.as_ref().ok_or("structured outputs are unavailable for this tokenizer")?;
    let matcher = gr.matcher(g).map_err(|e| e.to_string())?;
    Ok(Some(Constraint { matcher, after }))
}

fn submit(st: &AppState, tokens: Vec<i64>, params: SamplingParams, max_tokens: usize, stops: Vec<String>,
          stop_ids: Vec<i32>, constraint: Option<Constraint>) -> Result<Gen> {
    submit_ex(st, tokens, params, max_tokens, stops, stop_ids, None, constraint)
}

#[allow(clippy::too_many_arguments)]
fn submit_ex(st: &AppState, tokens: Vec<i64>, params: SamplingParams, max_tokens: usize, stops: Vec<String>,
             stop_ids: Vec<i32>, think: Option<ThinkBudget>, constraint: Option<Constraint>) -> Result<Gen> {
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
    st.sched
        .submit(GenRequest { tokens, params, max_tokens, stop_ids, cache_points, think, constraint, events: tx, cancel: cancel.clone() })
        .map_err(|_| anyhow!("scheduler stopped"))?;
    Ok(Gen {
        logprobs: Vec::new(),
        rx,
        cancel,
        detok: Detok::new(st.text.clone(), stops),
        prompt_tokens: 0,
        cached_tokens: 0,
        completion_tokens: 0,
    })
}

fn usage(g: &Gen) -> Value {
    json!({"prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens,
           "total_tokens": g.prompt_tokens + g.completion_tokens,
           "prompt_tokens_details": {"cached_tokens": g.cached_tokens}})
}

fn finish_str(f: Finish) -> &'static str {
    match f {
        Finish::Length => "length",
        _ => "stop",
    }
}

// completions

async fn completions(State(st): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let prompt = match body.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => a[0].as_str().unwrap().to_string(),
        _ => return error(StatusCode::BAD_REQUEST, "prompt must be a string"),
    };
    let tokens: Vec<i64> = match st.text.encode(&prompt) {
        Ok(t) => t.into_iter().map(i64::from).collect(),
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let params = sampling_of(&body, &st.cfg.defaults, &st.text);
    let max_tokens = body.get("max_tokens").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(16);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let constraint = match constraint_of(&st, &body, None) {
        Ok(c) => c,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let mut g = match submit(&st, tokens, params, max_tokens, stop_strings(&body), stop_ids_of(&body, &st.text), constraint) {
        Ok(g) => g,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e.to_string()),
    };
    let id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let model = st.cfg.served_name.clone();
    if stream {
        let s = async_stream::stream! {
            while let Some(p) = g.next().await {
                match p {
                    Piece::Text(t) => yield json!({"id": id, "object": "text_completion", "created": now(), "model": model,
                                                   "choices": [{"index": 0, "text": t, "finish_reason": null}]}).to_string(),
                    Piece::Finish(f) => {
                        yield json!({"id": id, "object": "text_completion", "created": now(), "model": model,
                                     "choices": [{"index": 0, "text": "", "finish_reason": finish_str(f)}], "usage": usage(&g)}).to_string();
                        break;
                    }
                    Piece::Error(e) => { yield json!({"error": {"message": e}}).to_string(); break; }
                }
            }
            yield "[DONE]".to_string();
        };
        return sse(s);
    }
    let mut text = String::new();
    let mut finish = Finish::Stop;
    while let Some(p) = g.next().await {
        match p {
            Piece::Text(t) => text.push_str(&t),
            Piece::Finish(f) => {
                finish = f;
                break;
            }
            Piece::Error(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }
    let logprobs = if g.logprobs.is_empty() {
        Value::Null
    } else {
        let tok = |t: u32| st.text.decode(&[t]).unwrap_or_default();
        json!({
            "tokens": g.logprobs.iter().map(|(t, _)| tok(*t)).collect::<Vec<_>>(),
            "token_logprobs": g.logprobs.iter().map(|(_, l)| l.logprob).collect::<Vec<_>>(),
            "top_logprobs": g.logprobs.iter().map(|(_, l)| {
                l.top.iter().map(|&(t, lp)| (tok(t as u32), json!(lp))).collect::<serde_json::Map<_, _>>()
            }).collect::<Vec<_>>(),
        })
    };
    Json(json!({"id": id, "object": "text_completion", "created": now(), "model": model,
                "choices": [{"index": 0, "text": text, "logprobs": logprobs, "finish_reason": finish_str(finish)}], "usage": usage(&g)}))
    .into_response()
}

fn sse(s: impl Stream<Item = String> + Send + 'static) -> Response {
    Sse::new(s.map(|d| Ok::<_, Infallible>(Event::default().data(d)))).keep_alive(KeepAlive::default()).into_response()
}

// chat

/// The request with the template kwargs the client sent (chat_template_kwargs), for the renderer.
struct TemplateRequest<'a> {
    request: &'a CreateChatCompletionRequest,
    kwargs: Option<&'a HashMap<String, Value>>,
}

impl OAIChatLikeRequest for TemplateRequest<'_> {
    fn model(&self) -> String {
        self.request.model()
    }
    fn messages(&self) -> minijinja::Value {
        self.request.messages()
    }
    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        self.request.typed_messages()
    }
    fn tools(&self) -> Option<minijinja::Value> {
        self.request.tools()
    }
    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.request.tool_choice()
    }
    fn response_format(&self) -> Option<minijinja::Value> {
        self.request.response_format()
    }
    fn reasoning_effort(&self) -> Option<minijinja::Value> {
        self.request.reasoning_effort()
    }
    fn should_add_generation_prompt(&self) -> bool {
        self.request.should_add_generation_prompt()
    }
    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        self.kwargs
    }
    fn extract_text(&self) -> Option<TextInput> {
        self.request.extract_text()
    }
    fn mm_processor_kwargs(&self) -> Option<&Value> {
        self.request.mm_processor_kwargs()
    }
}

fn finish_of(f: Finish) -> ChatFinishReason {
    match f {
        Finish::Length => ChatFinishReason::Length,
        _ => ChatFinishReason::Stop,
    }
}

fn finish_name(f: ChatFinishReason) -> &'static str {
    match f {
        ChatFinishReason::Stop => "stop",
        ChatFinishReason::Length => "length",
        ChatFinishReason::ContentFilter => "content_filter",
        ChatFinishReason::ToolCalls => "tool_calls",
    }
}

async fn chat(State(st): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let Some(formatter) = st.formatter.clone() else { return error(StatusCode::BAD_REQUEST, "this model has no chat template") };
    let request: CreateChatCompletionRequest = match serde_json::from_value(body.clone()) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("bad request: {e}")),
    };
    if request.messages.is_empty() {
        return error(StatusCode::BAD_REQUEST, "messages cannot be empty");
    }
    use blaze_chat::dynamo_protocols::types::ChatCompletionToolChoiceOption as Choice;
    let tools = match blaze_chat::chat_tool_definitions(request.tools.as_deref(), &request.messages) {
        Ok(t) => t,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let tools_enabled = !tools.is_empty() && !matches!(request.tool_choice, Some(Choice::None));
    let named = matches!(request.tool_choice, Some(Choice::Named(_)));
    let mut kwargs: Option<HashMap<String, Value>> =
        body.get("chat_template_kwargs").and_then(|v| serde_json::from_value(v.clone()).ok());
    let _thinking = formatter.resolve_thinking(&mut kwargs, tools_enabled, named);
    let prompt = match formatter.render_prompt(&TemplateRequest { request: &request, kwargs: kwargs.as_ref() }) {
        Ok(p) => p.into_text(),
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("chat template render failed: {e}")),
    };
    let tokens: Vec<i64> = match st.text.encode(&prompt) {
        Ok(t) => t.into_iter().map(i64::from).collect(),
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let params = sampling_of(&body, &st.cfg.defaults, &st.text);
    #[allow(deprecated)]
    let max_tokens = request.max_completion_tokens.or(request.max_tokens).map(|x| x as usize).unwrap_or(st.cfg.default_max_tokens);
    let stream = request.stream.unwrap_or(false);
    let include_usage = request.stream_options.as_ref().is_some_and(|o| o.include_usage);
    let starts_in_reasoning = matches!(st.cfg.reasoning_parser.as_deref(), Some("qwen3" | "deepseek-r1" | "glm45"))
        && prompt.trim_end_matches('\n').ends_with("<think>");
    let think = think_budget(&st, &body, starts_in_reasoning);
    // with reasoning first, the grammar applies to the answer after </think>
    let after = starts_in_reasoning.then(|| st.text.token_id("</think>")).flatten().map(|t| t as i32);
    let constraint = match constraint_of(&st, &body, after) {
        Ok(c) => c,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let mut g = match submit_ex(&st, tokens, params, max_tokens, stop_strings(&body), stop_ids_of(&body, &st.text), think,
                                constraint) {
        Ok(g) => g,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e.to_string()),
    };
    let tool_parser = tools_enabled.then(|| st.cfg.tool_parser.clone()).flatten();
    let processor = ChatResponseProcessor::new(
        tool_parser,
        st.cfg.reasoning_parser.clone(),
        (!tools.is_empty()).then_some(tools),
        request.tool_choice.clone(),
        false,
        request.parallel_tool_calls.unwrap_or(true),
        1,
    )
    .with_reasoning_state(starts_in_reasoning.then_some(true));
    // generation events -> decoded chat events (prompt tokens ride on every event; counts are per event)
    let (meta_tx, mut meta_rx) = tokio::sync::watch::channel((0usize, 0usize, 0usize));
    // logprobs of the tokens decoded so far, taken by the next response chunk
    let lps: Arc<std::sync::Mutex<Vec<(u32, TokenLogprob)>>> = Default::default();
    let lps_gen = lps.clone();
    let decoded = async_stream::stream! {
        loop {
            let piece = g.next().await;
            lps_gen.lock().unwrap().append(&mut g.logprobs);
            match piece {
                Some(Piece::Text(t)) => {
                    let _ = meta_tx.send((g.prompt_tokens, g.cached_tokens, g.completion_tokens));
                    yield Ok::<_, ApiError>(DecodedChatEvent { choice: 0, text: t, token_ids: Vec::new(), finish_reason: None,
                        logprobs: None, prompt_tokens: g.prompt_tokens as u32, completion_tokens: 1 });
                }
                Some(Piece::Finish(f)) => {
                    let _ = meta_tx.send((g.prompt_tokens, g.cached_tokens, g.completion_tokens));
                    yield Ok(DecodedChatEvent { choice: 0, text: String::new(), token_ids: Vec::new(), finish_reason: Some(finish_of(f)),
                        logprobs: None, prompt_tokens: g.prompt_tokens as u32, completion_tokens: 0 });
                    break;
                }
                Some(Piece::Error(e)) => { yield Err(ApiError { message: e }); break; }
                None => break,
            }
        }
    };
    let mut events = processor.process_stream(Box::pin(decoded));
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let model = st.cfg.served_name.clone();
    let created = now();
    let usage_of = |m: (usize, usize, usize)| {
        json!({"prompt_tokens": m.0, "completion_tokens": m.2, "total_tokens": m.0 + m.2,
               "prompt_tokens_details": {"cached_tokens": m.1}})
    };
    let text = st.text.clone();
    let take_lps = move || -> Value {
        let items = std::mem::take(&mut *lps.lock().unwrap());
        if items.is_empty() {
            return Value::Null;
        }
        let entry = |t: u32, lp: f32| {
            let s = text.decode(&[t]).unwrap_or_default();
            json!({"token": s, "logprob": lp, "bytes": s.as_bytes()})
        };
        json!({"content": items.iter().map(|(t, l)| {
            let mut e = entry(*t, l.logprob);
            e["top_logprobs"] = l.top.iter().map(|&(t, lp)| entry(t as u32, lp)).collect();
            e
        }).collect::<Vec<_>>()})
    };
    if stream {
        let s = async_stream::stream! {
            while let Some(ev) = events.next().await {
                let chunk = |choices: Value, usage: Value| json!({"id": id, "object": "chat.completion.chunk", "created": created,
                                                                  "model": model, "choices": choices, "usage": usage}).to_string();
                match ev {
                    Ok(ChatEvent::Role { .. }) => yield chunk(json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]), Value::Null),
                    Ok(ChatEvent::Delta { content, reasoning_content, tool_calls, finish_reason, .. }) => {
                        let mut delta = serde_json::Map::new();
                        if let Some(c) = content { delta.insert("content".into(), json!(c)); }
                        if let Some(r) = reasoning_content { delta.insert("reasoning_content".into(), json!(r)); }
                        if let Some(tc) = tool_calls {
                            let v: Vec<Value> = tc.iter().map(|t| {
                                let mut o = json!({"index": t.index, "type": "function", "function": {}});
                                if let Some(id) = &t.id { o["id"] = json!(id); }
                                if let Some(n) = &t.name { o["function"]["name"] = json!(n); }
                                if let Some(a) = &t.arguments { o["function"]["arguments"] = json!(a); }
                                o
                            }).collect();
                            delta.insert("tool_calls".into(), json!(v));
                        }
                        yield chunk(json!([{"index": 0, "delta": delta, "logprobs": take_lps(),
                                            "finish_reason": finish_reason.map(finish_name)}]), Value::Null);
                    }
                    Ok(ChatEvent::Usage { .. }) => {
                        let m = *meta_rx.borrow_and_update();
                        if include_usage { yield chunk(json!([]), usage_of(m)); }
                    }
                    Err(e) => { yield json!({"error": {"message": e.message}}).to_string(); break; }
                }
            }
            yield "[DONE]".to_string();
        };
        return sse(s);
    }
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut calls: Vec<Value> = Vec::new();
    let mut finish = None;
    while let Some(ev) = events.next().await {
        match ev {
            Ok(ChatEvent::Delta { content: c, reasoning_content: r, tool_calls, finish_reason, .. }) => {
                if let Some(c) = c {
                    content.push_str(&c);
                }
                if let Some(r) = r {
                    reasoning.push_str(&r);
                }
                for t in tool_calls.unwrap_or_default() {
                    let i = t.index as usize;
                    while calls.len() <= i {
                        calls.push(json!({"id": "", "type": "function", "function": {"name": "", "arguments": ""}}));
                    }
                    if let Some(id) = t.id {
                        calls[i]["id"] = json!(id);
                    }
                    if let Some(n) = t.name {
                        calls[i]["function"]["name"] = json!(n);
                    }
                    if let Some(a) = t.arguments {
                        let prev = calls[i]["function"]["arguments"].as_str().unwrap_or("").to_string();
                        calls[i]["function"]["arguments"] = json!(prev + &a);
                    }
                }
                if finish_reason.is_some() {
                    finish = finish_reason;
                }
            }
            Ok(_) => {}
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.message),
        }
    }
    let mut message = json!({"role": "assistant", "content": if content.is_empty() && !calls.is_empty() { Value::Null } else { json!(content) }});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !calls.is_empty() {
        message["tool_calls"] = json!(calls);
    }
    let m = *meta_rx.borrow_and_update();
    let u = usage_of(m);
    Json(json!({"id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": message, "logprobs": take_lps(),
                             "finish_reason": finish.map(finish_name).unwrap_or("stop")}],
                "usage": u}))
    .into_response()
}
