use super::{Checked, bad_param, error, json, media};
use crate::inkling::{Model, Options, generate, render_chat, tokenize_prompt};
use crate::sampler::SamplingParams;
use crate::text::Text;
use anyhow::{Context, Result, ensure};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

pub struct Config {
    pub path: PathBuf,
    pub options: Options,
    pub host: String,
    pub port: u16,
    pub name: String,
    pub max_tokens: usize,
    pub queue: usize,
    pub allowed_media_domains: Vec<String>,
}

struct StateData {
    tx: mpsc::Sender<Job>,
    name: String,
    max_tokens: usize,
    context: usize,
}

struct Job {
    body: Value,
    urls: Vec<(String, Option<String>)>,
    sampling: SamplingParams,
    max_tokens: usize,
    stream: bool,
    tx: mpsc::Sender<Reply>,
}

enum Reply {
    Delta {
        text: String,
        reasoning: bool,
    },
    Done {
        content: String,
        reasoning: String,
        prompt: usize,
        tokens: usize,
        reason: &'static str,
    },
    Error(String),
}

pub async fn serve(config: Config) -> Result<()> {
    ensure!(
        config.max_tokens > 0 && config.queue > 0,
        "Inkling generation budget and queue must be positive"
    );
    let mut early = super::listen(&config.host, config.port).await?;
    let (tx, mut rx) = mpsc::channel::<Job>(config.queue);
    let state = Arc::new(StateData {
        tx,
        name: config.name,
        max_tokens: config.max_tokens,
        context: config.options.context,
    });
    let (ready, status) = oneshot::channel();
    std::thread::Builder::new().name("blaze-inkling".into()).spawn(move || {
        let stage = crate::logging::stage("Inkling tokenizer, model weights and CUDA buffers");
        let loaded = (|| Ok::<_, anyhow::Error>((Text::load(&config.path)?, Model::load(&config.path, config.options)?)))();
        let (text, mut model) = match loaded {
            Ok(model) => {
                stage.done("ready");
                if ready.send(Ok(())).is_err() {
                    return;
                }
                model
            }
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        };
        let cfg = media::MmConfig::new(
            None,
            Default::default(),
            4,
            0,
            Duration::from_secs(5),
            16 << 20,
            config.allowed_media_domains,
        );
        while let Some(job) = rx.blocking_recv() {
            if job.tx.is_closed() {
                continue;
            }
            match run_job(&mut model, &text, &config.path, &cfg, &job) {
                Ok(reply) => {
                    let _ = job.tx.blocking_send(reply);
                }
                Err(e) => {
                    let _ = job.tx.blocking_send(Reply::Error(format!("{e:#}")));
                }
            }
        }
    })?;
    status.await.context("Inkling worker exited during startup")??;
    crate::eprintln!("{} ready at {}", state.name, early.listen);
    let app = Router::new()
        .route("/health", get(|| async { StatusCode::OK }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::max(32 << 20))
        .with_state(state);
    early
        .app
        .set(app)
        .map_err(|_| anyhow::anyhow!("Inkling HTTP app already initialized"))?;
    let _ = early.signal.wait_for(|shutdown| *shutdown).await;
    let _ = early.stop.send(());
    early.serving.await??;
    Ok(())
}

async fn models(State(st): State<Arc<StateData>>) -> Json<Value> {
    Json(json!({"object":"list","data":[{"id":st.name,"object":"model","owned_by":"blaze","max_model_len":st.context}]}))
}

fn sampling(body: &Value) -> Checked<SamplingParams> {
    let number = |key: &str, default: f32, valid: fn(f32) -> bool| -> Checked<f32> {
        match body.get(key).filter(|v| !v.is_null()) {
            None => Ok(default),
            Some(v) => v
                .as_f64()
                .map(|v| v as f32)
                .filter(|&v| v.is_finite() && valid(v))
                .ok_or_else(|| bad_param(key, format!("invalid {key}"))),
        }
    };
    let top_k = match body.get("top_k").filter(|v| !v.is_null()) {
        Some(v) => v
            .as_i64()
            .and_then(|n| i32::try_from(n).ok())
            .filter(|&n| n >= -1)
            .ok_or_else(|| bad_param("top_k", "top_k must be at least -1"))?,
        None => -1,
    };
    Ok(SamplingParams {
        temperature: number("temperature", 1.0, |v| v >= 0.0)?,
        top_p: number("top_p", 1.0, |v| v > 0.0 && v <= 1.0)?,
        top_k,
        min_p: number("min_p", 0.0, |v| (0.0..=1.0).contains(&v))?,
        seed: body["seed"].as_u64(),
        ..Default::default()
    })
}

async fn chat(State(st): State<Arc<StateData>>, Json(body): Json<Value>) -> Checked<Response> {
    if body["model"].as_str().is_some_and(|model| model != st.name) {
        return Err(error(StatusCode::NOT_FOUND, "unknown model"));
    }
    if !body.get("messages").and_then(Value::as_array).is_some_and(|m| !m.is_empty()) {
        return Err(bad_param("messages", "messages must be a nonempty array"));
    }
    if let Some(param) = [
        "tools",
        "tool_choice",
        "stop",
        "response_format",
        "logit_bias",
        "logprobs",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    ]
    .into_iter()
    .find(|p| !body[*p].is_null())
    {
        return Err(bad_param(param, format!("{param} is unavailable on the Inkling runner")));
    }
    if body.get("n").is_some_and(|v| !v.is_null() && v.as_u64() != Some(1)) {
        return Err(bad_param("n", "the Inkling runner generates one choice"));
    }
    let urls = media::image_parts(&body)?;
    if urls.len() > 4 {
        return Err(bad_param("messages", "at most four images per request"));
    }
    let max_tokens = match body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .filter(|v| !v.is_null())
    {
        Some(v) => v
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0 && n < st.context)
            .ok_or_else(|| {
                bad_param(
                    "max_tokens",
                    "generation budget must be positive and smaller than the context",
                )
            })?,
        None => st.max_tokens,
    };
    let stream = body["stream"].as_bool().unwrap_or(false);
    let sampling = sampling(&body)?;
    let (tx, mut rx) = mpsc::channel(32);
    st.tx
        .try_send(Job {
            body,
            urls,
            sampling,
            max_tokens,
            stream,
            tx,
        })
        .map_err(|e| {
            if matches!(e, mpsc::error::TrySendError::Full(_)) {
                error(StatusCode::TOO_MANY_REQUESTS, "Inkling generation queue is full")
            } else {
                error(StatusCode::SERVICE_UNAVAILABLE, "Inkling worker is unavailable")
            }
        })?;
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if stream {
        let events = async_stream::stream! {
            let chunk = |delta: Value, reason: Option<&str>| Event::default().data(json!({
                "id":id,"object":"chat.completion.chunk","created":time,"model":st.name,
                "choices":[{"index":0,"delta":delta,"finish_reason":reason}],
            }).to_string());
            yield Ok::<_, Infallible>(chunk(json!({"role":"assistant"}), None));
            while let Some(msg) = rx.recv().await {
                match msg {
                    Reply::Delta { text, reasoning } => {
                        let delta = if reasoning { json!({"reasoning_content":text}) } else { json!({"content":text}) };
                        yield Ok(chunk(delta, None));
                    }
                    Reply::Done { reason, .. } => { yield Ok(chunk(json!({}), Some(reason))); break; }
                    Reply::Error(message) => { yield Ok(Event::default().data(json!({"error":{"message":message}}).to_string())); break; }
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        return Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response());
    }
    match rx.recv().await {
        Some(Reply::Done {
            content,
            reasoning,
            prompt,
            tokens,
            reason,
        }) => {
            let mut message = json!({"role":"assistant","content":content});
            if !reasoning.is_empty() {
                message["reasoning_content"] = reasoning.into();
            }
            Ok(Json(json!({
                "id":id,"object":"chat.completion","created":time,"model":st.name,
                "choices":[{"index":0,"message":message,"finish_reason":reason}],
                "usage":{"prompt_tokens":prompt,"completion_tokens":tokens,"total_tokens":prompt+tokens},
            }))
            .into_response())
        }
        Some(Reply::Error(e)) => Err(error(StatusCode::BAD_REQUEST, e)),
        _ => Err(error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Inkling worker did not complete the request",
        )),
    }
}

fn run_job(model: &mut Model, text: &Text, path: &std::path::Path, media: &media::MmConfig, job: &Job) -> Result<Reply> {
    let prompt = render_chat(path, &job.body)?;
    let deadline = Instant::now() + media.fetch_timeout;
    let images = job
        .urls
        .iter()
        .map(|(url, _)| {
            ensure!(
                url.starts_with("data:") || !media.allowed_domains.is_empty(),
                "remote images require --allowed-media-domain"
            );
            media::load_url(media, url, deadline).map_err(anyhow::Error::msg)
        })
        .collect::<Result<Vec<_>>>()?;
    let input = tokenize_prompt(model, text, &prompt, &images)?;
    let max_tokens = job.max_tokens.min(model.context().saturating_sub(input.len()));
    let mut decoder = Output::new(text)?;
    let cpu_before = model.cpu_expert_routes();
    let result = generate(
        model,
        text,
        &input,
        max_tokens,
        &job.sampling,
        || job.tx.is_closed(),
        |id| {
            if job.tx.is_closed() {
                return false;
            }
            if let Some((delta, reasoning)) = decoder.push(id)
                && job.stream
            {
                return job.tx.blocking_send(Reply::Delta { text: delta, reasoning }).is_ok();
            }
            true
        },
    )?;
    if let Some((text, reasoning)) = decoder.flush()
        && job.stream
    {
        let _ = job.tx.blocking_send(Reply::Delta { text, reasoning });
    }
    crate::eprintln!(
        "Inkling {} prompt tokens in {:.3}s, {} generated tokens in {:.3}s, {} CPU expert routes",
        input.len(),
        result.prompt_elapsed.as_secs_f64(),
        result.ids.len(),
        result.decode_elapsed.as_secs_f64(),
        model.cpu_expert_routes() - cpu_before
    );
    Ok(Reply::Done {
        content: decoder.content,
        reasoning: decoder.thinking,
        prompt: input.len(),
        tokens: result.ids.len(),
        reason: if result.stop { "stop" } else { "length" },
    })
}

struct Output<'a> {
    text: &'a Text,
    begin_reason: u32,
    begin_text: u32,
    end: u32,
    model: u32,
    reasoning: bool,
    pending: Vec<u8>,
    content: String,
    thinking: String,
}

impl<'a> Output<'a> {
    fn new(text: &'a Text) -> Result<Self> {
        let id = |token| {
            text.token_id(token)
                .with_context(|| format!("Inkling tokenizer is missing {token}"))
        };
        Ok(Self {
            text,
            begin_reason: id("<|content_thinking|>")?,
            begin_text: id("<|content_text|>")?,
            end: id("<|end_message|>")?,
            model: id("<|message_model|>")?,
            reasoning: false,
            pending: Vec::new(),
            content: String::new(),
            thinking: String::new(),
        })
    }

    fn push(&mut self, id: u32) -> Option<(String, bool)> {
        if id == self.model {
            return None;
        }
        if id == self.begin_reason || id == self.begin_text || id == self.end {
            let last = self.flush();
            self.reasoning = id == self.begin_reason;
            return last;
        }
        self.pending.extend(self.text.token_bytes(id));
        match std::str::from_utf8(&self.pending) {
            Ok(_) => self.flush(),
            Err(e) if e.error_len().is_some() => self.flush(),
            Err(_) => None,
        }
    }

    fn flush(&mut self) -> Option<(String, bool)> {
        if self.pending.is_empty() {
            return None;
        }
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        if self.reasoning {
            self.thinking.push_str(&text);
        } else {
            self.content.push_str(&text);
        }
        Some((text, self.reasoning))
    }
}
