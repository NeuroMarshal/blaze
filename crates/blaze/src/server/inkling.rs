use super::{chat::ChatEvents, *};
use crate::inkling::{Model, Options, decode_image, generate, inputs_from_tokens, render_chat};
use anyhow::{Context, ensure};
use blaze_chat::{ChatEvent, ChatFinishReason, ChatToolCallDelta};
use std::time::Instant;
use tokio::sync::oneshot;

pub struct Config {
    pub path: std::path::PathBuf,
    pub options: Options,
    pub service: ServeConfig,
    pub queue: usize,
    pub media: MmConfig,
}

pub(super) struct Chat {
    path: std::path::PathBuf,
    vision: bool,
}

pub async fn serve(config: Config) -> Result<()> {
    let Config { path, options, service, queue, media } = config;
    ensure!(service.default_max_tokens > 0 && queue > 0, "generation budget and queue must be positive");
    let early = super::listen(&service.host, service.port).await?;
    let text = Arc::new(Text::load(&path)?);
    let vocab = crate::config::ModelConfig::load(&path)?.vocab;
    let max_context = options.context;
    let chat = Chat { path: path.clone(), vision: options.max_image_tokens > 0 };
    let (ready, status) = oneshot::channel();
    let worker_text = text.clone();
    let model_path = path.clone();
    let sched = Scheduler::start_worker(queue, "blaze-sched", move |rx, stop, stats| {
        let stage = crate::logging::stage("model weights and CUDA buffers");
        let mut model = match Model::load(&model_path, options) {
            Ok(model) => model,
            Err(err) => {
                let _ = ready.send(Err(err));
                return;
            }
        };
        stage.done("ready");
        if ready.send(Ok(())).is_err() {
            return;
        }
        for job in rx {
            if stop.load(Ordering::Acquire) {
                break;
            }
            if job.cancel.load(Ordering::Relaxed) || job.trace.cancelled.load(Ordering::Relaxed) {
                stats.cancelled.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            stats.running.store(1, Ordering::Relaxed);
            if let Err(err) = run_job(&mut model, &worker_text, job, &stop, &stats) {
                stats.errors.fetch_add(1, Ordering::Relaxed);
                crate::eprintln!("generation failed: {err:#}");
            }
            stats.running.store(0, Ordering::Relaxed);
        }
    })?;
    status.await.context("model worker exited during startup")??;
    let result = super::serve_with(early, sched.clone(), &path, text, None, service, max_context, vocab, media, Some(chat)).await;
    sched.shutdown();
    result
}

fn run_job(model: &mut Model, text: &Text, mut job: GenRequest, stop: &AtomicBool, stats: &crate::sched::SchedStats) -> Result<()> {
    stats.queue_time.observe(job.trace.received.elapsed().as_secs_f64());
    let mut run = || {
        ensure!(!job.embed && job.constraint.is_none() && job.think.is_none() && job.params.logprobs.is_none(),
            "generation option is unavailable for this model");
        let ids = std::mem::take(&mut job.tokens).into_iter().map(u32::try_from).collect::<std::result::Result<_, _>>()?;
        let input = inputs_from_tokens(model, text, ids, &job.images)?;
        let budget = job.max_tokens.min(model.context().saturating_sub(input.len()));
        let cpu_before = model.cpu_expert_routes();
        let mut last = None;
        let _ = job.events.send(GenEvent::Started { prompt_tokens: input.len(), cached_tokens: 0 });
        let result = generate(
            model, text, &input, budget, &job.params,
            || stop.load(Ordering::Relaxed) || job.cancel.load(Ordering::Relaxed)
                || job.trace.cancelled.load(Ordering::Relaxed) || job.events.is_closed(),
            |id| {
                let now = Instant::now();
                if let Some(previous) = last.replace(now) {
                    stats.inter_token.observe(now.duration_since(previous).as_secs_f64());
                } else {
                    stats.ttft.observe(now.duration_since(job.trace.received).as_secs_f64());
                }
                job.events.send(GenEvent::Token(id as i32, None)).is_ok()
            },
        )?;
        if result.stop && let Some(&eos) = result.ids.last() {
            let _ = job.events.send(GenEvent::Token(eos as i32, None));
        }
        let finish = if job.cancel.load(Ordering::Relaxed) || job.trace.cancelled.load(Ordering::Relaxed) || job.events.is_closed() {
            stats.cancelled.fetch_add(1, Ordering::Relaxed);
            Finish::Cancelled
        } else if result.stop {
            Finish::Stop
        } else {
            Finish::Length
        };
        let _ = job.events.send(GenEvent::Done(finish));
        stats.completed.fetch_add(1, Ordering::Relaxed);
        stats.prompt_tokens.fetch_add(input.len() as u64, Ordering::Relaxed);
        stats.prefill_tokens.fetch_add(input.len() as u64, Ordering::Relaxed);
        stats.generation_tokens.fetch_add(result.ids.len() as u64, Ordering::Relaxed);
        stats.prefill_micros.fetch_add(result.prompt_elapsed.as_micros() as u64, Ordering::Relaxed);
        stats.decode_micros.fetch_add(result.decode_elapsed.as_micros() as u64, Ordering::Relaxed);
        stats.prefill_step.observe(result.prompt_elapsed.as_secs_f64());
        stats.request_time.observe(job.trace.received.elapsed().as_secs_f64());
        crate::eprintln!(
            "Inkling {} prompt tokens in {:.3}s, {} generated tokens in {:.3}s, {} CPU expert routes",
            input.len(), result.prompt_elapsed.as_secs_f64(), result.ids.len(), result.decode_elapsed.as_secs_f64(),
            model.cpu_expert_routes() - cpu_before
        );
        Ok(())
    };
    let result = run();
    if let Err(err) = &result {
        let _ = job.events.send(GenEvent::Error(format!("{err:#}")));
    }
    result
}

pub(super) async fn chat(chat: &Chat, st: &Arc<AppState>, body: &mut Value) -> Checked<(ChatEvents, Arc<Totals>)> {
    if body["model"].as_str().is_some_and(|name| name != st.cfg.served_name) {
        return Err(error(StatusCode::NOT_FOUND, "unknown model"));
    }
    if !body["messages"].as_array().is_some_and(|messages| !messages.is_empty()) {
        return Err(bad_param("messages", "messages must be a nonempty array"));
    }
    if let Some(param) = ["stop", "stop_token_ids", "response_format", "logit_bias", "logprobs",
        "presence_penalty", "frequency_penalty", "repetition_penalty", "bad_words", "min_tokens", "ignore_eos"]
        .into_iter().find(|p| !body[*p].is_null())
    {
        return Err(bad_param(param, format!("{param} is unavailable for this model")));
    }
    check_ranges(st, body)?;
    if n_of(body) != 1 {
        return Err(bad_param("n", "this model generates one choice"));
    }
    if !body["tools"].is_null() && !body["tools"].is_array() {
        return Err(bad_param("tools", "tools must be an array"));
    }
    match body["tool_choice"].as_str() {
        Some("none") => body["tools"] = Value::Null,
        Some("auto") => {}
        _ if body["tool_choice"].is_null() => {}
        _ => return Err(bad_param("tool_choice", "this model supports auto and none")),
    }
    let urls = media::image_parts(body)?;
    if !urls.is_empty() && !chat.vision {
        return Err(bad_param("messages", "vision is disabled"));
    }
    if urls.len() > st.mm.limit_images {
        return Err(bad_param("messages", format!("at most {} images per request", st.mm.limit_images)));
    }
    let images = if urls.is_empty() {
        Vec::new()
    } else {
        let st = st.clone();
        tokio::task::spawn_blocking(move || {
            if st.mm.allowed_domains.is_empty() && urls.iter().any(|(url, _)| !url.starts_with("data:")) {
                return Err(bad_param("messages", "remote images require --allowed-media-domains"));
            }
            let images = media::load_images(&st.mm, &urls, Instant::now() + st.mm.fetch_timeout)
                .map_err(|err| bad_param("messages", err))?;
            images.iter().map(|bytes| decode_image(bytes).map_err(|err| bad_param("messages", err.to_string()))).collect()
        }).await.map_err(|err| error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))??
    };
    let prompt = render_chat(&chat.path, body)
        .map_err(|err| bad_param("messages", format!("{err:#}")))?;
    let tokens = st.text.encode(&prompt).map_err(|err| bad_param("messages", err.to_string()))?;
    check_fits(st, tokens.len())?;
    let max_tokens = body.get("max_completion_tokens").filter(|v| !v.is_null()).or_else(|| body.get("max_tokens"));
    let max_tokens = match max_tokens.filter(|v| !v.is_null()) {
        Some(v) => v.as_u64().and_then(|n| usize::try_from(n).ok()).filter(|&n| n > 0 && n < st.max_context)
            .ok_or_else(|| bad_param("max_tokens", "generation budget must be positive and smaller than the context"))?,
        None => st.cfg.default_max_tokens,
    };
    let spec = Spec::of(st, body, max_tokens, None, false)?;
    let names: Vec<_> = body["tools"].as_array().into_iter().flatten()
        .filter_map(|tool| tool["function"]["name"].as_str().map(str::to_owned)).collect();
    let text = st.text.clone();
    Output::new(&text, &names).map_err(|err| error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    let totals = Totals::new(tokens.len());
    let (tx, rx) = unbounded_channel();
    let mut req = GenRequest::new(tokens.into_iter().map(i64::from).collect(), spec.params.clone(), tx, Arc::new(AtomicBool::new(false)));
    req.max_tokens = spec.max_tokens;
    req.images = images;
    let mut generation = submit_prepared(st, req, rx, &spec)?;
    let counts = totals.clone();
    let events = async_stream::stream! {
        let mut output = Output::new(&text, &names).expect("validated output markers");
        let mut seen = 0;
        yield Ok(ChatEvent::Role { choice: 0 });
        loop {
            let event = generation.next_event().await;
            counts.note(&generation, &mut seen);
            match event {
                GenEvent::Started { prompt_tokens, .. } => counts.prompt.store(prompt_tokens, Ordering::Relaxed),
                GenEvent::Token(id, _) if !text.eos.contains(&(id as u32)) => {
                    if let Some((text, reasoning)) = output.push(id as u32) {
                        yield Ok(text_event(text, reasoning));
                    }
                }
                GenEvent::Done(finish) => {
                    if let Some((text, reasoning)) = output.flush() {
                        yield Ok(text_event(text, reasoning));
                    }
                    let reason = if output.calls.is_empty() {
                        if matches!(finish, Finish::Length) { ChatFinishReason::Length } else { ChatFinishReason::Stop }
                    } else { ChatFinishReason::ToolCalls };
                    yield Ok(ChatEvent::Delta {
                        choice: 0, content: None, reasoning_content: None, logprobs: None, finish_reason: Some(reason),
                        tool_calls: (!output.calls.is_empty()).then_some(output.calls),
                    });
                    yield Ok(ChatEvent::Usage { prompt_tokens: generation.prompt_tokens as u32, completion_tokens: generation.completion_tokens as u64 });
                    break;
                }
                GenEvent::Error(message) => { yield Err(ApiError { message }); break; }
                _ => {}
            }
        }
    };
    Ok((Box::pin(events), totals))
}

fn text_event(text: String, reasoning: bool) -> ChatEvent {
    let (content, reasoning_content) = if reasoning { (None, Some(text)) } else { (Some(text), None) };
    ChatEvent::Delta { choice: 0, content, reasoning_content, tool_calls: None, finish_reason: None, logprobs: None }
}
struct Output<'a> {
    text: &'a Text,
    invoke: u32,
    names: &'a [String],
    calls: Vec<ChatToolCallDelta>,
    header: bool,
    invocation: bool,
    begin_reason: u32,
    begin_text: u32,
    end: u32,
    model: u32,
    reasoning: bool,
    pending: Vec<u8>,
}

impl<'a> Output<'a> {
    fn new(text: &'a Text, names: &'a [String]) -> Result<Self> {
        let id = |token| {
            text.token_id(token)
                .with_context(|| format!("Inkling tokenizer is missing {token}"))
        };
        Ok(Self {
            text,
            invoke: id("<|content_invoke_tool_json|>")?,
            names,
            calls: Vec::new(),
            header: false,
            invocation: false,
            begin_reason: id("<|content_thinking|>")?,
            begin_text: id("<|content_text|>")?,
            end: id("<|end_message|>")?,
            model: id("<|message_model|>")?,
            reasoning: false,
            pending: Vec::new(),
        })
    }

    fn push(&mut self, id: u32) -> Option<(String, bool)> {
        if id == self.invoke {
            self.pending.clear();
            self.header = false;
            self.invocation = true;
            return None;
        }
        if self.invocation {
            if id == self.end {
                self.finish_call();
            } else {
                self.pending.extend(self.text.token_bytes(id));
            }
            return None;
        }
        if id == self.model {
            let last = self.flush();
            self.header = !self.names.is_empty();
            return last;
        }
        if id == self.begin_reason || id == self.begin_text || id == self.end {
            if self.header {
                self.pending.clear();
                self.header = false;
            }
            let last = self.flush();
            self.reasoning = id == self.begin_reason;
            return last;
        }
        self.pending.extend(self.text.token_bytes(id));
        if self.header {
            return None;
        }
        match std::str::from_utf8(&self.pending) {
            Ok(_) => self.flush(),
            Err(e) if e.error_len().is_some() => self.flush(),
            Err(_) => None,
        }
    }

    fn finish_call(&mut self) {
        if let Ok(call) = serde_json::from_slice::<Value>(&self.pending)
            && let Some(name) = call["name"].as_str()
            && self.names.iter().any(|item| item == name)
            && call["args"].is_object()
        {
            self.calls.push(ChatToolCallDelta {
                index: self.calls.len() as u32,
                id: Some(format!("call_{}", uuid::Uuid::new_v4().simple())),
                name: Some(name.to_owned()),
                arguments: Some(call["args"].to_string()),
            });
        }
        self.pending.clear();
        self.invocation = false;
        self.reasoning = false;
    }

    fn flush(&mut self) -> Option<(String, bool)> {
        if self.pending.is_empty() || self.invocation || self.header {
            return None;
        }
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        Some((text, self.reasoning))
    }
}
