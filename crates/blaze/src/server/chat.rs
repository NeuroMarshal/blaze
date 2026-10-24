//! Chat completions: template rendering, scheduler submission and streamed reasoning/tool-call events.

use super::*;
use blaze_chat::dynamo_protocols::types::{
    ChatChoiceLogprobs, ChatCompletionRequestMessage, ChatCompletionTokenLogprob, ChatCompletionToolChoiceOption as Choice,
    CreateChatCompletionRequest, TopLogprobs,
};
use blaze_chat::dynamo_renderer::{OAIChatLikeRequest, TextInput};
use blaze_chat::{ChatEvent, ChatFinishReason, ChatResponseProcessor, DecodedChatEvent};
use std::collections::HashMap;
use std::pin::Pin;

/// The request with the template kwargs the client sent (chat_template_kwargs), for the renderer.
struct TemplateRequest<'a> {
    request: &'a CreateChatCompletionRequest,
    kwargs: Option<&'a HashMap<String, Value>>,
    /// the template ends with the opening of an assistant turn (the renderer does not take it from the kwargs)
    add_generation_prompt: bool,
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
        self.add_generation_prompt
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

/// A chat request turned into a prompt.
pub(super) struct Prepared {
    pub request: CreateChatCompletionRequest,
    pub tokens: Vec<i64>,
    /// tools are listed and tool_choice is not "none": the output is searched for calls
    pub tools_enabled: bool,
    /// the prompt ends inside a reasoning section (the output starts with reasoning)
    pub starts_in_reasoning: bool,
    /// tool_choice forces a call: a grammar for the start of the answer
    pub forced_call: Option<String>,
    /// where each image's rows start in `tokens` (one per image, in prompt order)
    pub image_starts: Vec<usize>,
}

/// Text of a message's content (a string, or the text parts of a list).
fn content_text(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) => a.iter().filter_map(|p| p.get("text")?.as_str()).collect(),
        _ => String::new(),
    }
}

/// A grammar for the answer when tool_choice forces a call (required: any listed tool; a named function): the parser's
/// call syntax around the tool's name, whatever the model writes for the arguments up to the closing tag. `single`:
/// one call only (parallel_tool_calls false). None for parsers whose syntax is not known here (the model is then only
/// asked). The body is "any text without the closing tag", an intersection llguidance builds as a DFA.
/// The whitespace around the tags is bounded: the model writes "\n\n" after </think> and "\n" between calls, and a
/// model that would rather answer in words than call (a greeting with thinking off) finds whitespace the likeliest of
/// the allowed tokens; with `ws*` every such token leaves the grammar where it was and greedy decoding never leaves.
fn forced_call_grammar(parser: Option<&str>, names: &[String], single: bool) -> Option<String> {
    let (open, mid, close) = match parser? {
        "qwen3_coder" => ("<tool_call>", "<function=", ">"),
        "qwen25" | "qwen" | "hermes" => ("<tool_call>", "{\"name\": \"", "\""),
        _ => return None,
    };
    let q = |s: &str| serde_json::to_string(s).unwrap_or_default();
    let names = names.iter().map(|n| q(n)).collect::<Vec<_>>().join(" | ");
    let more = if single { "" } else { " (ws call)*" };
    Some(format!(
        "start: ws call{more} ws\ncall: {} ws {} name {} BODY {}\nname: {names}\nws: /[ \\t\\r\\n]{{0,4}}/\n\
         BODY: ~(/(?s:.*)/ {} /(?s:.*)/)\n",
        q(open), q(mid), q(close), q("</tool_call>"), q("</tool_call>")
    ))
}

/// The prompt of a chat request body (`model` filled in when absent): template, tools, tool_choice, thinking switch,
/// add_generation_prompt / continue_final_message; its tokens within the context.
pub(super) fn prepare(st: &AppState, body: &mut Value, enforce_context: bool, images: &[super::media::Part]) -> Checked<Prepared> {
    let Some(formatter) = st.formatter.as_ref() else { return Err(error(StatusCode::BAD_REQUEST, "this model has no chat template")) };
    if body.get("model").is_none_or(Value::is_null) {
        body["model"] = json!(st.cfg.served_name);
    }
    if st.developer_as_system
        && let Some(msgs) = body.get_mut("messages").and_then(Value::as_array_mut)
    {
        for m in msgs.iter_mut().filter(|m| m.get("role").and_then(Value::as_str) == Some("developer")) {
            m["role"] = json!("system");
        }
    }
    // a content of text parts only is their text, joined as the model's template joins them (no separator; the
    // renderer would put a newline between them); parts with images stay parts
    if let Some(msgs) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for m in msgs.iter_mut() {
            let Some(parts) = m.get("content").and_then(Value::as_array) else { continue };
            let texts: Option<Vec<&str>> = parts
                .iter()
                .map(|p| p.get("type").and_then(Value::as_str).filter(|t| *t == "text").and(p.get("text")).and_then(Value::as_str))
                .collect();
            if let Some(texts) = texts.filter(|t| !t.is_empty()) {
                m["content"] = json!(texts.concat());
            }
        }
    }
    let request: CreateChatCompletionRequest = serde_json::from_value(body.clone())
        .map_err(|e| error(StatusCode::BAD_REQUEST, format!("bad request: {e}")))?;
    if request.messages.is_empty() {
        return Err(bad_param("messages", "messages cannot be empty"));
    }
    let tools = blaze_chat::chat_tool_definitions(request.tools.as_deref(), &request.messages)
        .map_err(|e| error(StatusCode::BAD_REQUEST, e.to_string()))?;
    let forced_names: Option<Vec<String>> = match &request.tool_choice {
        Some(Choice::Required) if tools.is_empty() => {
            return Err(bad_param("tool_choice", "tool_choice is \"required\" but tools is empty"));
        }
        Some(Choice::Required) => Some(tools.iter().map(|t| t.name.clone()).collect()),
        Some(Choice::Named(c)) if !tools.iter().any(|t| t.name == c.function.name) => {
            return Err(bad_param("tool_choice", format!("tool named \"{}\" in tool_choice is not present in tools", c.function.name)));
        }
        Some(Choice::Named(c)) => Some(vec![c.function.name.clone()]),
        _ => None,
    };
    let tools_enabled = !tools.is_empty() && !matches!(request.tool_choice, Some(Choice::None));
    let named = matches!(request.tool_choice, Some(Choice::Named(_)));
    let forced_call = forced_names
        .filter(|_| tools_enabled)
        .and_then(|names| forced_call_grammar(st.cfg.tool_parser.as_deref(), &names, request.parallel_tool_calls == Some(false)));
    let mut kwargs: Option<HashMap<String, Value>> =
        body.get("chat_template_kwargs").and_then(|v| serde_json::from_value(v.clone()).ok());
    // vLLM's add_generation_prompt / continue_final_message: continuing an assistant message needs the template to
    // leave its turn open, which is done by rendering it closed and cutting the prompt after its text
    let add_generation = body.get("add_generation_prompt").and_then(Value::as_bool);
    let continue_final = body.get("continue_final_message").and_then(Value::as_bool).unwrap_or(false);
    let last = body.get("messages").and_then(Value::as_array).and_then(|m| m.last());
    if continue_final {
        if add_generation == Some(true) {
            return Err(bad_param("continue_final_message", "continue_final_message and add_generation_prompt cannot both be true"));
        }
        if last.and_then(|m| m.get("role")).and_then(Value::as_str) != Some("assistant") {
            return Err(bad_param("continue_final_message", "continue_final_message needs the last message to be an assistant message"));
        }
    }
    let tail = continue_final.then(|| last.map(content_text).unwrap_or_default());
    if tail.as_deref() == Some("") {
        return Err(bad_param("continue_final_message", "the last message has no text to continue"));
    }
    // SGLang's normalize_reasoning_inputs: reasoning_effort and reasoning.{effort,enabled} switch thinking and set the
    // effort the template announces; explicit chat_template_kwargs win
    let (thinking, effort) = (blaze_chat::requested_thinking(body), blaze_chat::requested_effort(body).cloned());
    if thinking.is_some() || effort.is_some() {
        let k = kwargs.get_or_insert_default();
        if let Some(t) = thinking {
            k.entry("thinking".into()).or_insert(t.into());
            k.entry("enable_thinking".into()).or_insert(t.into());
        }
        if let Some(e) = effort {
            k.entry("reasoning_effort".into()).or_insert(e);
        }
    }
    let _thinking = formatter.resolve_thinking(&mut kwargs, tools_enabled, named);
    let mut prompt = formatter
        .render_prompt(&TemplateRequest { request: &request, kwargs: kwargs.as_ref(), add_generation_prompt: add_generation.unwrap_or(true) && !continue_final })
        .map_err(|e| error(StatusCode::BAD_REQUEST, format!("chat template render failed: {e}")))?
        .into_text();
    if let Some(tail) = tail {
        let Some(at) = prompt.rfind(&tail) else {
            return Err(bad_param("continue_final_message", "the last message's text is not in the rendered prompt"));
        };
        prompt.truncate(at + tail.len());
    }
    let mut tokens: Vec<i64> = st.text.encode(&prompt).map_err(|e| error(StatusCode::BAD_REQUEST, e.to_string()))?.into_iter().map(i64::from).collect();
    let image_starts = if images.is_empty() {
        truncate_prompt(st, body, &mut tokens);
        Vec::new()
    } else {
        if body.get("truncate_prompt_tokens").is_some_and(|v| !v.is_null()) {
            return Err(bad_param("truncate_prompt_tokens", "truncate_prompt_tokens cannot be combined with images"));
        }
        let (t, starts) = super::media::expand(st, tokens, images)?;
        tokens = t;
        starts
    };
    if enforce_context {
        check_fits(st, tokens.len())?;
    }
    let starts_in_reasoning = !continue_final
        && matches!(st.cfg.reasoning_parser.as_deref(), Some("qwen3" | "deepseek-r1" | "glm45"))
        && prompt.trim_end_matches('\n').ends_with("<think>");
    Ok(Prepared { request, tokens, tools_enabled, starts_in_reasoning, forced_call, image_starts })
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

/// OpenAI logprobs entries of generated tokens.
fn chat_logprobs(text: &Text, items: &[(u32, TokenLogprob)]) -> Option<ChatChoiceLogprobs> {
    if items.is_empty() {
        return None;
    }
    let content = items
        .iter()
        .map(|(t, l)| {
            let (token, bytes) = text.token_str(*t);
            ChatCompletionTokenLogprob {
                bytes: Some(bytes),
                token,
                logprob: l.logprob,
                token_id: None,
                top_logprobs: l
                    .top
                    .iter()
                    .map(|&(t, lp)| {
                        let (token, bytes) = text.token_str(t as u32);
                        TopLogprobs { bytes: Some(bytes), token, logprob: lp }
                    })
                    .collect(),
            }
        })
        .collect();
    Some(ChatChoiceLogprobs { content: Some(content), refusal: None })
}

type Decoded = Pin<Box<dyn Stream<Item = std::result::Result<DecodedChatEvent, ApiError>> + Send>>;

/// One choice's generation as decoded chat events (prompt tokens ride on every event; counts are per event).
fn decoded_events(mut g: Gen, choice: usize, totals: Arc<Totals>, text: Arc<Text>) -> Decoded {
    let prompt_tokens = totals.prompt as u32;
    Box::pin(async_stream::stream! {
        let mut seen = 0;
        while let Some(piece) = g.next().await {
            totals.note(&g, &mut seen);
            match piece {
                Piece::Text(t, lps) => yield Ok(DecodedChatEvent { choice, text: t, token_ids: Vec::new(), finish_reason: None,
                    logprobs: chat_logprobs(&text, &lps), prompt_tokens, completion_tokens: 1 }),
                Piece::Finish(f) => {
                    yield Ok(DecodedChatEvent { choice, text: String::new(), token_ids: Vec::new(), finish_reason: Some(finish_of(f)),
                        logprobs: None, prompt_tokens, completion_tokens: 0 });
                    break;
                }
                Piece::Error(e) => { yield Err(ApiError { message: e }); break; }
            }
        }
    })
}

pub(super) async fn chat(State(st): State<Arc<AppState>>, bytes: Bytes) -> Response {
    let mut body = match parse_body(&bytes) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match run(st, &mut body).await {
        Ok(r) | Err(r) => r,
    }
}

async fn run(st: Arc<AppState>, body: &mut Value) -> Checked<Response> {
    check_ranges(&st, body)?;
    let images = super::media::gather(&st, body).await?;
    let p = prepare(&st, body, true, &images)?;
    let n = n_of(body);
    #[allow(deprecated)]
    let max_tokens = p.request.max_completion_tokens.or(p.request.max_tokens).map(|x| x as usize).unwrap_or(st.cfg.default_max_tokens);
    let stream = p.request.stream.unwrap_or(false);
    let include_usage = p.request.stream_options.as_ref().is_some_and(|o| o.include_usage);
    let include_reasoning = body.get("include_reasoning").and_then(Value::as_bool).unwrap_or(true);
    let mut spec = Spec::of(&st, body, max_tokens, think_budget(&st, body, p.starts_in_reasoning));
    // images: encoded now (the prompt fits the context), their rows in the token list become prefix-cache keys
    let mut tokens = p.tokens.clone();
    if !images.is_empty() {
        let mm = super::media::encode(&st, images, p.image_starts.clone()).await?;
        mm.keyed(&mut tokens);
        spec.mm = Some(mm);
    }
    // with reasoning first, the grammar applies to the answer after </think>
    let after = p.starts_in_reasoning.then(|| st.text.token_id("</think>")).flatten().map(|t| t as i32);
    let gens = submit_n(&st, &tokens, &spec, n, 0, &|| constraint_of(&st, body, after, p.forced_call.as_deref()))?;
    let tools = blaze_chat::chat_tool_definitions(p.request.tools.as_deref(), &p.request.messages)
        .map_err(|e| error(StatusCode::BAD_REQUEST, e.to_string()))?;
    // SGLang's separate_reasoning=false leaves the reasoning in the content
    let separate = body.get("separate_reasoning").and_then(Value::as_bool).unwrap_or(true);
    let processor = ChatResponseProcessor::new(
        p.tools_enabled.then(|| st.cfg.tool_parser.clone()).flatten(),
        st.cfg.reasoning_parser.clone().filter(|_| separate),
        (!tools.is_empty()).then_some(tools),
        p.request.tool_choice.clone(),
        false,
        p.request.parallel_tool_calls.unwrap_or(true),
        n,
    )
    .with_reasoning_state((p.starts_in_reasoning && separate).then_some(true));
    let totals = Totals::new(p.tokens.len());
    let decoded = futures::stream::select_all(
        gens.into_iter().enumerate().map(|(i, g)| decoded_events(g, i, totals.clone(), st.text.clone())).collect::<Vec<_>>(),
    );
    let mut events = processor.process_stream(decoded);
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let model = st.cfg.served_name.clone();
    let created = now();
    if stream {
        let s = async_stream::stream! {
            let chunk = |choices: Value, usage: Option<Value>| {
                let mut c = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": choices});
                if include_usage { c["usage"] = usage.unwrap_or(Value::Null); }
                c.to_string()
            };
            let mut had_calls = vec![false; n];
            while let Some(ev) = events.next().await {
                match ev {
                    Ok(ChatEvent::Role { choice }) => yield chunk(json!([{"index": choice, "delta": {"role": "assistant", "content": ""},
                                                                         "logprobs": null, "finish_reason": null}]), None),
                    Ok(ChatEvent::Delta { choice, content, reasoning_content, tool_calls, finish_reason, logprobs }) => {
                        let mut delta = serde_json::Map::new();
                        // blank text between or after calls is the parser's leftover newlines, not content
                        let content = content.filter(|c| !(had_calls[choice] && c.trim().is_empty()));
                        if let Some(c) = content { delta.insert("content".into(), json!(c)); }
                        if let Some(r) = reasoning_content.filter(|_| include_reasoning) { delta.insert("reasoning_content".into(), json!(r)); }
                        if let Some(tc) = tool_calls {
                            had_calls[choice] = true;
                            // OpenAI: id and type on the first delta of a call, then argument fragments (one entry per
                            // call and chunk)
                            let mut v: Vec<Value> = Vec::new();
                            for t in &tc {
                                let fragment = t.id.is_none() && t.name.is_none();
                                match v.last_mut().filter(|l| fragment && l["index"] == json!(t.index)) {
                                    Some(last) => {
                                        let prev = last["function"]["arguments"].as_str().unwrap_or("").to_string();
                                        last["function"]["arguments"] = json!(prev + t.arguments.as_deref().unwrap_or(""));
                                    }
                                    None => {
                                        let mut o = json!({"index": t.index, "function": {}});
                                        if let Some(id) = &t.id { o["id"] = json!(id); o["type"] = json!("function"); }
                                        if let Some(n) = &t.name { o["function"]["name"] = json!(n); }
                                        if let Some(a) = &t.arguments { o["function"]["arguments"] = json!(a); }
                                        v.push(o);
                                    }
                                }
                            }
                            delta.insert("tool_calls".into(), json!(v));
                        }
                        let logprobs = logprobs.and_then(|l| serde_json::to_value(l).ok()).unwrap_or(Value::Null);
                        if delta.is_empty() && finish_reason.is_none() && logprobs.is_null() { continue; }
                        yield chunk(json!([{"index": choice, "delta": delta, "logprobs": logprobs,
                                            "finish_reason": finish_reason.map(finish_name)}]), None);
                    }
                    Ok(ChatEvent::Usage { .. }) => {
                        if include_usage { yield chunk(json!([]), Some(totals.usage())); }
                    }
                    Err(e) => { yield stream_error(&e.message); break; }
                }
            }
            yield "[DONE]".to_string();
        };
        return Ok(sse(s));
    }
    // one message per choice
    #[derive(Default)]
    struct Acc {
        content: String,
        reasoning: String,
        calls: Vec<Value>,
        logprobs: Vec<ChatCompletionTokenLogprob>,
        finish: Option<ChatFinishReason>,
    }
    let mut accs: Vec<Acc> = (0..n).map(|_| Acc::default()).collect();
    while let Some(ev) = events.next().await {
        match ev {
            Ok(ChatEvent::Delta { choice, content, reasoning_content, tool_calls, finish_reason, logprobs }) => {
                let Some(a) = accs.get_mut(choice) else { continue };
                a.content.extend(content);
                a.reasoning.extend(reasoning_content);
                for t in tool_calls.unwrap_or_default() {
                    let i = t.index as usize;
                    // the parser numbers calls from 0 in order: an index past the next one is a parser fault, not
                    // a reason to grow the list
                    if i > a.calls.len() {
                        eprintln!("blaze: tool call index {i} after {} calls dropped", a.calls.len());
                        continue;
                    }
                    while a.calls.len() <= i {
                        a.calls.push(json!({"id": "", "type": "function", "function": {"name": "", "arguments": ""}}));
                    }
                    if let Some(id) = t.id {
                        a.calls[i]["id"] = json!(id);
                    }
                    if let Some(n) = t.name {
                        a.calls[i]["function"]["name"] = json!(n);
                    }
                    if let Some(frag) = t.arguments {
                        let prev = a.calls[i]["function"]["arguments"].as_str().unwrap_or("").to_string();
                        a.calls[i]["function"]["arguments"] = json!(prev + &frag);
                    }
                }
                a.logprobs.extend(logprobs.and_then(|l| l.content).unwrap_or_default());
                a.finish = finish_reason.or(a.finish);
            }
            Ok(_) => {}
            Err(e) => return Err(error(StatusCode::INTERNAL_SERVER_ERROR, e.message)),
        }
    }
    let choices: Vec<Value> = accs
        .into_iter()
        .enumerate()
        .map(|(i, a)| {
            // a call comes after at most blank text: the content is then null, as the engines return it
            let content = if a.calls.is_empty() { a.content.as_str() } else { a.content.trim_end() };
            let mut message = json!({"role": "assistant", "content": if content.trim().is_empty() && !a.calls.is_empty() { Value::Null } else { json!(content) }});
            if !a.reasoning.is_empty() && include_reasoning {
                message["reasoning_content"] = json!(a.reasoning);
            }
            if !a.calls.is_empty() {
                message["tool_calls"] = json!(a.calls);
            }
            let logprobs = if a.logprobs.is_empty() {
                Value::Null
            } else {
                json!({"content": a.logprobs, "refusal": null})
            };
            json!({"index": i, "message": message, "logprobs": logprobs, "finish_reason": a.finish.map(finish_name).unwrap_or("stop")})
        })
        .collect();
    Ok(Json(json!({"id": id, "object": "chat.completion", "created": created, "model": model, "choices": choices,
                   "usage": totals.usage()}))
    .into_response())
}
