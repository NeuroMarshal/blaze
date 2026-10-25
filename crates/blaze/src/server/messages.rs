//! Anthropic Messages and count_tokens adapters over the chat pipeline.
//! Responses, streams and errors use the Messages API shapes.

use super::chat::{ChatOut, call};
use super::*;
use std::pin::Pin;

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

/// Anthropic's error body.
fn anthropic_error(status: StatusCode, message: impl Into<String>) -> Response {
    let kind = match status.as_u16() {
        400 | 413 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        529 => "overloaded_error",
        _ => "api_error",
    };
    (status, Json(json!({"type": "error", "error": {"type": kind, "message": message.into()}}))).into_response()
}

/// An error of the chat path in Anthropic's shape.
fn translate_error(status: StatusCode, v: &Value) -> Response {
    let msg = v.pointer("/error/message").and_then(Value::as_str).unwrap_or("request failed").to_owned();
    anthropic_error(status, msg)
}

pub(super) async fn create(State(st): State<Arc<AppState>>, bytes: Bytes) -> Response {
    let body = match parse_body(&bytes) {
        Ok(b) => b,
        Err(r) => return anthropic_error(StatusCode::BAD_REQUEST, error_text(r).await),
    };
    match run(st, body).await {
        Ok(r) => r,
        Err(r) => anthropic_error(r.status(), error_text(r).await),
    }
}

pub(super) async fn count_tokens(State(st): State<Arc<AppState>>, bytes: Bytes) -> Response {
    let body = match parse_body(&bytes) {
        Ok(b) => b,
        Err(r) => return anthropic_error(StatusCode::BAD_REQUEST, error_text(r).await),
    };
    let counted = async {
        let mut chat = chat_body(&body, false)?;
        let images = super::media::gather(&st, &chat).await?;
        Ok::<_, Response>(super::chat::prepare(&st, &mut chat, false, &images)?.tokens.len())
    };
    match counted.await {
        Ok(n) => Json(json!({"input_tokens": n})).into_response(),
        Err(r) => anthropic_error(r.status(), error_text(r).await),
    }
}

/// The message of an error response of the OpenAI shape.
async fn error_text(r: Response) -> String {
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap_or_default();
    serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|v| v.pointer("/error/message").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned())
}

/// Text of a tool_result's content (a string or text blocks; images are not carried into a tool message).
fn result_text(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"),
        Value::Null => String::new(),
        v => v.to_string(),
    }
}

/// An image block as a chat image part (base64 source as a data URL, or a URL).
fn image_part(b: &Value) -> Checked<Value> {
    let src = &b["source"];
    let url = match src.get("type").and_then(Value::as_str) {
        Some("base64") => format!("data:{};base64,{}", src["media_type"].as_str().unwrap_or("image/png"), src["data"].as_str().unwrap_or("")),
        Some("url") => src["url"].as_str().unwrap_or("").to_owned(),
        other => return Err(bad_param("messages", format!("image source {other:?} is not supported (base64, url)"))),
    };
    Ok(json!({"type": "image_url", "image_url": {"url": url}}))
}

/// The chat request of a Messages request.
fn chat_body(body: &Value, stream: bool) -> Checked<Value> {
    let mut messages: Vec<Value> = Vec::new();
    match body.get("system") {
        Some(Value::String(s)) => messages.push(json!({"role": "system", "content": s})),
        Some(Value::Array(a)) => {
            let text: String = a.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n");
            messages.push(json!({"role": "system", "content": text}));
        }
        _ => {}
    }
    for m in body.get("messages").and_then(Value::as_array).ok_or_else(|| bad_param("messages", "messages is required"))? {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let blocks: Vec<Value> = match m.get("content") {
            Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
            Some(Value::Array(a)) => a.clone(),
            _ => Vec::new(),
        };
        if role == "assistant" {
            let (mut text, mut thinking, mut calls) = (String::new(), String::new(), Vec::new());
            for b in &blocks {
                match b.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => text += b["text"].as_str().unwrap_or(""),
                    "thinking" => thinking += b["thinking"].as_str().unwrap_or(""),
                    "redacted_thinking" => {}
                    "tool_use" => calls.push(json!({"id": b["id"], "type": "function",
                                                    "function": {"name": b["name"], "arguments": b.get("input").unwrap_or(&json!({})).to_string()}})),
                    other => return Err(bad_param("messages", format!("assistant block '{other}' is not supported"))),
                }
            }
            let mut msg = json!({"role": "assistant", "content": if text.is_empty() && !calls.is_empty() { Value::Null } else { json!(text) }});
            if !thinking.is_empty() {
                msg["reasoning_content"] = json!(thinking);
            }
            if !calls.is_empty() {
                msg["tool_calls"] = json!(calls);
            }
            messages.push(msg);
            continue;
        }
        // a user turn: its tool results become tool messages (first, as they answer the turn before), the rest a
        // user message
        let mut parts = Vec::new();
        for b in &blocks {
            match b.get("type").and_then(Value::as_str).unwrap_or("") {
                "text" => parts.push(json!({"type": "text", "text": b["text"]})),
                "image" => parts.push(image_part(b)?),
                "tool_result" => {
                    let mut content = result_text(b.get("content").unwrap_or(&Value::Null));
                    if b.get("is_error").and_then(Value::as_bool) == Some(true) {
                        content = format!("Error: {content}");
                    }
                    messages.push(json!({"role": "tool", "tool_call_id": b["tool_use_id"], "content": content}));
                }
                "document" | "search_result" => return Err(bad_param("messages", "document blocks are not supported")),
                other => return Err(bad_param("messages", format!("user block '{other}' is not supported"))),
            }
        }
        if !parts.is_empty() {
            messages.push(json!({"role": "user", "content": parts}));
        }
    }
    let mut chat = json!({"messages": messages});
    let o = chat.as_object_mut().unwrap();
    for (from, to) in [("model", "model"), ("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")] {
        if let Some(v) = body.get(from).filter(|v| !v.is_null()) {
            o.insert(to.into(), v.clone());
        }
    }
    if let Some(s) = body.get("stop_sequences").filter(|v| v.is_array()) {
        o.insert("stop".into(), s.clone());
    }
    if let Some(tools) = body.get("tools").and_then(Value::as_array).filter(|t| !t.is_empty()) {
        let tools: Vec<Value> = tools
            .iter()
            .filter(|t| t.get("type").is_none_or(|k| k == "custom" || k.is_null()))
            .map(|t| json!({"type": "function", "function": {"name": t["name"], "description": t.get("description").cloned().unwrap_or(Value::Null),
                                                           "parameters": t.get("input_schema").cloned().unwrap_or(json!({"type": "object"}))}}))
            .collect();
        if !tools.is_empty() {
            o.insert("tools".into(), json!(tools));
        }
    }
    if let Some(tc) = body.get("tool_choice").filter(|v| !v.is_null()) {
        let choice = match tc.get("type").and_then(Value::as_str) {
            Some("auto") => json!("auto"),
            Some("any") => json!("required"),
            Some("none") => json!("none"),
            Some("tool") => json!({"type": "function", "function": {"name": tc["name"]}}),
            other => return Err(bad_param("tool_choice", format!("tool_choice type {other:?}"))),
        };
        o.insert("tool_choice".into(), choice);
        if tc.get("disable_parallel_tool_use").and_then(Value::as_bool) == Some(true) {
            o.insert("parallel_tool_calls".into(), json!(false));
        }
    }
    // thinking only when asked for (the API's default); budget_tokens is the thinking budget
    let thinking = body.get("thinking").filter(|t| t.get("type").and_then(Value::as_str) == Some("enabled"));
    o.insert("chat_template_kwargs".into(), json!({"enable_thinking": thinking.is_some()}));
    if let Some(b) = thinking.and_then(|t| t.get("budget_tokens")).and_then(Value::as_u64) {
        o.insert("thinking_token_budget".into(), json!(b));
    }
    if stream {
        o.insert("stream".into(), json!(true));
        o.insert("stream_options".into(), json!({"include_usage": true}));
    }
    Ok(chat)
}

fn stop_reason(finish: &Value, has_calls: bool) -> &'static str {
    match finish.as_str() {
        Some("length") => "max_tokens",
        Some("tool_calls") => "tool_use",
        _ if has_calls => "tool_use",
        _ => "end_turn",
    }
}

fn usage_of(u: &Value) -> Value {
    json!({"input_tokens": u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
           "output_tokens": u.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
           "cache_read_input_tokens": u.pointer("/prompt_tokens_details/cached_tokens").and_then(Value::as_u64).unwrap_or(0),
           "cache_creation_input_tokens": 0})
}

/// A tool call's arguments as the tool_use block's input object.
fn input_of(args: &str) -> Value {
    serde_json::from_str::<Value>(args).ok().filter(Value::is_object).unwrap_or(json!({}))
}

/// The content blocks of a chat message.
fn blocks_of(m: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(r) = m.get("reasoning_content").and_then(Value::as_str).filter(|r| !r.is_empty()) {
        out.push(json!({"type": "thinking", "thinking": r, "signature": ""}));
    }
    if let Some(t) = m.get("content").and_then(Value::as_str).filter(|t| !t.trim().is_empty()) {
        out.push(json!({"type": "text", "text": t}));
    }
    for c in m.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
        out.push(json!({"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                        "input": input_of(c["function"]["arguments"].as_str().unwrap_or("{}"))}));
    }
    out
}

async fn run(st: Arc<AppState>, body: Value) -> Checked<Response> {
    if body.get("max_tokens").and_then(Value::as_u64).is_none() {
        return Err(bad_param("max_tokens", "max_tokens is required"));
    }
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let chat = chat_body(&body, stream)?;
    let id = new_id("msg");
    let model = body.get("model").cloned().unwrap_or(json!(st.cfg.served_name));
    match call(st, chat).await {
        ChatOut::Json(status, v) if !status.is_success() => Ok(translate_error(status, &v)),
        ChatOut::Json(_, v) => {
            let choice = &v["choices"][0];
            let mut message = choice["message"].clone();
            super::responses::unclosed_answer(&mut message, &choice["finish_reason"]);
            let content = blocks_of(&message);
            let has_calls = content.iter().any(|b| b["type"] == "tool_use");
            Ok(Json(json!({"id": id, "type": "message", "role": "assistant", "model": model, "content": content,
                           "stop_reason": stop_reason(&choice["finish_reason"], has_calls), "stop_sequence": null,
                           "usage": usage_of(&v["usage"])}))
            .into_response())
        }
        ChatOut::Stream(chunks) => Ok(sse(stream_events(id, model, chunks))),
    }
}

fn sse(s: impl Stream<Item = Value> + Send + 'static) -> Response {
    let events = s.map(|v| {
        let kind = v.get("type").and_then(Value::as_str).unwrap_or("message").to_owned();
        Ok::<_, Infallible>(Event::default().event(kind).data(v.to_string()))
    });
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

/// The Messages stream of a chat chunk stream: a content block per kind of output (thinking, text, each tool call),
/// opened as its first delta comes and stopped when the next one starts.
fn stream_events(id: String, model: Value, mut chunks: Pin<Box<dyn Stream<Item = Value> + Send>>) -> impl Stream<Item = Value> + Send {
    async_stream::stream! {
        yield json!({"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant", "model": model,
                     "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 0, "output_tokens": 0}}});
        yield json!({"type": "ping"});
        // the open block: (kind, index); tool call index -> its block
        let mut open: Option<(&'static str, usize)> = None;
        let mut next = 0usize;
        let mut calls: std::collections::HashMap<u64, usize> = Default::default();
        let (mut usage, mut finish) = (Value::Null, Value::Null);
        while let Some(c) = chunks.next().await {
            if let Some(e) = c.get("error") {
                yield json!({"type": "error", "error": {"type": "api_error", "message": e.get("message").cloned().unwrap_or(json!("error"))}});
                return;
            }
            if !c["usage"].is_null() {
                usage = c["usage"].clone();
            }
            let Some(choice) = c["choices"].get(0) else { continue };
            if !choice["finish_reason"].is_null() {
                finish = choice["finish_reason"].clone();
            }
            let d = &choice["delta"];
            if let Some(r) = d.get("reasoning_content").and_then(Value::as_str).filter(|r| !r.is_empty()) {
                if open.is_none_or(|o| o.0 != "thinking") {
                    if let Some((k, i)) = open.take() {
                        if k == "thinking" { yield json!({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": ""}}); }
                        yield json!({"type": "content_block_stop", "index": i});
                    }
                    open = Some(("thinking", next));
                    yield json!({"type": "content_block_start", "index": next, "content_block": {"type": "thinking", "thinking": "", "signature": ""}});
                    next += 1;
                }
                yield json!({"type": "content_block_delta", "index": open.unwrap().1, "delta": {"type": "thinking_delta", "thinking": r}});
            }
            if let Some(t) = d.get("content").and_then(Value::as_str).filter(|t| !t.is_empty()) {
                if open.is_none_or(|o| o.0 != "text") && !t.trim().is_empty() {
                    if let Some((k, i)) = open.take() {
                        if k == "thinking" { yield json!({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": ""}}); }
                        yield json!({"type": "content_block_stop", "index": i});
                    }
                    open = Some(("text", next));
                    yield json!({"type": "content_block_start", "index": next, "content_block": {"type": "text", "text": ""}});
                    next += 1;
                }
                if let Some((_, i)) = open.filter(|o| o.0 == "text") {
                    yield json!({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": t}});
                }
            }
            for tc in d.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                let ti = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
                if !calls.contains_key(&ti) {
                    if let Some((k, i)) = open.take() {
                        if k == "thinking" { yield json!({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": ""}}); }
                        yield json!({"type": "content_block_stop", "index": i});
                    }
                    let call_id = tc.get("id").cloned().filter(|v| !v.is_null()).unwrap_or(json!(new_id("toolu")));
                    calls.insert(ti, next);
                    open = Some(("tool_use", next));
                    yield json!({"type": "content_block_start", "index": next, "content_block": {"type": "tool_use", "id": call_id,
                                 "name": tc.pointer("/function/name").cloned().unwrap_or(json!("")), "input": {}}});
                    next += 1;
                }
                if let Some(a) = tc.pointer("/function/arguments").and_then(Value::as_str).filter(|a| !a.is_empty()) {
                    yield json!({"type": "content_block_delta", "index": calls[&ti], "delta": {"type": "input_json_delta", "partial_json": a}});
                }
            }
        }
        if let Some((k, i)) = open.take() {
            if k == "thinking" { yield json!({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": ""}}); }
            yield json!({"type": "content_block_stop", "index": i});
        }
        let u = usage_of(&usage);
        yield json!({"type": "message_delta", "delta": {"stop_reason": stop_reason(&finish, !calls.is_empty()), "stop_sequence": null},
                     "usage": {"output_tokens": u["output_tokens"], "input_tokens": u["input_tokens"], "cache_read_input_tokens": u["cache_read_input_tokens"]}});
        yield json!({"type": "message_stop"});
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tool_loop_becomes_chat_messages() {
        let body = json!({"model": "m", "max_tokens": 100, "system": [{"type": "text", "text": "Be brief."}],
            "messages": [
                {"role": "user", "content": "Weather in Oslo?"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "use the tool"},
                                                  {"type": "tool_use", "id": "t1", "name": "w", "input": {"city": "Oslo"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "sunny"}]},
                                             {"type": "text", "text": "And now?"}]},
            ],
            "tools": [{"name": "w", "description": "weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"}, "stop_sequences": ["END"], "thinking": {"type": "enabled", "budget_tokens": 64}});
        let chat = chat_body(&body, true).unwrap();
        let m = chat["messages"].as_array().unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "Be brief."}));
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Oslo\"}");
        assert_eq!(m[2]["reasoning_content"], "use the tool");
        assert_eq!(m[3], json!({"role": "tool", "tool_call_id": "t1", "content": "sunny"}));
        assert_eq!(m[4]["role"], "user");
        assert_eq!(chat["tool_choice"], "required");
        assert_eq!(chat["stop"], json!(["END"]));
        assert_eq!(chat["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(chat["thinking_token_budget"], 64);
        assert_eq!(chat["stream_options"]["include_usage"], true);
    }

    #[test]
    fn a_chat_answer_becomes_blocks() {
        let b = blocks_of(&json!({"reasoning_content": "hm", "content": "Hi",
                                  "tool_calls": [{"id": "c", "function": {"name": "w", "arguments": "{\"a\":1}"}}]}));
        assert_eq!(b.iter().map(|x| x["type"].as_str().unwrap()).collect::<Vec<_>>(), ["thinking", "text", "tool_use"]);
        assert_eq!(b[2]["input"], json!({"a": 1}));
        assert_eq!(stop_reason(&json!("stop"), true), "tool_use");
        assert_eq!(stop_reason(&json!("length"), false), "max_tokens");
    }
}
