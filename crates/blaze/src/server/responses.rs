//! OpenAI Responses adapter over the chat pipeline.
//! Recent responses stay in memory for GET by id and previous_response_id.

use super::chat::{ChatOut, call};
use super::*;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Mutex;

/// Responses kept for previous_response_id: their conversation (chat messages) and the response object.
const KEPT: usize = 256;

struct Kept {
    id: String,
    messages: Vec<Value>,
    response: Value,
}

fn store() -> &'static Mutex<VecDeque<Kept>> {
    static S: std::sync::OnceLock<Mutex<VecDeque<Kept>>> = std::sync::OnceLock::new();
    S.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

pub(super) async fn create(State(st): State<Arc<AppState>>, bytes: Bytes) -> Response {
    let body = match parse_body(&bytes) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match run(st, body).await {
        Ok(r) | Err(r) => r,
    }
}

pub(super) async fn get_one(Path(id): Path<String>) -> Response {
    let kept = store().lock().ok().and_then(|s| s.iter().find(|k| k.id == id).map(|k| k.response.clone()));
    match kept {
        Some(r) => Json(r).into_response(),
        None => api_error(StatusCode::NOT_FOUND, format!("no response with id '{id}'"), None, Some("response_id")),
    }
}

/// Content parts of a Responses message as chat content (a string when it is text only).
fn content_of(c: &Value) -> Checked<Value> {
    let Some(parts) = c.as_array() else {
        return Ok(c.as_str().map_or(Value::Null, |s| json!(s)));
    };
    let mut out = Vec::new();
    for p in parts {
        match p.get("type").and_then(Value::as_str).unwrap_or("") {
            "input_text" | "output_text" | "text" | "summary_text" => out.push(json!({"type": "text", "text": p.get("text").cloned().unwrap_or(json!(""))})),
            "input_image" => {
                let url = p.get("image_url").and_then(|u| u.as_str().map(str::to_owned).or_else(|| u.get("url").and_then(Value::as_str).map(str::to_owned)));
                let Some(url) = url else { return Err(bad_param("input", "an input_image needs an image_url (file ids are not supported)")) };
                let mut img = json!({"type": "image_url", "image_url": {"url": url}});
                if let Some(d) = p.get("detail").filter(|d| !d.is_null()) {
                    img["image_url"]["detail"] = d.clone();
                }
                out.push(img);
            }
            "refusal" => out.push(json!({"type": "text", "text": p.get("refusal").cloned().unwrap_or(json!(""))})),
            other => return Err(bad_param("input", format!("content part '{other}' is not supported"))),
        }
    }
    Ok(json!(out))
}

/// The input items as chat messages: messages, function calls (consecutive ones in one assistant message) and their
/// outputs; reasoning items are left out (the template does not take earlier reasoning from them).
fn messages_of(input: &Value) -> Checked<Vec<Value>> {
    let items: Vec<Value> = match input {
        Value::String(s) => return Ok(vec![json!({"role": "user", "content": s})]),
        Value::Array(a) => a.clone(),
        Value::Null => Vec::new(),
        _ => return Err(bad_param("input", "input must be a string or a list of items")),
    };
    let mut out: Vec<Value> = Vec::new();
    for it in items {
        let kind = it.get("type").and_then(Value::as_str).unwrap_or(if it.get("role").is_some() { "message" } else { "" });
        match kind {
            "message" => {
                let role = it.get("role").and_then(Value::as_str).unwrap_or("user");
                let role = if role == "developer" { "system" } else { role };
                out.push(json!({"role": role, "content": content_of(it.get("content").unwrap_or(&Value::Null))?}));
            }
            "function_call" => {
                let call = json!({"id": it.get("call_id").or(it.get("id")).cloned().unwrap_or(json!(new_id("call"))), "type": "function",
                                  "function": {"name": it.get("name").cloned().unwrap_or(json!("")),
                                               "arguments": it.get("arguments").cloned().unwrap_or(json!("{}"))}});
                // calls in a row are one assistant turn
                match out.last_mut() {
                    Some(m) if m["role"] == "assistant" && m.get("tool_calls").is_some_and(Value::is_array) => {
                        m["tool_calls"].as_array_mut().unwrap().push(call);
                    }
                    _ => out.push(json!({"role": "assistant", "content": null, "tool_calls": [call]})),
                }
            }
            "function_call_output" => {
                let output = match it.get("output") {
                    Some(Value::String(s)) => json!(s),
                    Some(v @ Value::Array(_)) => content_of(v)?,
                    Some(v) => json!(v.to_string()),
                    None => json!(""),
                };
                out.push(json!({"role": "tool", "tool_call_id": it.get("call_id").cloned().unwrap_or(json!("")), "content": output}));
            }
            "reasoning" | "item_reference" => {}
            other => return Err(bad_param("input", format!("input item '{other}' is not supported"))),
        }
    }
    Ok(out)
}

/// The chat request of a Responses request, and its conversation before the answer (kept for previous_response_id).
fn chat_body(body: &Value) -> Checked<(Value, Vec<Value>)> {
    let mut messages = Vec::new();
    if let Some(prev) = body.get("previous_response_id").and_then(Value::as_str) {
        let kept = store().lock().ok().and_then(|s| s.iter().find(|k| k.id == prev).map(|k| k.messages.clone()));
        messages = kept.ok_or_else(|| bad_param("previous_response_id", format!("no response with id '{prev}' is kept")))?;
    }
    if let Some(i) = body.get("instructions").and_then(Value::as_str) {
        // instructions are this request's: they replace a kept system message
        messages.retain(|m| m["role"] != "system");
        messages.insert(0, json!({"role": "system", "content": i}));
    }
    messages.extend(messages_of(body.get("input").unwrap_or(&Value::Null))?);
    if messages.is_empty() {
        return Err(bad_param("input", "input is empty"));
    }
    let mut chat = json!({"messages": messages.clone()});
    let o = chat.as_object_mut().unwrap();
    for (from, to) in [("model", "model"), ("temperature", "temperature"), ("top_p", "top_p"), ("max_output_tokens", "max_tokens"),
                       ("parallel_tool_calls", "parallel_tool_calls"), ("top_logprobs", "top_logprobs"), ("seed", "seed")] {
        if let Some(v) = body.get(from).filter(|v| !v.is_null()) {
            o.insert(to.into(), v.clone());
        }
    }
    if let Some(tools) = body.get("tools").and_then(Value::as_array).filter(|t| !t.is_empty()) {
        let mut out = Vec::new();
        for t in tools {
            match t.get("type").and_then(Value::as_str) {
                Some("function") => out.push(json!({"type": "function", "function": {
                    "name": t.get("name").cloned().unwrap_or(json!("")),
                    "description": t.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": t.get("parameters").cloned().unwrap_or(json!({"type": "object", "properties": {}}))}})),
                other => return Err(bad_param("tools", format!("tool type {other:?} is not supported (function tools only)"))),
            }
        }
        o.insert("tools".into(), json!(out));
    }
    match body.get("tool_choice") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => {
            o.insert("tool_choice".into(), json!(s));
        }
        Some(c) if c.get("type").and_then(Value::as_str) == Some("function") => {
            o.insert("tool_choice".into(), json!({"type": "function", "function": {"name": c.get("name").cloned().unwrap_or(json!(""))}}));
        }
        Some(c) => return Err(bad_param("tool_choice", format!("tool_choice {c} is not supported"))),
    }
    if let Some(effort) = body.pointer("/reasoning/effort").and_then(Value::as_str) {
        o.insert("reasoning_effort".into(), json!(effort));
    }
    match body.pointer("/text/format") {
        None | Some(Value::Null) => {}
        Some(f) => match f.get("type").and_then(Value::as_str) {
            Some("text") => {}
            Some("json_object") => {
                o.insert("response_format".into(), json!({"type": "json_object"}));
            }
            Some("json_schema") => {
                o.insert("response_format".into(), json!({"type": "json_schema", "json_schema": {
                    "name": f.get("name").cloned().unwrap_or(json!("schema")), "schema": f.get("schema").cloned().unwrap_or(json!({})),
                    "strict": f.get("strict").cloned().unwrap_or(json!(false))}}));
            }
            other => return Err(bad_param("text.format", format!("format type {other:?} is not supported"))),
        },
    }
    if body.get("stream").and_then(Value::as_bool) == Some(true) {
        o.insert("stream".into(), json!(true));
        o.insert("stream_options".into(), json!({"include_usage": true}));
    }
    Ok((chat, messages))
}

/// The response object's fixed fields (a copy of the request's settings).
fn shell(body: &Value, id: &str, model: &Value, status: &str) -> Value {
    json!({
        "id": id, "object": "response", "created_at": now(), "status": status, "model": model, "output": [],
        "error": null, "incomplete_details": null,
        "instructions": body.get("instructions").cloned().unwrap_or(Value::Null),
        "max_output_tokens": body.get("max_output_tokens").cloned().unwrap_or(Value::Null),
        "parallel_tool_calls": body.get("parallel_tool_calls").cloned().unwrap_or(json!(true)),
        "previous_response_id": body.get("previous_response_id").cloned().unwrap_or(Value::Null),
        "reasoning": body.get("reasoning").cloned().unwrap_or(json!({"effort": null, "summary": null})),
        "temperature": body.get("temperature").cloned().unwrap_or(json!(1.0)),
        "text": body.get("text").cloned().unwrap_or(json!({"format": {"type": "text"}})),
        "tool_choice": body.get("tool_choice").cloned().unwrap_or(json!("auto")),
        "tools": body.get("tools").cloned().unwrap_or(json!([])),
        "top_p": body.get("top_p").cloned().unwrap_or(json!(1.0)),
        "store": body.get("store").cloned().unwrap_or(json!(true)),
        "metadata": body.get("metadata").cloned().unwrap_or(json!({})),
        "usage": null,
    })
}

fn usage_of(u: &Value) -> Value {
    let input = u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
    let output = u.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0);
    json!({"input_tokens": input, "output_tokens": output, "total_tokens": input + output,
           "input_tokens_details": {"cached_tokens": u.pointer("/prompt_tokens_details/cached_tokens").and_then(Value::as_u64).unwrap_or(0)},
           "output_tokens_details": {"reasoning_tokens": u.pointer("/completion_tokens_details/reasoning_tokens").and_then(Value::as_u64).unwrap_or(0)}})
}

/// Output items of a chat message: reasoning, the message, the function calls.
fn items_of(m: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(r) = m.get("reasoning_content").and_then(Value::as_str).filter(|r| !r.is_empty()) {
        out.push(json!({"type": "reasoning", "id": new_id("rs"), "summary": [], "content": [{"type": "reasoning_text", "text": r}]}));
    }
    if let Some(t) = m.get("content").and_then(Value::as_str).filter(|t| !t.trim().is_empty()) {
        out.push(json!({"type": "message", "id": new_id("msg"), "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": t, "annotations": []}]}));
    }
    for c in m.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
        out.push(json!({"type": "function_call", "id": new_id("fc"), "call_id": c.get("id").cloned().unwrap_or(json!(new_id("call"))),
                        "name": c.pointer("/function/name").cloned().unwrap_or(json!("")),
                        "arguments": c.pointer("/function/arguments").cloned().unwrap_or(json!("{}")), "status": "completed"}));
    }
    out
}

/// The conversation after the answer, for a later previous_response_id.
fn keep(id: &str, mut messages: Vec<Value>, answer: &Value, response: &Value) {
    let mut m = json!({"role": "assistant", "content": answer.get("content").cloned().unwrap_or(Value::Null)});
    if let Some(tc) = answer.get("tool_calls").filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
        m["tool_calls"] = tc.clone();
    }
    if let Some(r) = answer.get("reasoning_content").filter(|r| !r.is_null()) {
        m["reasoning_content"] = r.clone();
    }
    messages.push(m);
    if let Ok(mut s) = store().lock() {
        s.push_back(Kept { id: id.to_owned(), messages, response: response.clone() });
        while s.len() > KEPT {
            s.pop_front();
        }
    }
}

async fn run(st: Arc<AppState>, body: Value) -> Checked<Response> {
    let (chat, messages) = chat_body(&body)?;
    let id = new_id("resp");
    let model = json!(st.cfg.served_name);
    let stream = chat.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let store_it = body.get("store").and_then(Value::as_bool).unwrap_or(true);
    match call(st, chat).await {
        ChatOut::Json(status, v) if !status.is_success() => Ok((status, Json(v)).into_response()),
        ChatOut::Json(_, v) => {
            let choice = &v["choices"][0];
            let mut r = shell(&body, &id, &model, "completed");
            r["output"] = json!(items_of(&choice["message"]));
            r["usage"] = usage_of(&v["usage"]);
            if choice["finish_reason"] == "length" {
                r["status"] = json!("incomplete");
                r["incomplete_details"] = json!({"reason": "max_output_tokens"});
            }
            if store_it {
                keep(&id, messages, &choice["message"], &r);
            }
            Ok(Json(r).into_response())
        }
        ChatOut::Stream(chunks) if stream => Ok(sse(stream_events(body, id, model, messages, store_it, chunks))),
        ChatOut::Stream(_) => Err(error(StatusCode::INTERNAL_SERVER_ERROR, "a stream for a non-streaming request")),
    }
}

fn sse(s: impl Stream<Item = Value> + Send + 'static) -> Response {
    let events = s.map(|v| {
        let kind = v.get("type").and_then(Value::as_str).unwrap_or("message").to_owned();
        Ok::<_, Infallible>(Event::default().event(kind).data(v.to_string()))
    });
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

/// Sequence numbers of a stream's events.
struct Seq(u64);

impl Seq {
    fn ev(&mut self, kind: &str, mut v: Value) -> Value {
        v["type"] = json!(kind);
        v["sequence_number"] = json!(self.0);
        self.0 += 1;
        v
    }
}

/// The events that close the open item (its final text or arguments).
fn close(open: &mut Option<(&'static str, usize, String)>, output: &mut [Value], ev: &mut Seq) -> Vec<Value> {
    let Some((kind, oi, text)) = open.take() else { return Vec::new() };
    let mut out = Vec::new();
    let item = &mut output[oi];
    let item_id = item["id"].clone();
    match kind {
        "reasoning" => {
            item["content"] = json!([{"type": "reasoning_text", "text": text}]);
            out.push(ev.ev("response.reasoning_text.done", json!({"item_id": item_id, "output_index": oi, "content_index": 0, "text": text})));
        }
        "message" => {
            let part = json!({"type": "output_text", "text": text, "annotations": []});
            item["content"] = json!([part.clone()]);
            item["status"] = json!("completed");
            out.push(ev.ev("response.output_text.done", json!({"item_id": item_id, "output_index": oi, "content_index": 0, "text": text})));
            out.push(ev.ev("response.content_part.done", json!({"item_id": item_id, "output_index": oi, "content_index": 0, "part": part})));
        }
        _ => {
            item["arguments"] = json!(text);
            item["status"] = json!("completed");
            out.push(ev.ev("response.function_call_arguments.done", json!({"item_id": item_id, "output_index": oi, "arguments": text})));
        }
    }
    out.push(ev.ev("response.output_item.done", json!({"output_index": oi, "item": output[oi].clone()})));
    out
}

/// The Responses stream of a chat chunk stream: one item per kind of output, opened as its first delta comes and
/// closed when another kind starts or at the end.
fn stream_events(body: Value, id: String, model: Value, messages: Vec<Value>, store_it: bool,
                 mut chunks: Pin<Box<dyn Stream<Item = Value> + Send>>) -> impl Stream<Item = Value> + Send {
    async_stream::stream! {
        let mut ev = Seq(0);
        let mut resp = shell(&body, &id, &model, "in_progress");
        yield ev.ev("response.created", json!({"response": resp.clone()}));
        yield ev.ev("response.in_progress", json!({"response": resp.clone()}));
        let mut output: Vec<Value> = Vec::new();
        // the open item: (kind, its output index, its text or arguments so far)
        let mut open: Option<(&'static str, usize, String)> = None;
        // tool call index -> output index
        let mut calls: std::collections::HashMap<u64, usize> = Default::default();
        let (mut reasoning, mut content, mut tool_calls) = (String::new(), String::new(), Vec::<Value>::new());
        let (mut usage, mut finish) = (Value::Null, Value::Null);
        while let Some(c) = chunks.next().await {
            if let Some(e) = c.get("error") {
                resp["status"] = json!("failed");
                resp["error"] = json!({"code": "server_error", "message": e.get("message").cloned().unwrap_or(json!("error"))});
                yield ev.ev("response.failed", json!({"response": resp.clone()}));
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
                if open.as_ref().is_none_or(|o| o.0 != "reasoning") {
                    for e in close(&mut open, &mut output, &mut ev) { yield e; }
                    let item = json!({"type": "reasoning", "id": new_id("rs"), "summary": [], "content": []});
                    output.push(item.clone());
                    open = Some(("reasoning", output.len() - 1, String::new()));
                    yield ev.ev("response.output_item.added", json!({"output_index": output.len() - 1, "item": item}));
                }
                let (_, oi, text) = open.as_mut().unwrap();
                text.push_str(r);
                reasoning.push_str(r);
                let oi = *oi;
                yield ev.ev("response.reasoning_text.delta", json!({"item_id": output[oi]["id"], "output_index": oi, "content_index": 0, "delta": r}));
            }
            if let Some(t) = d.get("content").and_then(Value::as_str).filter(|t| !t.is_empty()) {
                // text before the first visible character opens no message (the blank line after the reasoning)
                if open.as_ref().is_none_or(|o| o.0 != "message") && !t.trim().is_empty() {
                    for e in close(&mut open, &mut output, &mut ev) { yield e; }
                    let item = json!({"type": "message", "id": new_id("msg"), "status": "in_progress", "role": "assistant", "content": []});
                    output.push(item.clone());
                    let oi = output.len() - 1;
                    open = Some(("message", oi, String::new()));
                    yield ev.ev("response.output_item.added", json!({"output_index": oi, "item": item}));
                    yield ev.ev("response.content_part.added", json!({"item_id": output[oi]["id"], "output_index": oi, "content_index": 0,
                                                                   "part": {"type": "output_text", "text": "", "annotations": []}}));
                }
                content.push_str(t);
                if let Some((_, oi, text)) = open.as_mut().filter(|o| o.0 == "message") {
                    text.push_str(t);
                    let oi = *oi;
                    yield ev.ev("response.output_text.delta", json!({"item_id": output[oi]["id"], "output_index": oi, "content_index": 0, "delta": t}));
                }
            }
            for tc in d.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                let ti = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
                if !calls.contains_key(&ti) {
                    for e in close(&mut open, &mut output, &mut ev) { yield e; }
                    let call_id = tc.get("id").cloned().filter(|v| !v.is_null()).unwrap_or(json!(new_id("call")));
                    let name = tc.pointer("/function/name").cloned().unwrap_or(json!(""));
                    let item = json!({"type": "function_call", "id": new_id("fc"), "call_id": call_id, "name": name, "arguments": "", "status": "in_progress"});
                    output.push(item.clone());
                    calls.insert(ti, output.len() - 1);
                    tool_calls.push(json!({"id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}));
                    open = Some(("function_call", output.len() - 1, String::new()));
                    yield ev.ev("response.output_item.added", json!({"output_index": output.len() - 1, "item": item}));
                }
                if let Some(a) = tc.pointer("/function/arguments").and_then(Value::as_str).filter(|a| !a.is_empty()) {
                    let oi = calls[&ti];
                    if let Some((_, o, text)) = open.as_mut().filter(|o| o.1 == oi) {
                        text.push_str(a);
                        let _ = o;
                    }
                    if let Some(t) = tool_calls.get_mut(ti as usize) {
                        let prev = t["function"]["arguments"].as_str().unwrap_or("").to_owned();
                        t["function"]["arguments"] = json!(prev + a);
                    }
                    yield ev.ev("response.function_call_arguments.delta", json!({"item_id": output[oi]["id"], "output_index": oi, "delta": a}));
                }
            }
        }
        for e in close(&mut open, &mut output, &mut ev) { yield e; }
        resp["status"] = json!(if finish == "length" { "incomplete" } else { "completed" });
        if finish == "length" {
            resp["incomplete_details"] = json!({"reason": "max_output_tokens"});
        }
        resp["output"] = json!(output);
        resp["usage"] = usage_of(&usage);
        if store_it {
            let answer = json!({"content": content, "reasoning_content": if reasoning.is_empty() { Value::Null } else { json!(reasoning) },
                                "tool_calls": tool_calls});
            keep(&id, messages, &answer, &resp);
        }
        let kind = if finish == "length" { "response.incomplete" } else { "response.completed" };
        yield ev.ev(kind, json!({"response": resp}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_items_become_chat_messages() {
        let body = json!({"instructions": "Be brief.", "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Weather?"}]},
            {"type": "function_call", "call_id": "c1", "name": "w", "arguments": "{\"city\":\"Oslo\"}"},
            {"type": "function_call", "call_id": "c2", "name": "w", "arguments": "{\"city\":\"Rome\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            {"type": "reasoning", "id": "rs_1", "summary": []},
        ], "tools": [{"type": "function", "name": "w", "parameters": {"type": "object"}}], "tool_choice": {"type": "function", "name": "w"},
           "max_output_tokens": 50, "text": {"format": {"type": "json_object"}}, "reasoning": {"effort": "low"}});
        let (chat, _) = chat_body(&body).unwrap();
        let m = chat["messages"].as_array().unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "Be brief."}));
        assert_eq!(m[1]["content"], json!([{"type": "text", "text": "Weather?"}]));
        assert_eq!(m[2]["tool_calls"].as_array().unwrap().len(), 2);
        assert_eq!(m[3], json!({"role": "tool", "tool_call_id": "c1", "content": "sunny"}));
        assert_eq!(m.len(), 4);
        assert_eq!(chat["tools"][0]["function"]["name"], "w");
        assert_eq!(chat["tool_choice"], json!({"type": "function", "function": {"name": "w"}}));
        assert_eq!(chat["max_tokens"], 50);
        assert_eq!(chat["response_format"], json!({"type": "json_object"}));
        assert_eq!(chat["reasoning_effort"], "low");
    }

    #[test]
    fn a_chat_answer_becomes_output_items() {
        let items = items_of(&json!({"reasoning_content": "think", "content": "Hi",
                                     "tool_calls": [{"id": "c", "function": {"name": "w", "arguments": "{}"}}]}));
        let kinds: Vec<&str> = items.iter().map(|i| i["type"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["reasoning", "message", "function_call"]);
        assert_eq!(items[1]["content"][0]["text"], "Hi");
        assert_eq!(items[2]["call_id"], "c");
    }
}
