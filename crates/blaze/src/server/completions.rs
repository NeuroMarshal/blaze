//! /v1/completions: raw prompts (text, token ids, or lists of them), n choices, echo, logprobs, streaming.

use super::*;
use std::pin::Pin;

enum Prompt {
    Text(String),
    Ids(Vec<u32>),
}

/// The `prompt` field: a string, a list of strings, a list of token ids, or a list of such lists.
fn prompts_of(body: &Value) -> Checked<Vec<Prompt>> {
    let err = || bad_param("prompt", "prompt must be a string, a list of strings, a list of token ids or a list of those");
    let ids = |v: &Value| -> Option<Vec<u32>> { v.as_array()?.iter().map(|x| x.as_u64().and_then(|x| u32::try_from(x).ok())).collect() };
    match body.get("prompt") {
        Some(Value::String(s)) => Ok(vec![Prompt::Text(s.clone())]),
        Some(Value::Array(a)) if !a.is_empty() => {
            if let Some(p) = ids(&Value::Array(a.clone())) {
                return Ok(vec![Prompt::Ids(p)]);
            }
            a.iter()
                .map(|p| match p {
                    Value::String(s) => Ok(Prompt::Text(s.clone())),
                    v => ids(v).map(Prompt::Ids).ok_or_else(err),
                })
                .collect()
        }
        _ => Err(err()),
    }
}

/// OpenAI completions logprobs of generated tokens; `offset`: characters of text before them (advanced).
fn completion_logprobs(text: &Text, items: &[(u32, TokenLogprob)], offset: &mut usize) -> Value {
    if items.is_empty() {
        return Value::Null;
    }
    let tok = |t: u32| text.token_str(t).0;
    let strs: Vec<String> = items.iter().map(|(t, _)| tok(*t)).collect();
    let mut offsets = Vec::with_capacity(strs.len());
    for s in &strs {
        offsets.push(*offset);
        *offset += s.chars().count();
    }
    json!({
        "tokens": strs,
        "token_logprobs": items.iter().map(|(_, l)| l.logprob).collect::<Vec<_>>(),
        "top_logprobs": items.iter().map(|(_, l)| {
            l.top.iter().map(|&(t, lp)| (tok(t as u32), json!(lp))).collect::<serde_json::Map<_, _>>()
        }).collect::<Vec<_>>(),
        "text_offset": offsets,
    })
}

/// A prompt after tokenization.
struct Job {
    tokens: Vec<i64>,
    /// the prompt as text, for echo
    text: String,
}

/// One choice's generation as chunks of a streaming response.
fn chunks(mut g: Gen, index: usize, echo: Option<String>, totals: Arc<Totals>, st: Arc<AppState>, id: String, model: String,
          created: u64) -> Pin<Box<dyn Stream<Item = String> + Send>> {
    Box::pin(async_stream::stream! {
        let chunk = |text: &str, logprobs: Value, finish: Option<&str>| {
            json!({"id": id, "object": "text_completion", "created": created, "model": model,
                   "choices": [{"index": index, "text": text, "logprobs": logprobs, "finish_reason": finish}]}).to_string()
        };
        let mut offset = 0;
        let mut seen = 0;
        if let Some(e) = &echo { yield chunk(e, Value::Null, None); }
        while let Some(piece) = g.next().await {
            totals.note(&g, &mut seen);
            match piece {
                Piece::Text(t, lps) => yield chunk(&t, completion_logprobs(&st.text, &lps, &mut offset), None),
                Piece::Finish(f) => { yield chunk("", Value::Null, Some(finish_str(f))); break; }
                Piece::Error(e) => { yield stream_error(&e); break; }
            }
        }
    })
}

fn finish_str(f: Finish) -> &'static str {
    match f {
        Finish::Length => "length",
        _ => "stop",
    }
}

pub(super) async fn completions(State(st): State<Arc<AppState>>, bytes: Bytes) -> Response {
    let body = match parse_body(&bytes) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match run(st, body).await {
        Ok(r) | Err(r) => r,
    }
}

async fn run(st: Arc<AppState>, body: Value) -> Checked<Response> {
    check_ranges(&st, &body)?;
    if body.get("suffix").is_some_and(|v| !v.is_null()) {
        return Err(bad_param("suffix", "suffix is not supported"));
    }
    let n = n_of(&body);
    if body.get("best_of").and_then(Value::as_u64).is_some_and(|b| b as usize > n) {
        return Err(bad_param("best_of", "best_of larger than n is not supported"));
    }
    let echo = body.get("echo").and_then(Value::as_bool).unwrap_or(false);
    if echo && body.get("logprobs").is_some_and(|v| !v.is_null()) {
        return Err(bad_param("echo", "logprobs of the prompt are not supported: echo and logprobs cannot be combined"));
    }
    let prompts = prompts_of(&body)?;
    if prompts.len() * n > super::MAX_GENERATIONS {
        return Err(bad_param("prompt", format!("{} prompts x n = {n} generations: at most {} in one request", prompts.len(), super::MAX_GENERATIONS)));
    }
    let mut jobs = Vec::new();
    for p in prompts {
        let (mut tokens, text): (Vec<i64>, String) = match p {
            Prompt::Text(s) => (st.text.encode(&s).map_err(|e| error(StatusCode::BAD_REQUEST, e.to_string()))?.into_iter().map(i64::from).collect(), s),
            Prompt::Ids(ids) => {
                if let Some(bad) = ids.iter().find(|&&t| t as usize >= st.vocab) {
                    return Err(bad_param("prompt", format!("token id {bad} is outside the vocabulary of {}", st.vocab)));
                }
                (ids.iter().map(|&t| t as i64).collect(), if echo { st.text.decode(&ids).unwrap_or_default() } else { String::new() })
            }
        };
        truncate_prompt(&st, &body, &mut tokens);
        check_fits(&st, tokens.len())?;
        jobs.push(Job { tokens, text });
    }
    let max_tokens = ["max_tokens", "max_completion_tokens"].iter().find_map(|k| body.get(*k).and_then(Value::as_u64)).map(|x| x as usize).unwrap_or(16);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(Value::as_bool).unwrap_or(false);
    let spec = Spec::of(&st, &body, max_tokens, None);
    let mut gens = Vec::new();
    for (pi, job) in jobs.iter().enumerate() {
        gens.extend(submit_n(&st, &job.tokens, &spec, n, pi * n, &|| constraint_of(&st, &body, None, None))?);
    }
    let totals = Totals::new(jobs.iter().map(|j| j.tokens.len()).sum());
    let id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let model = st.cfg.served_name.clone();
    let created = now();
    let echoed = |index: usize| echo.then(|| jobs[index / n].text.clone());
    if stream {
        let parts: Vec<_> = gens
            .into_iter()
            .enumerate()
            .map(|(i, g)| chunks(g, i, echoed(i), totals.clone(), st.clone(), id.clone(), model.clone(), created))
            .collect();
        let mut merged = futures::stream::select_all(parts);
        let s = async_stream::stream! {
            while let Some(c) = merged.next().await { yield c; }
            if include_usage {
                yield json!({"id": id, "object": "text_completion", "created": created, "model": model, "choices": [],
                             "usage": totals.usage()}).to_string();
            }
            yield "[DONE]".to_string();
        };
        return Ok(sse(s));
    }
    let mut choices = Vec::with_capacity(gens.len());
    for (i, mut g) in gens.into_iter().enumerate() {
        let (mut text, mut logprobs, mut finish) = (echoed(i).unwrap_or_default(), Vec::new(), Finish::Stop);
        let mut seen = 0;
        while let Some(p) = g.next().await {
            totals.note(&g, &mut seen);
            match p {
                Piece::Text(t, lps) => {
                    text.push_str(&t);
                    logprobs.extend(lps);
                }
                Piece::Finish(f) => {
                    finish = f;
                    break;
                }
                Piece::Error(e) => return Err(error(StatusCode::INTERNAL_SERVER_ERROR, e)),
            }
        }
        choices.push(json!({"index": i, "text": text, "logprobs": completion_logprobs(&st.text, &logprobs, &mut 0),
                            "finish_reason": finish_str(finish)}));
    }
    Ok(Json(json!({"id": id, "object": "text_completion", "created": created, "model": model, "choices": choices,
                   "usage": totals.usage()}))
    .into_response())
}
