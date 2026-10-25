//! Embeddings use the final token hidden state, L2-normalized.
//! Reduced dimensions are normalized again; base64 output contains little-endian f32 values.

use super::*;
use base64::Engine as _;

/// Most inputs of one request (OpenAI's own limit is 2048).
const MAX_INPUTS: usize = 256;

pub(super) async fn embeddings(State(st): State<Arc<AppState>>, bytes: Bytes) -> Response {
    let body = match parse_body(&bytes) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match run(&st, &body).await {
        Ok(r) | Err(r) => r,
    }
}

/// The inputs as token lists: a string, a list of strings, a list of token ids, a list of such lists.
fn inputs(st: &AppState, body: &Value) -> Checked<Vec<Vec<i64>>> {
    let bad = || bad_param("input", "input must be a string, a list of strings, a list of token ids or a list of such lists");
    let encode = |s: &str| -> Checked<Vec<i64>> {
        Ok(st.text.encode(s).map_err(|e| error(StatusCode::BAD_REQUEST, e.to_string()))?.into_iter().map(i64::from).collect())
    };
    let ids = |a: &[Value]| -> Checked<Vec<i64>> {
        a.iter()
            .map(|v| v.as_i64().filter(|&t| t >= 0 && (t as usize) < st.vocab).ok_or_else(|| bad_param("input", "token ids must be in the vocabulary")))
            .collect()
    };
    let out = match body.get("input") {
        Some(Value::String(s)) => vec![encode(s)?],
        Some(Value::Array(a)) if a.is_empty() => return Err(bad()),
        Some(Value::Array(a)) if a.iter().all(Value::is_string) => a.iter().map(|v| encode(v.as_str().unwrap())).collect::<Checked<_>>()?,
        Some(Value::Array(a)) if a.iter().all(Value::is_number) => vec![ids(a)?],
        Some(Value::Array(a)) if a.iter().all(Value::is_array) => a.iter().map(|v| ids(v.as_array().unwrap())).collect::<Checked<_>>()?,
        _ => return Err(bad()),
    };
    if out.len() > MAX_INPUTS {
        return Err(bad_param("input", format!("at most {MAX_INPUTS} inputs in one request")));
    }
    if out.iter().any(Vec::is_empty) {
        return Err(bad_param("input", "an input is empty"));
    }
    Ok(out)
}

async fn run(st: &Arc<AppState>, body: &Value) -> Checked<Response> {
    let mut all = inputs(st, body)?;
    let dims = match body.get("dimensions").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_u64().filter(|&d| d >= 1).ok_or_else(|| bad_param("dimensions", "dimensions must be a positive integer"))? as usize),
    };
    let base64_out = match body.get("encoding_format").and_then(Value::as_str) {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => return Err(bad_param("encoding_format", format!("encoding_format '{other}': float or base64"))),
    };
    for tokens in &mut all {
        truncate_prompt(st, body, tokens);
        check_fits(st, tokens.len())?;
    }
    // every input at once: the scheduler batches their prompts
    let mut rxs = Vec::with_capacity(all.len());
    for tokens in &all {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        st.sched
            .submit(GenRequest {
                tokens: tokens.clone(),
                params: SamplingParams { temperature: 0.0, ..Default::default() },
                max_tokens: 1,
                stop_ids: Vec::new(),
                cache_points: Vec::new(),
                think: None,
                constraint: None,
                mm: None,
                events: tx,
                cancel: Arc::new(AtomicBool::new(false)),
                embed: true,
            })
            .map_err(|_| error(StatusCode::SERVICE_UNAVAILABLE, "the scheduler has stopped"))?;
        rxs.push(rx);
    }
    st.metrics.requests.fetch_add(all.len() as u64, Ordering::Relaxed);
    let mut data = Vec::with_capacity(all.len());
    for (index, mut rx) in rxs.into_iter().enumerate() {
        let mut v = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                GenEvent::Embedding(e) => v = Some(e),
                GenEvent::Error(e) => return Err(error(StatusCode::INTERNAL_SERVER_ERROR, e)),
                GenEvent::Done(_) => break,
                _ => {}
            }
        }
        let mut v = v.ok_or_else(|| error(StatusCode::INTERNAL_SERVER_ERROR, "the embedding did not come"))?;
        if let Some(d) = dims {
            if d > v.len() {
                return Err(bad_param("dimensions", format!("dimensions {d} above the model's {}", v.len())));
            }
            v.truncate(d);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter_mut().for_each(|x| *x /= norm);
        let embedding = if base64_out {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            json!(base64::engine::general_purpose::STANDARD.encode(bytes))
        } else {
            json!(v)
        };
        data.push(json!({"object": "embedding", "index": index, "embedding": embedding}));
    }
    let prompt: usize = all.iter().map(Vec::len).sum();
    Ok(Json(json!({
        "object": "list",
        "data": data,
        "model": st.cfg.served_name,
        "usage": {"prompt_tokens": prompt, "total_tokens": prompt},
    }))
    .into_response())
}
