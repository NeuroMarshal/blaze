//! vLLM's /tokenize and /detokenize: token counts for clients that budget their context.

use super::*;

pub(super) async fn tokenize(State(st): State<Arc<AppState>>, bytes: Bytes) -> Response {
    let mut body = match parse_body(&bytes) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match run_tokenize(&st, &mut body) {
        Ok(r) | Err(r) => r,
    }
}

fn run_tokenize(st: &AppState, body: &mut Value) -> Checked<Response> {
    let tokens: Vec<i64> = if body.get("messages").is_some() {
        // the chat template's prompt, whatever its length (the point is to learn that it does not fit)
        super::chat::prepare(st, body, false)?.tokens
    } else {
        let Some(prompt) = body.get("prompt").and_then(Value::as_str) else {
            return Err(bad_param("prompt", "give a prompt (string) or messages"));
        };
        st.text.encode(prompt).map_err(|e| error(StatusCode::BAD_REQUEST, e.to_string()))?.into_iter().map(i64::from).collect()
    };
    let strs = body.get("return_token_strs").and_then(Value::as_bool).unwrap_or(false).then(|| {
        tokens.iter().map(|&t| st.text.decode(&[t as u32]).unwrap_or_default()).collect::<Vec<_>>()
    });
    Ok(Json(json!({"count": tokens.len(), "max_model_len": st.max_context, "tokens": tokens, "token_strs": strs})).into_response())
}

pub(super) async fn detokenize(State(st): State<Arc<AppState>>, bytes: Bytes) -> Response {
    let body = match parse_body(&bytes) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let Some(ids) = body.get("tokens").and_then(Value::as_array).and_then(|a| a.iter().map(Value::as_u64).collect::<Option<Vec<_>>>()) else {
        return bad_param("tokens", "tokens must be a list of token ids");
    };
    if let Some(bad) = ids.iter().find(|&&t| t as usize >= st.vocab) {
        return bad_param("tokens", format!("token id {bad} is outside the vocabulary of {}", st.vocab));
    }
    let ids: Vec<u32> = ids.into_iter().map(|t| t as u32).collect();
    match st.text.decode(&ids) {
        Ok(prompt) => Json(json!({"prompt": prompt})).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, e.to_string()),
    }
}
