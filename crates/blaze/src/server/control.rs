//! Authentication, bounded HTTP admission, request tracing and deadlines (stream bodies included).
use super::*;
use crate::logging::{self, Level, RequestTrace};
use axum::extract::ConnectInfo;

fn body_error_status(error: &axum::Error) -> StatusCode {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(error) = source {
        if error.is::<http_body_util::LengthLimitError>() {
            return StatusCode::PAYLOAD_TOO_LARGE;
        }
        source = error.source();
    }
    StatusCode::BAD_REQUEST
}

/// Each API retains its own wire protocol, including when a deadline closes a stream.
fn timeout_event(path: &str, trace: &RequestTrace) -> Bytes {
    let message = "request deadline exceeded";
    if path == "/v1/messages" {
        Bytes::from(format!("event: error\ndata: {}\n\n", json!({"type":"error","error":{"type":"api_error","message":message}})))
    } else if path == "/v1/responses" {
        let sequence = trace.stream_sequence.fetch_add(1, Ordering::Relaxed);
        Bytes::from(format!(
            "event: error\ndata: {}\n\n",
            json!({"type":"error","code":"request_timeout","message":message,"param":null,"sequence_number":sequence})
        ))
    } else {
        Bytes::from(format!("data: {}\n\ndata: [DONE]\n\n", stream_error(message)))
    }
}

pub(super) fn submit_error(err: crate::sched::SubmitError) -> Response {
    match err {
        crate::sched::SubmitError::Overloaded => {
            let mut r = api_error(StatusCode::TOO_MANY_REQUESTS, "generation queue is full; retry later", Some("server_overloaded"), None);
            r.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
            r
        }
        crate::sched::SubmitError::Stopped => error(StatusCode::SERVICE_UNAVAILABLE, "the scheduler has stopped"),
    }
}

fn authenticated(st: &AppState, req: &Request) -> bool {
    if st.key_hashes.is_empty() {
        return true;
    }
    let bearer = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, v)| v);
    let key = bearer.or_else(|| req.headers().get("x-api-key").and_then(|v| v.to_str().ok()));
    let Some(key) = key else { return false };
    let hash = blake3::hash(key.as_bytes());
    st.key_hashes.iter().fold(false, |found, k| (hash == *k) | found)
}

struct HttpGuard {
    st: Arc<AppState>,
    trace: RequestTrace,
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
    path: String,
    status: u16,
    bytes: u64,
    complete: bool,
    preview: Vec<u8>,
    deadline_task: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for HttpGuard {
    fn drop(&mut self) {
        self.trace.cancelled.store(true, Ordering::Relaxed);
        if let Some(task) = &self.deadline_task {
            task.abort();
        }
        self.st.metrics.http_active.fetch_sub(1, Ordering::Relaxed);
        self.st.metrics.http_latency.observe(self.trace.received.elapsed().as_secs_f64());
        if logging::options().requests {
            logging::event(
                if self.status >= 500 { Level::Error } else { Level::Info },
                "http.finished",
                json!({
                "request_id":self.trace.id,"path":self.path,"status":self.status,"elapsed_ms":self.trace.received.elapsed().as_secs_f64()*1000.,
                "response_bytes":self.bytes,"complete":self.complete}),
            );
        }
        if logging::options().outputs && !self.preview.is_empty() {
            logging::event(
                Level::Info,
                "http.output",
                json!({"request_id":self.trace.id,"payload":logging::preview(&String::from_utf8_lossy(&self.preview),logging::options().max_len)}),
            );
        }
    }
}

pub(super) async fn request(State(st): State<Arc<AppState>>, mut req: Request, next: Next) -> Response {
    let id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b)))
        .map(str::to_owned)
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    let trace = RequestTrace { id, ..RequestTrace::default() };
    let response_trace = trace.clone();
    let origin = req.headers().get(header::ORIGIN).cloned();
    let cors_origin = origin.filter(|v| st.cfg.allowed_origins.iter().any(|x| x == "*" || v.as_bytes() == x.as_bytes()));
    let preflight = req.method() == Method::OPTIONS;
    let ask = req.headers().get(header::ACCESS_CONTROL_REQUEST_HEADERS).cloned();
    let result = logging::REQUEST.scope(trace.clone(), async {
        let path = req.uri().path().to_string();
        if preflight { return StatusCode::NO_CONTENT.into_response(); }
        if path=="/health" || path=="/ping" { return next.run(req).await; }
        st.metrics.http_requests.fetch_add(1,Ordering::Relaxed);
        if !authenticated(&st,&req) {
            st.metrics.http_rejected.fetch_add(1,Ordering::Relaxed);
            let mut r = api_error(StatusCode::UNAUTHORIZED,"invalid or missing API key",Some("invalid_api_key"),None);
            r.headers_mut().insert(header::WWW_AUTHENTICATE,HeaderValue::from_static("Bearer"));
            return r;
        }
        // Operational endpoints remain observable while generation admission is saturated.
        let operational=req.method()==Method::GET && matches!(path.as_str(),"/metrics"|"/server_info"|"/get_server_info"|"/version"|"/v1/models");
        let permit = if operational {None}else{Some(match st.http_slots.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => { st.metrics.http_rejected.fetch_add(1,Ordering::Relaxed); return submit_error(crate::sched::SubmitError::Overloaded); }
        })};
        st.metrics.http_active.fetch_add(1,Ordering::Relaxed);
        let peer = req.extensions().get::<ConnectInfo<std::net::SocketAddr>>().map(|x|x.0.to_string());
        if logging::options().requests {
            logging::event(Level::Info,"http.received",json!({"request_id":trace.id,"peer":peer,"method":req.method().as_str(),"path":path,
                "user_agent":req.headers().get(header::USER_AGENT).and_then(|v|v.to_str().ok()),"content_length":req.headers().get(header::CONTENT_LENGTH).and_then(|v|v.to_str().ok())}));
        }
        let deadline = (!st.cfg.request_timeout.is_zero()).then(||tokio::time::Instant::now()+st.cfg.request_timeout);
        // This timer also runs while a slow client stops polling its response body.
        let deadline_task=deadline.map(|d| {
            let cancelled=trace.cancelled.clone();
            tokio::spawn(async move {tokio::time::sleep_until(d).await;cancelled.store(true,Ordering::Relaxed);})
        });
        // Before headers exist, dropping the handler means the client disconnected (the conventional
        // log-only 499), not an HTTP 500 generated by the server.
        let mut guard = HttpGuard {st:st.clone(),trace:trace.clone(),permit,path,status:499,bytes:0,complete:false,preview:Vec::new(),deadline_task};
        if logging::options().inputs {
            let (parts, body) = req.into_parts();
            let read = axum::body::to_bytes(body,st.cfg.body_limit);
            let read = match deadline {
                Some(d) => match tokio::time::timeout_at(d,read).await {
                    Ok(r)=>r,
                    Err(_)=>{guard.status=504;guard.complete=true;st.metrics.http_timeouts.fetch_add(1,Ordering::Relaxed);return error(StatusCode::GATEWAY_TIMEOUT,"request body deadline exceeded");}
                },
                None => read.await,
            };
            let bytes = match read {
                Ok(b) => b,
                Err(e) => { let status=body_error_status(&e); guard.status=status.as_u16(); guard.complete=true; return error(status,e.to_string()); }
            };
            logging::event(Level::Info,"http.input",json!({"request_id":trace.id,"body_bytes":bytes.len(),"payload":logging::preview(&String::from_utf8_lossy(&bytes),logging::options().max_len)}));
            req = Request::from_parts(parts,axum::body::Body::from(bytes));
        }
        let mut body_deadline = deadline;
        let response = match deadline {
            Some(d) => match tokio::time::timeout_at(d,next.run(req)).await {
                Ok(r) => r,
                Err(_) => {
                    // The generated error must remain readable after the original deadline. Applying
                    // that deadline again would discard the JSON body and count this timeout twice.
                    body_deadline = None;
                    st.metrics.http_timeouts.fetch_add(1,Ordering::Relaxed);
                    error(StatusCode::GATEWAY_TIMEOUT,"request deadline exceeded")
                }
            },
            None => next.run(req).await,
        };
        guard.status = response.status().as_u16();
        if response.status().is_server_error() {st.metrics.http_errors.fetch_add(1,Ordering::Relaxed);}
        let streaming = response.headers().get(header::CONTENT_TYPE).is_some_and(|v|v.as_bytes().starts_with(b"text/event-stream"));
        let (parts, body) = response.into_parts();
        let mut stream = body.into_data_stream();
        let wrapped = async_stream::stream! {
            loop {
                let frame = match body_deadline {
                    Some(d) => match tokio::time::timeout_at(d,stream.next()).await {
                        Ok(v)=>v,
                        Err(_)=>{
                            guard.st.metrics.http_timeouts.fetch_add(1,Ordering::Relaxed);
                            logging::event(Level::Warn,"http.timeout",json!({"request_id":guard.trace.id,"phase":"response_body"}));
                            if streaming {
                                let frame=timeout_event(&guard.path,&guard.trace);
                                guard.bytes+=frame.len() as u64;
                                yield Ok::<Bytes,axum::Error>(frame);
                            }
                            break;
                        }
                    },
                    None => stream.next().await,
                };
                let Some(frame) = frame else {guard.complete=true;break};
                if let Ok(ref bytes)=frame {
                    guard.bytes+=bytes.len() as u64;
                    if logging::options().outputs {
                        let remaining = logging::options().max_len.saturating_mul(4).saturating_sub(guard.preview.len());
                        guard.preview.extend_from_slice(&bytes[..bytes.len().min(remaining)]);
                    }
                }
                let failed=frame.is_err();
                yield frame;
                if failed {break;}
            }
            // Keep the permit alive until the client finishes polling or drops the body.
            guard.permit.take();
        };
        Response::from_parts(parts,axum::body::Body::from_stream(wrapped))
    }).await;
    let mut result = result;
    result.headers_mut().insert("x-request-id", HeaderValue::from_str(&response_trace.id).unwrap());
    if let Some(origin) = cors_origin {
        let h = result.headers_mut();
        h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin);
        h.insert(header::VARY, HeaderValue::from_static("Origin"));
        h.insert(header::ACCESS_CONTROL_EXPOSE_HEADERS, HeaderValue::from_static("x-request-id"));
        if preflight {
            h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, OPTIONS"));
            h.insert(
                header::ACCESS_CONTROL_ALLOW_HEADERS,
                ask.unwrap_or(HeaderValue::from_static("authorization, content-type, x-api-key, x-request-id")),
            );
        }
    }
    result
}

pub(super) async fn info(State(st): State<Arc<AppState>>) -> Json<Value> {
    let s = &st.sched.stats;
    Json(json!({"model":st.cfg.served_name,"version":env!("CARGO_PKG_VERSION"),"max_model_len":st.max_context,
        "sampling_candidate_limit":null,
        "sampling_defaults":st.cfg.defaults,"instruct_sampling_defaults":st.cfg.instruct_defaults,"config":st.cfg.effective,"api_key_count":st.key_hashes.len(),
        "scheduler":{"running":s.running.load(Ordering::Relaxed),"waiting":s.waiting.load(Ordering::Relaxed),
        "outstanding":s.outstanding.load(Ordering::Relaxed),"kv_pages_free":s.kv_pages_free.load(Ordering::Relaxed),
        "kv_pages_total":s.kv_pages_total.load(Ordering::Relaxed),"cache_entries":s.cache_entries.load(Ordering::Relaxed)}}))
}

pub(super) async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! { _=tokio::signal::ctrl_c()=>{}, _=term.recv()=>{} }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn body_limit_and_transport_errors_have_distinct_statuses() {
        let limited = axum::body::to_bytes(axum::body::Body::from("large"), 1).await.unwrap_err();
        assert_eq!(body_error_status(&limited), StatusCode::PAYLOAD_TOO_LARGE);
        let body =
            axum::body::Body::from_stream(futures::stream::once(async { Err::<Bytes, _>(std::io::Error::other("broken transport")) }));
        let broken = axum::body::to_bytes(body, 1024).await.unwrap_err();
        assert_eq!(body_error_status(&broken), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn deadline_events_follow_each_api_protocol() {
        let trace = RequestTrace::default();
        trace.stream_sequence.store(12, Ordering::Relaxed);
        let messages = String::from_utf8(timeout_event("/v1/messages", &trace).to_vec()).unwrap();
        assert!(messages.starts_with("event: error\ndata: "));
        assert!(!messages.contains("[DONE]"));
        let responses = String::from_utf8(timeout_event("/v1/responses", &trace).to_vec()).unwrap();
        let event: Value = serde_json::from_str(responses.lines().find_map(|l| l.strip_prefix("data: ")).unwrap()).unwrap();
        assert_eq!(event["sequence_number"], 12);
        assert_eq!(event["type"], "error");
        assert_eq!(event["code"], "request_timeout");
        assert!(!responses.contains("[DONE]"));
        let chat = String::from_utf8(timeout_event("/v1/chat/completions", &trace).to_vec()).unwrap();
        assert!(chat.ends_with("data: [DONE]\n\n"));
    }
}
