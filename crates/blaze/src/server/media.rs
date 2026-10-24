//! Chat image fetching, size/domain limits, processor settings and encoder submission.

use super::*;
use crate::mm::{ImageEmbeds, MmInput, hash_bytes};
use crate::vision::{ImageSettings, Overrides, RgbImage, Vision, decode_image, preprocess};
use base64::Engine as _;
use std::io::Read;
use std::time::Duration;

/// Server-side image settings (command line).
pub struct MmConfig {
    pub vision: Option<Arc<Vision>>,
    /// `--mm-processor-kwargs`: defaults over the checkpoint's processor config, under the request's
    pub kwargs: Overrides,
    /// `--limit-mm-per-prompt` image: most images in one request (vLLM's default: 999)
    pub limit_images: usize,
    /// `image_url.detail` "low": the image's max_pixels is capped to this (0: `detail` is ignored, as vLLM does)
    pub low_detail_pixels: u64,
    /// http(s) image URLs: total time (vLLM's VLLM_IMAGE_FETCH_TIMEOUT: 5 s) and size limit of one image
    pub fetch_timeout: Duration,
    pub fetch_max_bytes: usize,
    /// vLLM's --allowed-media-domains: hosts http(s) image URLs may name (empty: any)
    pub allowed_domains: Vec<String>,
    /// embeddings of recent images (one entry per image content and settings), so that a conversation that sends its
    /// image again does not run the encoder again
    pub cache: std::sync::Mutex<EmbedCache>,
}

/// Most recently used embeddings within a byte budget.
#[derive(Default)]
pub struct EmbedCache {
    entries: Vec<(u64, Arc<Vec<u16>>)>,
    bytes: usize,
}

const CACHE_BYTES: usize = 256 << 20;

impl EmbedCache {
    fn get(&mut self, key: u64) -> Option<Arc<Vec<u16>>> {
        let i = self.entries.iter().position(|e| e.0 == key)?;
        let e = self.entries.remove(i);
        let v = e.1.clone();
        self.entries.push(e);
        Some(v)
    }

    fn put(&mut self, key: u64, v: Arc<Vec<u16>>) {
        self.bytes += v.len() * 2;
        self.entries.push((key, v));
        while self.bytes > CACHE_BYTES && self.entries.len() > 1 {
            self.bytes -= self.entries.remove(0).1.len() * 2;
        }
    }
}

impl MmConfig {
    pub fn new(vision: Option<Arc<Vision>>, kwargs: Overrides, limit_images: usize, low_detail_pixels: u64, fetch_timeout: Duration,
               fetch_max_bytes: usize, allowed_domains: Vec<String>) -> Self {
        Self { vision, kwargs, limit_images, low_detail_pixels, fetch_timeout, fetch_max_bytes, allowed_domains, cache: Default::default() }
    }
}

/// One image of the request: its bytes, the settings it is processed with and the size it will have.
pub(super) struct Part {
    bytes: Vec<u8>,
    settings: ImageSettings,
    /// merged grid (rows, columns of the image's tokens)
    pub grid: (usize, usize),
}

impl Part {
    pub fn tokens(&self) -> usize {
        self.grid.0 * self.grid.1
    }
}

/// `--limit-mm-per-prompt`: JSON ({"image": 4} or {"image": {"count": 4}}) or `image=4` pairs; 999 for what it leaves out.
pub fn parse_limit(s: &str) -> std::result::Result<usize, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(999);
    }
    let count = |v: &Value| v.as_u64().or_else(|| v.get("count").and_then(Value::as_u64)).map(|n| n as usize);
    if s.starts_with('{') {
        let v: Value = serde_json::from_str(s).map_err(|e| format!("--limit-mm-per-prompt: {e}"))?;
        return match v.get("image") {
            None => Ok(999),
            Some(x) => count(x).ok_or_else(|| "--limit-mm-per-prompt: image needs a count".to_string()),
        };
    }
    for pair in s.split(',') {
        if let Some(("image", n)) = pair.split_once('=').map(|(k, v)| (k.trim(), v.trim())) {
            return n.parse().map_err(|_| format!("--limit-mm-per-prompt: bad count '{n}'"));
        }
    }
    Ok(999)
}

/// The image content parts of a body's messages, in prompt order: (url, detail). Other media are rejected here.
fn image_parts(body: &Value) -> Checked<Vec<(String, Option<String>)>> {
    const OTHER: [&str; 6] = ["video_url", "video", "input_audio", "audio_url", "audio", "input_video"];
    let mut out = Vec::new();
    let parts = body.get("messages").and_then(Value::as_array).into_iter().flatten().filter_map(|m| m.get("content")?.as_array());
    for part in parts.flatten() {
        let Some(t) = part.get("type").and_then(Value::as_str) else { continue };
        match t {
            "image_url" | "input_image" | "image" => {
                let src = part.get("image_url").or_else(|| part.get("image")).or_else(|| part.get("url"));
                let (url, detail) = match src {
                    Some(Value::String(u)) => (Some(u.clone()), part.get("detail").and_then(Value::as_str).map(str::to_string)),
                    Some(o) if o.is_object() => (
                        o.get("url").and_then(Value::as_str).map(str::to_string),
                        o.get("detail").or_else(|| part.get("detail")).and_then(Value::as_str).map(str::to_string),
                    ),
                    _ => (None, None),
                };
                let Some(url) = url else { return Err(bad_param("messages", format!("an '{t}' content part needs a url"))) };
                out.push((url, detail));
            }
            t if OTHER.contains(&t) => {
                return Err(bad_param("messages", format!("content part '{t}' is not supported: this server reads images only")));
            }
            _ => {}
        }
    }
    Ok(out)
}

/// The bytes behind an image URL.
fn load_url(cfg: &MmConfig, url: &str) -> std::result::Result<Vec<u8>, String> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',').ok_or("malformed data URL: no comma")?;
        if !meta.split(';').any(|p| p.eq_ignore_ascii_case("base64")) {
            return Err("data URLs of images must be base64".into());
        }
        let data: String = data.chars().filter(|c| !c.is_whitespace()).collect();
        let e = &base64::engine::general_purpose::STANDARD;
        return e.decode(&data).or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(&data)).map_err(|e| format!("invalid base64 image: {e}"));
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        let host = url.split("://").nth(1).unwrap_or("").split(['/', '?', '#']).next().unwrap_or("").rsplit('@').next().unwrap_or("");
        let host = host.rsplit_once(':').filter(|(_, p)| p.chars().all(|c| c.is_ascii_digit())).map_or(host, |(h, _)| h);
        if !cfg.allowed_domains.is_empty() && !cfg.allowed_domains.iter().any(|d| d == host) {
            return Err(format!("the URL's domain '{host}' is not allowed: allowed domains are {:?}", cfg.allowed_domains));
        }
        let agent = ureq::AgentBuilder::new().timeout(cfg.fetch_timeout).redirects(5).build();
        let resp = agent.get(url).call().map_err(|e| format!("cannot fetch the image: {e}"))?;
        let mut bytes = Vec::new();
        resp.into_reader().take(cfg.fetch_max_bytes as u64 + 1).read_to_end(&mut bytes).map_err(|e| format!("cannot fetch the image: {e}"))?;
        if bytes.len() > cfg.fetch_max_bytes {
            return Err(format!("the image is larger than the limit of {} bytes", cfg.fetch_max_bytes));
        }
        return Ok(bytes);
    }
    Err("the image URL must be a data: URL (base64) or an http(s) URL".into())
}

/// The images of a request body: counted against the per-prompt limit, loaded, sized. Empty for a text-only request.
/// The tokens the images will take are checked against the context before anything is decoded.
pub(super) async fn gather(st: &Arc<AppState>, body: &Value) -> Checked<Vec<Part>> {
    let urls = image_parts(body)?;
    if urls.is_empty() {
        return Ok(Vec::new());
    }
    let cfg = &st.mm;
    let Some(vision) = cfg.vision.clone() else {
        return Err(bad_param("messages", "this server takes no image input (it was started with --vision off, or the model has no vision tower)"));
    };
    if urls.len() > cfg.limit_images {
        return Err(bad_param("messages", format!("At most {} image(s) may be provided in one prompt.", cfg.limit_images)));
    }
    // request kwargs over the server's over the checkpoint's
    let request = match body.get("mm_processor_kwargs").filter(|v| !v.is_null()) {
        Some(v) => Overrides::parse(v).map_err(|e| bad_param("mm_processor_kwargs", e))?,
        None => Overrides::default(),
    };
    let base = vision.settings.apply(&cfg.kwargs.clone().over(request));
    if base.min_pixels > base.max_pixels {
        return Err(bad_param("mm_processor_kwargs", format!("min_pixels {} is above max_pixels {}", base.min_pixels, base.max_pixels)));
    }
    let st2 = st.clone();
    let parts = tokio::task::spawn_blocking(move || -> Checked<Vec<Part>> {
        let cfg = &st2.mm;
        let mut parts = Vec::new();
        let mut tokens = 0usize;
        for (url, detail) in urls {
            let bytes = load_url(cfg, &url).map_err(|e| bad_param("messages", e))?;
            let (w, h) = image::ImageReader::new(std::io::Cursor::new(&bytes))
                .with_guessed_format()
                .map_err(|e| bad_param("messages", format!("cannot read the image: {e}")))?
                .into_dimensions()
                .map_err(|e| bad_param("messages", format!("cannot decode the image: {e}")))?;
            let mut settings = base.clone();
            // OpenAI's detail: auto and high are the processor's own limits (vLLM ignores the field); low is the small
            // fixed-size reading, a cap on max_pixels
            if detail.as_deref() == Some("low") && cfg.low_detail_pixels > 0 {
                settings.max_pixels = settings.max_pixels.min(cfg.low_detail_pixels);
                settings.min_pixels = settings.min_pixels.min(settings.max_pixels);
            } else if let Some(d) = detail.as_deref().filter(|d| !["auto", "high", "low"].contains(d)) {
                return Err(bad_param("messages", format!("image_url.detail must be auto, low or high, not '{d}'")));
            }
            let grid = settings.merged_grid(w as usize, h as usize).map_err(|e| bad_param("messages", e))?;
            tokens += grid.0 * grid.1;
            // the image tokens alone must fit the context: refuse before decoding
            if tokens + 1 > st2.max_context {
                return Err(too_long(tokens, st2.max_context));
            }
            parts.push(Part { bytes, settings, grid });
        }
        Ok(parts)
    })
    .await
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, format!("image loading failed: {e}")))??;
    Ok(parts)
}

/// Expand the template's one image placeholder per image into the rows of its tokens: the token list and the first
/// row of each image. The count of placeholders must match the images.
pub(super) fn expand(st: &AppState, tokens: Vec<i64>, parts: &[Part]) -> Checked<(Vec<i64>, Vec<usize>)> {
    if parts.is_empty() {
        return Ok((tokens, Vec::new()));
    }
    let pad = st.mm.vision.as_ref().map(|v| v.cfg.image_token_id).unwrap_or(-1);
    let found = tokens.iter().filter(|&&t| t == pad).count();
    if found != parts.len() {
        return Err(bad_param("messages", format!("the chat template wrote {found} image placeholders for {} images", parts.len())));
    }
    let mut out = Vec::with_capacity(tokens.len() + parts.iter().map(Part::tokens).sum::<usize>());
    let mut starts = Vec::with_capacity(parts.len());
    for t in tokens {
        if t == pad {
            starts.push(out.len());
            out.extend(std::iter::repeat_n(pad, parts[starts.len() - 1].tokens()));
        } else {
            out.push(t);
        }
    }
    Ok((out, starts))
}

/// Decode, process and encode the images (on a blocking thread: the encoder owns the GPU or the CPU cores for the
/// image); the embeddings of an image seen recently come from the cache.
pub(super) async fn encode(st: &Arc<AppState>, parts: Vec<Part>, starts: Vec<usize>) -> Checked<Arc<MmInput>> {
    let st2 = st.clone();
    tokio::task::spawn_blocking(move || -> std::result::Result<Arc<MmInput>, String> {
        let vision = st2.mm.vision.clone().ok_or("no vision tower")?;
        let mut images = Vec::with_capacity(parts.len());
        for (p, start) in parts.into_iter().zip(starts) {
            let img: RgbImage = decode_image(&p.bytes)?;
            // the embeddings depend on the pixels and on how they were processed
            let key = hash_bytes(&img.data) ^ hash_bytes(format!("{:?}{}x{}", p.settings, img.w, img.h).as_bytes()).rotate_left(31);
            let cached = st2.mm.cache.lock().ok().and_then(|mut c| c.get(key));
            let emb = match cached {
                Some(e) => e,
                None => {
                    let patches = preprocess(&img, &p.settings)?;
                    let e = Arc::new(vision.encode(&patches).map_err(|e| format!("{e:#}"))?);
                    if let Ok(mut c) = st2.mm.cache.lock() {
                        c.put(key, e.clone());
                    }
                    e
                }
            };
            if emb.len() != p.tokens() * vision.cfg.out_hidden {
                return Err(format!("the encoder returned {} values for {} tokens", emb.len(), p.tokens()));
            }
            images.push(ImageEmbeds { start, grid_h: p.grid.0, grid_w: p.grid.1, hash: key, emb });
        }
        Ok(Arc::new(MmInput::new(images, vision.cfg.out_hidden)))
    })
    .await
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, format!("image encoding failed: {e}")))?
    .map_err(|e| bad_param("messages", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;

    fn cfg(timeout_ms: u64, max: usize, domains: &[&str]) -> MmConfig {
        MmConfig::new(None, Overrides::default(), 999, 0, Duration::from_millis(timeout_ms), max, domains.iter().map(|s| s.to_string()).collect())
    }

    /// A one-shot HTTP server on a free port answering with `body` after `delay_ms`.
    fn serve_once(body: Vec<u8>, delay_ms: u64) -> u16 {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = l.accept() {
                let mut buf = [0u8; 2048];
                let _ = s.read(&mut buf);
                std::thread::sleep(Duration::from_millis(delay_ms));
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = s.write_all(&body);
            }
        });
        port
    }

    #[test]
    fn data_urls() {
        let c = cfg(1000, 1 << 20, &[]);
        assert_eq!(load_url(&c, "data:image/png;base64,aGk=").unwrap(), b"hi");
        assert_eq!(load_url(&c, "data:image/png;base64,aGk").unwrap(), b"hi");
        assert!(load_url(&c, "data:image/png,hi").is_err());
        assert!(load_url(&c, "file:///etc/passwd").is_err());
    }

    #[test]
    fn http_fetch_limits() {
        let body = vec![7u8; 1000];
        let port = serve_once(body.clone(), 0);
        assert_eq!(load_url(&cfg(2000, 4096, &[]), &format!("http://127.0.0.1:{port}/a.png")).unwrap(), body);
        // larger than the limit
        let port = serve_once(body.clone(), 0);
        assert!(load_url(&cfg(2000, 500, &[]), &format!("http://127.0.0.1:{port}/a.png")).unwrap_err().contains("larger"));
        // too slow
        let port = serve_once(body.clone(), 1500);
        assert!(load_url(&cfg(300, 4096, &[]), &format!("http://127.0.0.1:{port}/a.png")).is_err());
        // domain list
        let port = serve_once(body, 0);
        let e = load_url(&cfg(2000, 4096, &["example.com"]), &format!("http://127.0.0.1:{port}/a.png")).unwrap_err();
        assert!(e.contains("not allowed"), "{e}");
    }

    #[test]
    fn limit_option() {
        assert_eq!(parse_limit("").unwrap(), 999);
        assert_eq!(parse_limit("image=4").unwrap(), 4);
        assert_eq!(parse_limit("video=1,image=2").unwrap(), 2);
        assert_eq!(parse_limit("{\"image\": 3}").unwrap(), 3);
        assert_eq!(parse_limit("{\"image\": {\"count\": 5, \"width\": 512}}").unwrap(), 5);
        assert_eq!(parse_limit("{\"video\": 1}").unwrap(), 999);
        assert!(parse_limit("image=x").is_err());
    }
}
