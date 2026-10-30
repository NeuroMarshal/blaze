//! Structured process logging with bounded, escaped fields.
//! Text output tracks load stages; JSON preserves event names and field values.
use clap::ValueEnum;
use serde_json::{Map, Value, json};
use std::ffi::{CStr, c_char, c_int};
use std::fmt::Write as _;
use std::io::{IsTerminal, Write};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Level {
    Error,
    Warn,
    #[default]
    Info,
    Debug,
    Trace,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum Format {
    #[default]
    Text,
    Json,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum Color {
    #[default]
    Auto,
    Always,
    Never,
}

pub struct Options {
    pub level: Level,
    pub format: Format,
    pub color: Color,
    pub requests: bool,
    pub inputs: bool,
    pub outputs: bool,
    pub max_len: usize,
    pub stats_secs: f64,
    pub progress_secs: f64,
    pub profile_prefill: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            level: Level::Info,
            format: Format::Text,
            color: Color::Auto,
            requests: true,
            inputs: false,
            outputs: false,
            max_len: 1000,
            stats_secs: 5.0,
            progress_secs: 5.0,
            profile_prefill: false,
        }
    }
}

static OPTIONS: OnceLock<Options> = OnceLock::new();
pub fn init(options: Options) {
    let _ = OPTIONS.set(options);
    unsafe {
        blaze_sys::blaze_set_log_callback(Some(cpp_log_callback));
        blaze_sys::blaze_set_stage_callback(Some(cpp_stage_callback));
    }
}

unsafe extern "C" fn cpp_log_callback(message: *const c_char) {
    if message.is_null() {
        return;
    }
    let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    legacy(format_args!("{message}"));
}

unsafe extern "C" fn cpp_stage_callback(op: c_int, id: i64, text: *const c_char, done: f64, total: f64, unit: *const c_char, ok: c_int) -> i64 {
    let s = |p: *const c_char| if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() };
    match op {
        0 => begin(s(text), total, &s(unit)),
        1 => {
            progress(id, done, total);
            0
        }
        _ => {
            end(id, &s(text), ok != 0);
            0
        }
    }
}

pub fn options() -> &'static Options {
    OPTIONS.get_or_init(Options::default)
}
pub fn enabled(level: Level) -> bool {
    level <= options().level
}

/// Colours (stderr is a terminal, or --log-color always) and the live progress bar (a terminal and text output).
struct Style {
    color: bool,
    live: bool,
}
fn style() -> &'static Style {
    static STYLE: OnceLock<Style> = OnceLock::new();
    STYLE.get_or_init(|| {
        let o = options();
        let tty = std::io::stderr().is_terminal() && std::env::var("TERM").map_or(true, |t| t != "dumb");
        let color = o.format == Format::Text
            && (o.color == Color::Always || (o.color == Color::Auto && tty && std::env::var_os("NO_COLOR").is_none()));
        Style { color, live: o.format == Format::Text && tty }
    })
}

const DIM: &str = "2";
const BOLD: &str = "1";
const CYAN: &str = "36";
const GREEN: &str = "32";
const RED: &str = "1;31";
const BLUE: &str = "1;34";
const YELLOW: &str = "1;33";
/// Request ids take one of these colours by their hash: the lines of one request stand out among the others.
const ID_COLORS: [&str; 6] = ["35", "33", "34", "32", "36", "91"];

fn paint(code: &str, text: &str) -> String {
    if style().color && !text.is_empty() { format!("\x1b[{code}m{text}\x1b[0m") } else { text.to_owned() }
}

fn level_code(level: Level) -> &'static str {
    match level {
        Level::Error => "1;31",
        Level::Warn => "1;33",
        Level::Info => "32",
        Level::Debug => "34",
        Level::Trace => "35",
    }
}

/// Local wall-clock time of day, HH:MM:SS.mmm.
fn clock() -> String {
    #[repr(C)]
    struct Tm {
        sec: c_int,
        min: c_int,
        hour: c_int,
        mday: c_int,
        mon: c_int,
        year: c_int,
        wday: c_int,
        yday: c_int,
        isdst: c_int,
        gmtoff: i64,
        zone: *const c_char,
    }
    unsafe extern "C" {
        fn localtime_r(t: *const i64, tm: *mut Tm) -> *mut Tm;
    }
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = t.as_secs() as i64;
    let mut tm = Tm { sec: 0, min: 0, hour: 0, mday: 0, mon: 0, year: 0, wday: 0, yday: 0, isdst: 0, gmtoff: 0, zone: std::ptr::null() };
    let (h, m, s) = if unsafe { localtime_r(&secs, &mut tm) }.is_null() {
        let d = secs % 86400;
        (d / 3600, d / 60 % 60, d % 60)
    } else {
        (tm.hour as i64, tm.min as i64, tm.sec as i64)
    };
    format!("{h:02}:{m:02}:{s:02}.{:03}", t.subsec_millis())
}

/// Width of the line prefix before the message: `HH:MM:SS.mmm LEVEL target   `.
const PREFIX: usize = 12 + 1 + 5 + 1 + 9;

fn compose(level: Level, target: &str, target_code: &str, depth: usize, message: &str, rest: &str) -> String {
    let name = format!("{level:?}").to_uppercase();
    let mut line = format!("{} {} {} {}{}", paint(DIM, &clock()), paint(level_code(level), &format!("{name:5}")),
                           paint(target_code, &format!("{target:8}")), "  ".repeat(depth), message);
    if !rest.is_empty() {
        line.push_str("  ");
        line.push_str(rest);
    }
    line
}

// value formatting

fn trim(s: String) -> String {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_owned() } else { s }
}

/// A number with the precision that reads well: integers as they are, three significant digits below 1, two decimals
/// below 10, one below 100, none above.
pub fn num(x: f64) -> String {
    let a = x.abs();
    if !x.is_finite() || x.fract() == 0.0 && a < 1e15 {
        return format!("{x}");
    }
    if a >= 100.0 {
        format!("{x:.0}")
    } else if a >= 10.0 {
        trim(format!("{x:.1}"))
    } else if a >= 1.0 {
        trim(format!("{x:.2}"))
    } else {
        let d = ((-a.log10()).floor() as usize + 3).min(9);
        trim(format!("{x:.d$}"))
    }
}

/// A duration: 850 µs, 667 ms, 1.52 s, 12.3 s, 5m38s, 1h04m.
pub fn dur(secs: f64) -> String {
    let s = secs.max(0.0);
    if s < 1e-3 {
        format!("{:.0} µs", s * 1e6)
    } else if s < 1.0 {
        format!("{} ms", num(s * 1e3))
    } else if s < 60.0 {
        format!("{} s", num(s))
    } else if s < 3600.0 {
        format!("{}m{:02}s", (s / 60.0) as u64, s as u64 % 60)
    } else {
        format!("{}h{:02}m", (s / 3600.0) as u64, s as u64 / 60 % 60)
    }
}

/// A size in binary units: 512 B, 3.7 KiB, 1.32 MiB, 64.6 GiB.
pub fn bytes(b: f64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let (mut v, mut i) = (b.max(0.0), 0);
    while v >= 1024.0 && i + 1 < UNITS.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 { format!("{v:.0} B") } else { format!("{} {}", num(v), UNITS[i]) }
}

fn f(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// A JSON value as text: numbers by `num`, plain strings bare, others compact JSON (nulls and empty entries left out).
fn plain(v: &Value) -> String {
    match v {
        Value::Number(_) => num(f(v).unwrap_or(0.0)),
        Value::String(s) if !s.is_empty() && !s.contains([' ', '"', '=', '\n', '\t', '\r']) && !s.chars().any(char::is_control) => s.clone(),
        Value::String(s) => format!("{s:?}"),
        Value::Array(a) if a.iter().all(|x| matches!(x, Value::String(_) | Value::Number(_))) => {
            a.iter().map(plain).collect::<Vec<_>>().join(",")
        }
        Value::Object(m) => {
            let parts: Vec<String> = m.iter().filter(|(_, v)| !empty(v)).map(|(k, v)| format!("{k}:{}", plain(v))).collect();
            format!("{{{}}}", parts.join(" "))
        }
        _ => v.to_string(),
    }
}

fn empty(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Array(a) => a.is_empty(),
        Value::Object(m) => m.is_empty(),
        Value::String(s) => s.is_empty(),
        _ => false,
    }
}

/// The remaining fields as `key=value`, units folded into the values.
fn pairs(fields: &Map<String, Value>) -> String {
    let mut out = Vec::new();
    for (k, v) in fields {
        if empty(v) {
            continue;
        }
        let x = f(v).filter(|_| v.is_number());
        let (key, val) = match x {
            Some(x) if k.ends_with("_tok_s") => (k.trim_end_matches("_tok_s").to_owned(), format!("{} tok/s", num(x))),
            Some(x) if k.ends_with("_ms") => (k.trim_end_matches("_ms").to_owned(), dur(x / 1e3)),
            Some(x) if k.ends_with("_s") || k.ends_with("_seconds") => {
                (k.trim_end_matches("_seconds").trim_end_matches("_s").to_owned(), dur(x))
            }
            Some(x) if k.ends_with("_bytes") => (k.trim_end_matches("_bytes").to_owned(), bytes(x)),
            Some(x) if k.ends_with("_pct") => (k.trim_end_matches("_pct").to_owned(), format!("{}%", num(x))),
            _ => (k.clone(), plain(v)),
        };
        out.push(format!("{}{val}", paint(DIM, &format!("{key}="))));
    }
    out.join(" ")
}

// the events

/// Who a line is about: a component, or a request (its id's first 8 characters, in its own colour).
struct Target {
    text: String,
    code: &'static str,
}

impl Target {
    fn of(name: &str) -> Self {
        Self { text: name.to_owned(), code: CYAN }
    }
}

fn take(m: &mut Map<String, Value>, k: &str) -> Option<Value> {
    m.remove(k).filter(|v| !v.is_null())
}
fn take_f(m: &mut Map<String, Value>, k: &str) -> Option<f64> {
    take(m, k).as_ref().and_then(f)
}
fn take_s(m: &mut Map<String, Value>, k: &str) -> Option<String> {
    take(m, k).map(|v| match v {
        Value::String(s) => s,
        v => v.to_string(),
    })
}
fn tok_s(x: f64) -> String {
    paint(BOLD, &format!("{} tok/s", num(x)))
}
/// A count or time the sentence is about, in bold.
fn hi(text: impl AsRef<str>) -> String {
    paint(BOLD, text.as_ref())
}
/// A problem counter: yellow (red for errors) when it is not zero.
fn alarm(name: &str, x: f64) -> String {
    let text = format!("{name} {}", num(x));
    if x == 0.0 { text } else { paint(if name == "errors" { RED } else { YELLOW }, &text) }
}

/// Sampling parameters that differ from the neutral ones, `temperature=0.75 top_p=0.95 …`.
fn sampling(v: &Value) -> String {
    let Some(m) = v.as_object() else { return plain(v) };
    let neutral = |k: &str, v: &Value| match (k, f(v)) {
        ("min_p" | "frequency_penalty" | "presence_penalty" | "min_tokens", Some(x)) => x == 0.0,
        ("repetition_penalty" | "top_p", Some(x)) => x == 1.0,
        _ => empty(v),
    };
    m.iter().filter(|(k, v)| !neutral(k, v)).map(|(k, v)| format!("{k}={}", plain(v))).collect::<Vec<_>>().join(" ")
}

/// The target and message of an event; the fields it uses leave `m`.
fn render(name: &str, m: &mut Map<String, Value>) -> (Target, String) {
    let id = take_s(m, "request_id");
    let choice = take_f(m, "choice").unwrap_or(0.0) as usize;
    let target = match &id {
        Some(id) => {
            let short: String = id.chars().take(8).collect();
            let h = id.bytes().fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize));
            Target { text: if choice > 0 { format!("{}/{choice}", &short[..short.len().min(6)]) } else { short }, code: ID_COLORS[h % ID_COLORS.len()] }
        }
        None => Target::of(name.split('.').next().unwrap_or(name)),
    };
    let mut s = String::new();
    match name {
        "http.received" => {
            let method = take_s(m, "method").unwrap_or_default();
            let path = take_s(m, "path").unwrap_or_default();
            let _ = write!(s, "{} {method} {}", paint(BLUE, "→"), hi(&path));
            if let Some(b) = take_f(m, "content_length") {
                let _ = write!(s, " · {}", bytes(b));
            }
            if let Some(p) = take_s(m, "peer") {
                let _ = write!(s, " · from {p}");
            }
            if let Some(ua) = take_s(m, "user_agent") {
                let _ = write!(s, " · {ua}");
            }
        }
        "http.finished" => {
            let status = take_f(m, "status").unwrap_or(0.0);
            let path = take_s(m, "path").unwrap_or_default();
            let code = if status >= 500.0 { RED } else if status >= 400.0 { YELLOW } else { GREEN };
            let _ = write!(s, "{} {} {path}", paint(code, "←"), paint(code, &num(status)));
            if let Some(ms) = take_f(m, "elapsed_ms") {
                let _ = write!(s, " · {}", hi(dur(ms / 1e3)));
            }
            if let Some(b) = take_f(m, "response_bytes") {
                let _ = write!(s, " · {}", bytes(b));
            }
            if take(m, "complete") == Some(Value::Bool(false)) {
                s.push_str(&paint(YELLOW, " · the client left before the end"));
            }
        }
        "request.admitted" => {
            let prompt = take_f(m, "prompt_tokens").unwrap_or(0.0);
            let cached = take_f(m, "cached_tokens").unwrap_or(0.0);
            let _ = write!(s, "admitted · prompt {} tokens", hi(num(prompt)));
            if cached > 0.0 {
                let _ = write!(s, " ({} cached, {}%)", num(cached), num((100.0 * cached / prompt.max(1.0)).round()));
            }
            if let Some(x) = take_f(m, "max_tokens") {
                let _ = write!(s, " · up to {} new", num(x));
            }
            if let Some(ms) = take_f(m, "queue_ms") {
                let _ = write!(s, " · queued {}", dur(ms / 1e3));
            }
            if let (Some(r), Some(w)) = (take_f(m, "running"), take_f(m, "waiting")) {
                let _ = write!(s, " · running {}, waiting {}", num(r), num(w));
            }
            if let Some(p) = take(m, "sampling") {
                let _ = write!(s, " · sampling {}", sampling(&p));
            }
        }
        "request.prefill" => {
            let fed = take_f(m, "processed_tokens").unwrap_or(0.0);
            let prompt = take_f(m, "prompt_tokens").unwrap_or(0.0);
            let pct = take_f(m, "progress_pct").unwrap_or(0.0);
            let _ = write!(s, "prefill {}/{} ({})", hi(num(fed)), num(prompt), hi(format!("{}%", num((pct * 10.0).round() / 10.0))));
            if let (Some(n), Some(ms)) = (take_f(m, "chunk_tokens"), take_f(m, "chunk_ms")) {
                let _ = write!(s, " · chunk of {} in {}", num(n), dur(ms / 1e3));
                if let Some(r) = take_f(m, "chunk_tok_s") {
                    let _ = write!(s, " ({} tok/s)", num(r));
                }
            }
            if let Some(r) = take_f(m, "prefill_tok_s") {
                let _ = write!(s, " · average {}", tok_s(r));
                if r > 0.0 && prompt > fed {
                    let _ = write!(s, " · ETA {}", paint(YELLOW, &dur((prompt - fed) / r)));
                }
            }
            if let Some(c) = take_f(m, "cached_tokens").filter(|&c| c > 0.0) {
                let _ = write!(s, " · {} cached", num(c));
            }
        }
        "request.first_token" => {
            let _ = write!(s, "first token");
            if let Some(ms) = take_f(m, "ttft_ms") {
                let _ = write!(s, " after {}", hi(dur(ms / 1e3)));
            }
            if let Some(n) = take_f(m, "prefill_tokens") {
                let _ = write!(s, " · prefilled {} tokens", num(n));
                if let Some(r) = take_f(m, "prefill_tok_s").filter(|_| n > 0.0) {
                    let _ = write!(s, " at {}", tok_s(r));
                }
            }
            if let Some(c) = take_f(m, "cached_tokens").filter(|&c| c > 0.0) {
                let _ = write!(s, " · {} from the prefix cache", num(c));
            }
        }
        "request.decode" => {
            let _ = write!(s, "decoding · {} tokens out", hi(num(take_f(m, "output_tokens").unwrap_or(0.0))));
            if let Some(c) = take_f(m, "context_tokens") {
                let _ = write!(s, " · context {}", num(c));
            }
            if let Some(r) = take_f(m, "decode_tok_s") {
                let _ = write!(s, " · {}", tok_s(r));
            }
        }
        "request.finished" => {
            let reason = take_s(m, "finish_reason").unwrap_or_default().to_lowercase();
            let _ = write!(s, "{} finished ({reason})", paint(GREEN, "✔"));
            let prompt = take_f(m, "prompt_tokens").unwrap_or(0.0);
            let _ = write!(s, " · prompt {}", num(prompt));
            if let Some(c) = take_f(m, "cached_tokens").filter(|&c| c > 0.0) {
                let _ = write!(s, " ({} cached)", num(c));
            }
            let _ = write!(s, " → {} tokens", hi(num(take_f(m, "output_tokens").unwrap_or(0.0))));
            if let Some(ms) = take_f(m, "e2e_ms") {
                let _ = write!(s, " in {}", hi(dur(ms / 1e3)));
            }
            if let (Some(ms), Some(r)) = (take_f(m, "prefill_ms"), take_f(m, "prefill_tok_s")) {
                let _ = write!(s, " · prefill {} ({})", dur(ms / 1e3), tok_s(r));
            }
            let decode = take_f(m, "decode_ms");
            if let (Some(ms), Some(r)) = (decode, take_f(m, "decode_tok_s")) {
                let _ = write!(s, " · decode {} ({}", dur(ms / 1e3), tok_s(r));
                if let Some(t) = take_f(m, "tpot_ms") {
                    let _ = write!(s, ", {} a token", dur(t / 1e3));
                }
                s.push(')');
            }
        }
        "engine.stats" => {
            let r = take_f(m, "running").unwrap_or(0.0);
            let w = take_f(m, "waiting").unwrap_or(0.0);
            if let Some(i) = take_f(m, "interval_s") {
                let _ = write!(s, "last {} · ", dur(i));
            }
            let _ = write!(s, "running {} · waiting {}", hi(num(r)), hi(num(w)));
            if let Some(p) = take_f(m, "waiting_peak") {
                let _ = write!(s, " (peak {})", num(p));
            }
            let pre = take_f(m, "prefill_tokens").unwrap_or(0.0);
            let pre_r = take_f(m, "prefill_tok_s").unwrap_or(0.0);
            let _ = write!(s, " · prefill {} ({} tokens)", tok_s(if pre > 0.0 { pre_r } else { 0.0 }), num(pre));
            let out = take_f(m, "output_tokens").unwrap_or(0.0);
            let dec = take_f(m, "decode_tok_s").unwrap_or(0.0);
            let _ = write!(s, " · decode {} ({} tokens", tok_s(if out > 0.0 { dec } else { 0.0 }), num(out));
            if let Some(x) = take_f(m, "output_tok_s_wall") {
                let _ = write!(s, ", {} tok/s over the wall clock", num(x));
            }
            s.push(')');
            if let Some(x) = take_f(m, "spec_tokens_per_verify") {
                let _ = write!(s, " · {} tokens per verify", hi(num(x)));
            }
            if let (Some(p), Some(c)) = (take_f(m, "admitted_prompt_tokens"), take_f(m, "cached_tokens")) {
                let _ = write!(s, " · admitted prompts {} ({} cached)", num(p), num(c));
            }
            if let (Some(u), Some(free), Some(total)) = (take_f(m, "kv_usage_pct"), take_f(m, "kv_pages_free"), take_f(m, "kv_pages_total")) {
                let _ = write!(s, " · KV {} used ({} of {} pages free)", hi(format!("{}%", num((u * 10.0).round() / 10.0))), num(free), num(total));
            }
            if let Some(c) = take_f(m, "cache_entries") {
                let _ = write!(s, " · prefix cache {}", num(c));
            }
            if let (Some(mean), Some(max)) = (take_f(m, "step_mean_ms"), take_f(m, "step_max_ms")) {
                let _ = write!(s, " · step {} (max {})", dur(mean / 1e3), dur(max / 1e3));
            }
            let _ = write!(s, " · finished {}", num(take_f(m, "completed").unwrap_or(0.0)));
            for k in ["errors", "cancelled", "preempted", "kv_full"] {
                let _ = write!(s, " · {}", alarm(&k.replace('_', " "), take_f(m, k).unwrap_or(0.0)));
            }
        }
        "api.error" => {
            let _ = write!(s, "{} {}", num(take_f(m, "status").unwrap_or(0.0)), take_s(m, "message").unwrap_or_default());
        }
        "scheduler.latency_outlier" => {
            let _ = write!(s, "slow step: {}", dur(take_f(m, "elapsed_ms").unwrap_or(0.0) / 1e3));
            if let Some(x) = take_f(m, "phase_mean_ms") {
                let _ = write!(s, " (usually {})", dur(x / 1e3));
            }
            if let Some(x) = take_s(m, "shape") {
                let _ = write!(s, " · {x}");
            }
        }
        "gpu.prefill" => {
            let rows = take_f(m, "rows").unwrap_or(0.0);
            let _ = write!(s, "prefill of {} rows in {}", num(rows), dur(take_f(m, "total_ms").unwrap_or(0.0) / 1e3));
            for (k, label) in [("ple_ms", "PLE"), ("attention_ms", "attention"), ("gdn_ms", "GDN"), ("mlp_ms", "MLP"), ("other_ms", "other")] {
                if let Some(x) = take_f(m, k) {
                    let _ = write!(s, " · {label} {}", dur(x / 1e3));
                }
            }
        }
        _ => {
            match take(m, "message") {
                Some(Value::String(msg)) => s = msg,
                Some(v) => {
                    m.insert("message".into(), v);
                }
                None => {}
            }
            if s.is_empty() {
                s = name.split_once('.').map_or(name, |(_, rest)| rest).replace(['.', '_'], " ");
            }
        }
    }
    (target, s)
}

pub fn event(level: Level, stage: &str, fields: Value) {
    if !enabled(level) {
        return;
    }
    if options().format == Format::Json {
        let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let name = format!("{level:?}").to_uppercase();
        write_line(&json!({"time_unix_ms": t.as_millis() as u64, "level": name, "event": stage, "fields": fields}).to_string());
        return;
    }
    let mut m = match fields {
        Value::Object(m) => m,
        Value::Null => Map::new(),
        v => Map::from_iter([("value".to_owned(), v)]),
    };
    let (target, message) = render(stage, &mut m);
    let depth = board().tasks.len();
    write_line(&compose(level, &target.text, target.code, depth, &message, &pairs(&m)));
}

/// A titled table: in text, the title and then one aligned `key  value` row a line; in JSON, one event of the rows.
pub fn block(level: Level, target: &str, title: &str, rows: &[(String, String)]) {
    if !enabled(level) {
        return;
    }
    if options().format == Format::Json {
        let mut m = Map::new();
        m.insert("title".into(), Value::String(title.to_owned()));
        for (k, v) in rows {
            m.insert(k.clone(), Value::String(v.clone()));
        }
        event(level, &format!("{target}.summary"), Value::Object(m));
        return;
    }
    let depth = board().tasks.len();
    let width = rows.iter().map(|(k, _)| k.chars().count()).max().unwrap_or(0);
    let mut text = compose(level, target, CYAN, depth, title, "");
    let pad = " ".repeat(PREFIX + 2 * depth + 2);
    for (k, v) in rows {
        let key = format!("{k:width$}");
        let _ = write!(text, "\n{pad}{}  {v}", paint(DIM, &key));
    }
    write_line(&text);
}

pub fn legacy(args: std::fmt::Arguments<'_>) {
    let message = args.to_string();
    let level = if message.contains("panicked") || message.contains("step failed") || message.contains("blaze hang") {
        Level::Error
    } else if message.contains("failed")
        || message.contains("disabled:")
        || message.contains("KV pool full")
        || message.contains("grammar dropped")
    {
        Level::Warn
    } else {
        Level::Info
    };
    if !enabled(level) {
        return;
    }
    // "blaze: text" or "blaze <part>: text"
    let (part, text) = match message.strip_prefix("blaze") {
        Some(rest) => match rest.split_once(':') {
            Some((part, text)) if part.len() <= 16 => (part.trim(), text.trim_start()),
            _ => ("", message.as_str()),
        },
        None => ("", message.as_str()),
    };
    let target = legacy_target(part, text);
    if options().format == Format::Json {
        event(level, "runtime", json!({"target": target, "message": text}));
        return;
    }
    let depth = board().tasks.len();
    write_line(&compose(level, target, CYAN, depth, text, ""));
}

/// The component a runtime message is about, by its prefix.
fn legacy_target(part: &str, text: &str) -> &'static str {
    match part {
        "experts" | "swap" | "GPU1 tier" | "hybrid" => return "experts",
        "hang" => return "watchdog",
        _ => {}
    }
    const PREFIXES: [(&str, &str); 22] = [
        ("VRAM", "memory"),
        ("largest activation", "memory"),
        ("MoE engine", "memory"),
        ("KV", "kv"),
        ("NVFP4", "kernels"),
        ("attention", "kernels"),
        ("dense causal", "kernels"),
        ("tuned", "kernels"),
        ("verify attention", "kernels"),
        ("prefill attention", "kernels"),
        ("--prefill-a16", "kernels"),
        ("speculation", "spec"),
        ("DFlash", "spec"),
        ("EAGLE", "spec"),
        ("MTP", "spec"),
        ("grammar", "grammar"),
        ("structured", "grammar"),
        ("PLE", "ple"),
        ("all-reduce", "tp"),
        ("pipeline", "tp"),
        ("vision", "vision"),
        ("chat completions", "server"),
    ];
    if text.starts_with("routed experts") || text.contains("CPU expert") || text.contains(" experts from ") {
        return "experts";
    }
    PREFIXES.iter().find(|(p, _)| text.starts_with(p)).map_or("engine", |(_, t)| t)
}

// stages and progress

struct Task {
    id: i64,
    label: String,
    t0: Instant,
    done: f64,
    total: f64,
    unit: String,
}

/// A finished stage: its id (stages begin in id order), nesting depth, label, seconds and whether it succeeded.
#[derive(Clone, Debug)]
pub struct Finished {
    pub id: i64,
    pub depth: usize,
    pub label: String,
    pub secs: f64,
    pub ok: bool,
}

struct Board {
    tasks: Vec<Task>,
    finished: Vec<Finished>,
    next: i64,
    /// a progress bar is on the terminal's last line
    bar: bool,
    ticker: bool,
    noted: Option<Instant>,
    spin: usize,
}

static BOARD: Mutex<Board> =
    Mutex::new(Board { tasks: Vec::new(), finished: Vec::new(), next: 0, bar: false, ticker: false, noted: None, spin: 0 });

fn board() -> MutexGuard<'static, Board> {
    BOARD.lock().unwrap_or_else(|e| e.into_inner())
}

/// Write one line (or a block of lines) with the progress bar kept under it.
fn write_line(line: &str) {
    let mut b = board();
    emit(&mut b, line);
}

fn emit(b: &mut Board, line: &str) {
    let mut out = String::with_capacity(line.len() + 160);
    if b.bar {
        out.push_str("\r\x1b[2K");
        b.bar = false;
    }
    out.push_str(line);
    out.push('\n');
    if style().live && !b.tasks.is_empty() {
        out.push_str(&bar_line(b));
        b.bar = true;
    }
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(out.as_bytes());
    let _ = err.flush();
}

fn term_width() -> usize {
    #[repr(C)]
    struct Winsize {
        row: u16,
        col: u16,
        x: u16,
        y: u16,
    }
    unsafe extern "C" {
        fn ioctl(fd: c_int, request: std::ffi::c_ulong, ...) -> c_int;
    }
    let mut w = Winsize { row: 0, col: 0, x: 0, y: 0 };
    // TIOCGWINSZ
    if unsafe { ioctl(2, 0x5413, &mut w) } == 0 && w.col >= 20 { w.col as usize } else { 100 }
}

/// `41% · 26.5/64.6 GiB · 1.21 GB/s · 22 s · ETA 31 s`, or the time so far without a total.
fn status(t: &Task) -> String {
    let secs = t.t0.elapsed().as_secs_f64();
    if t.total <= 0.0 {
        return dur(secs);
    }
    let bytes_unit = t.unit == "B";
    let frac = (t.done / t.total).clamp(0.0, 1.0);
    let amount = if bytes_unit {
        let (div, u) = if t.total >= (1u64 << 30) as f64 { ((1u64 << 30) as f64, "GiB") } else { ((1u64 << 20) as f64, "MiB") };
        format!("{}/{} {u}", num((t.done / div * 10.0).round() / 10.0), num((t.total / div * 10.0).round() / 10.0))
    } else {
        format!("{}/{} {}", num(t.done), num(t.total), t.unit)
    };
    let mut s = format!("{}% · {amount}", (frac * 100.0).floor() as u64);
    if secs > 0.5 && t.done > 0.0 {
        let rate = t.done / secs;
        if bytes_unit {
            let _ = write!(s, " · {} GB/s", num(rate / 1e9));
        }
        let _ = write!(s, " · {}", dur(secs));
        if t.done < t.total {
            let _ = write!(s, " · ETA {}", dur((t.total - t.done) / rate));
        }
    } else {
        let _ = write!(s, " · {}", dur(secs));
    }
    s
}

fn bar_line(b: &Board) -> String {
    const SPIN: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let Some(t) = b.tasks.last() else { return String::new() };
    let width = term_width().saturating_sub(1);
    let st = status(t);
    let cells = 24;
    let bar = if t.total > 0.0 {
        let eighths = ((t.done / t.total).clamp(0.0, 1.0) * (cells * 8) as f64) as usize;
        let mut s: String = "█".repeat(eighths / 8);
        if eighths % 8 > 0 {
            s.push([' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'][eighths % 8]);
        }
        let n = s.chars().count();
        s.extend(std::iter::repeat_n('░', cells - n.min(cells)));
        s
    } else {
        String::new()
    };
    // the label's room leaves the longest status (`100% · 64.6/64.6 GiB · 1.21 GB/s · 59.9 s · ETA 59.9 s`): the bar
    // stays in place while the numbers change
    let fixed = 3 + 2 + bar.chars().count() + 1 + if t.total > 0.0 { 62 } else { 10 };
    let room = width.saturating_sub(fixed).max(12);
    let mut label: String = t.label.chars().take(room).collect();
    if t.label.chars().count() > room {
        label.pop();
        label.push('…');
    }
    let bar = if bar.is_empty() { String::new() } else { format!("{} ", paint(CYAN, &bar)) };
    format!(" {} {}  {bar}{}", paint(CYAN, &SPIN[b.spin % SPIN.len()].to_string()), paint(BOLD, &label), st)
}

fn ticker() {
    loop {
        std::thread::sleep(Duration::from_millis(100));
        let mut b = board();
        if b.tasks.is_empty() {
            b.ticker = false;
            return;
        }
        b.spin += 1;
        if style().live {
            let line = bar_line(&b);
            let mut err = std::io::stderr().lock();
            let _ = write!(err, "\r\x1b[2K{line}");
            let _ = err.flush();
            b.bar = true;
            continue;
        }
        let every = options().progress_secs;
        if every > 0.0 && b.noted.is_some_and(|n| n.elapsed().as_secs_f64() >= every) {
            b.noted = Some(Instant::now());
            let t = b.tasks.last().unwrap();
            let line = if options().format == Format::Json {
                let t = b.tasks.last().unwrap();
                json!({"time_unix_ms": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64,
                       "level": "INFO", "event": "stage.progress",
                       "fields": {"stage": t.label, "done": t.done, "total": t.total, "unit": t.unit, "elapsed_s": t.t0.elapsed().as_secs_f64()}})
                .to_string()
            } else {
                compose(Level::Info, "load", CYAN, b.tasks.len(), &format!("… {}: {}", t.label, status(t)), "")
            };
            emit(&mut b, &line);
        }
    }
}

fn begin(label: String, total: f64, unit: &str) -> i64 {
    let mut b = board();
    let id = b.next;
    b.next += 1;
    let depth = b.tasks.len();
    if enabled(Level::Info) {
        let line = if options().format == Format::Json {
            json!({"time_unix_ms": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64,
                   "level": "INFO", "event": "stage.begin", "fields": {"stage": label, "depth": depth}})
            .to_string()
        } else {
            compose(Level::Info, "load", CYAN, depth, &format!("{} {label}", paint(BLUE, "▶")), "")
        };
        emit(&mut b, &line);
    }
    b.tasks.push(Task { id, label, t0: Instant::now(), done: 0.0, total, unit: unit.to_owned() });
    b.noted = Some(Instant::now());
    if !b.ticker {
        b.ticker = std::thread::Builder::new().name("log-progress".into()).spawn(ticker).is_ok();
    }
    id
}

fn progress(id: i64, done: f64, total: f64) {
    let mut b = board();
    if let Some(t) = b.tasks.iter_mut().find(|t| t.id == id) {
        t.done = done;
        if total >= 0.0 {
            t.total = total;
        }
    }
}

fn end(id: i64, note: &str, ok: bool) -> f64 {
    let mut b = board();
    let Some(i) = b.tasks.iter().position(|t| t.id == id) else { return 0.0 };
    let t = b.tasks.remove(i);
    let secs = t.t0.elapsed().as_secs_f64();
    b.finished.push(Finished { id: t.id, depth: i, label: t.label.clone(), secs, ok });
    let level = if ok { Level::Info } else { Level::Error };
    if enabled(level) {
        let line = if options().format == Format::Json {
            json!({"time_unix_ms": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64,
                   "level": format!("{level:?}").to_uppercase(), "event": "stage.end",
                   "fields": {"stage": t.label, "depth": i, "elapsed_s": secs, "ok": ok, "note": note}})
            .to_string()
        } else if ok {
            let note = if note.is_empty() { String::new() } else { format!(" · {note}") };
            compose(level, "load", CYAN, i, &format!("{} {} · {}{note}", paint(GREEN, "✔"), t.label, paint(BOLD, &dur(secs))), "")
        } else {
            compose(level, "load", CYAN, i, &format!("{} {} · failed after {}", paint(RED, "✖"), t.label, dur(secs)), "")
        };
        emit(&mut b, &line);
    }
    b.noted = Some(Instant::now());
    secs
}

/// A stage of the load (see the module's comment). It ends with `done`; dropped before, it reports a failure.
#[must_use = "a stage ends with done(); dropped unfinished it reports a failure"]
pub struct Stage {
    id: i64,
    open: bool,
}

pub fn stage(label: impl Into<String>) -> Stage {
    Stage { id: begin(label.into(), 0.0, ""), open: true }
}

impl Stage {
    /// A stage that prints nothing (another tensor-parallel rank's copy of rank 0's).
    pub fn none() -> Self {
        Self { id: -1, open: false }
    }
    /// The amount the stage works through (`unit` "B": bytes, shown in GiB with a GB/s rate).
    pub fn total(self, total: f64, unit: &str) -> Self {
        if self.open {
            let mut b = board();
            if let Some(t) = b.tasks.iter_mut().find(|t| t.id == self.id) {
                t.total = total;
                t.unit = unit.to_owned();
            }
        }
        self
    }
    pub fn progress(&self, done: f64) {
        if self.open {
            progress(self.id, done, -1.0);
        }
    }
    /// End the stage; `note` follows its time. The stage's seconds.
    pub fn done(mut self, note: impl AsRef<str>) -> f64 {
        if !self.open {
            return 0.0;
        }
        self.open = false;
        end(self.id, note.as_ref(), true)
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if self.open {
            end(self.id, "", false);
        }
    }
}

/// The stages that have finished, in the order they ended.
pub fn finished() -> Vec<Finished> {
    board().finished.clone()
}

/// The running stages, outermost first, for a loading server's answers: `[{stage, elapsed_s, done, total, unit}]`.
pub fn running() -> Vec<Value> {
    board()
        .tasks
        .iter()
        .map(|t| {
            let mut v = json!({"stage": t.label, "elapsed_s": (t.t0.elapsed().as_secs_f64() * 10.0).round() / 10.0});
            if t.total > 0.0 {
                v["done"] = json!(t.done);
                v["total"] = json!(t.total);
                v["unit"] = json!(t.unit);
                v["progress_pct"] = json!((1000.0 * t.done / t.total).round() / 10.0);
            }
            v
        })
        .collect()
}

/// The innermost running stage in words, `reading … (41% · 26.5/64.6 GiB · ETA 31 s)`.
pub fn running_text() -> Option<String> {
    let b = board();
    b.tasks.last().map(|t| format!("{} ({})", t.label, status(t)))
}

/// UTF-8 safe, bounded preview. Formatting by the logger escapes control characters.
pub fn preview(text: &str, max: usize) -> String {
    let end = text.char_indices().nth(max).map_or(text.len(), |(i, _)| i);
    let mut value = text[..end].to_owned();
    if end < text.len() {
        value.push_str("…");
    }
    value
}

#[macro_export]
macro_rules! eprintln {
    ($($arg:tt)*) => { $crate::logging::legacy(format_args!($($arg)*)) };
}

#[derive(Clone, Debug)]
pub struct RequestTrace {
    pub id: String,
    pub received: Instant,
    pub choice: usize,
    pub cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Next Responses SSE sequence number, shared with the HTTP deadline handler.
    pub stream_sequence: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl Default for RequestTrace {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().simple().to_string(),
            received: Instant::now(),
            choice: 0,
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stream_sequence: Default::default(),
        }
    }
}

tokio::task_local! { pub static REQUEST: RequestTrace; }

pub fn request_trace(choice: usize) -> RequestTrace {
    let mut trace = REQUEST.try_with(Clone::clone).unwrap_or_default();
    trace.choice = choice;
    trace
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare(s: &str) -> String {
        let mut out = String::new();
        let mut esc = false;
        for c in s.chars() {
            match (esc, c) {
                (false, '\x1b') => esc = true,
                (true, 'm') => esc = false,
                (true, _) => {}
                (false, c) => out.push(c),
            }
        }
        out.replace('\u{a0}', " ")
    }

    #[test]
    fn preview_keeps_utf8_and_bounds_chars() {
        assert_eq!(super::preview("Привет\nмир", 3), "При…");
        assert_eq!(super::preview("你好", 2), "你好");
    }

    #[test]
    fn numbers_durations_sizes_read_well() {
        assert_eq!(num(0.949999988079071), "0.95");
        assert_eq!(num(0.30000001192092896), "0.3");
        assert_eq!(num(0.0012), "0.0012");
        assert_eq!(num(667.1163829999999), "667");
        assert_eq!(num(12.34), "12.3");
        assert_eq!(num(786432.0), "786432");
        assert_eq!(dur(0.6671), "667 ms");
        assert_eq!(dur(1.5234), "1.52 s");
        assert_eq!(dur(338.2), "5m38s");
        assert_eq!(dur(3840.0), "1h04m");
        assert_eq!(bytes(1379844.0), "1.32 MiB");
        assert_eq!(bytes(64.6 * (1u64 << 30) as f64), "64.6 GiB");
    }

    #[test]
    fn request_events_read_as_sentences() {
        let mut m = json!({"request_id":"56a8c9b8bf2d43e6b31b08ed562ed90e","choice":0,"finish_reason":"Stop","prompt_tokens":67514,
            "cached_tokens":63868,"output_tokens":349,"e2e_ms":4852.94,"prefill_ms":1656.7,"decode_ms":2246.5,"prefill_tok_s":2200.76,
            "decode_tok_s":154.9,"tpot_ms":6.4556}).as_object().unwrap().clone();
        let (target, s) = render("request.finished", &mut m);
        let s = bare(&s);
        assert_eq!(target.text, "56a8c9b8");
        assert!(s.ends_with("finished (stop) · prompt 67514 (63868 cached) → 349 tokens in 4.85 s · prefill 1.66 s (2201 tok/s) · decode 2.25 s (155 tok/s, 6.46 ms a token)"), "{s}");
        assert!(m.is_empty(), "{m:?}");
        let mut m = json!({"request_id":"b1aaad814c854bb98d2acd6479ba8e6b","choice":0,"processed_tokens":16384,"prompt_tokens":396108,
            "cached_tokens":0,"progress_pct":4.136,"chunk_tokens":2048,"chunk_ms":667.1,"chunk_tok_s":3069.9,"prefill_tok_s":3192.8,
            "kv_pages_free":525568}).as_object().unwrap().clone();
        let (_, s) = render("request.prefill", &mut m);
        assert_eq!(bare(&s), "prefill 16384/396108 (4.1%) · chunk of 2048 in 667 ms (3070 tok/s) · average 3193 tok/s · ETA 1m58s");
        assert_eq!(bare(&pairs(&m)), "kv_pages_free=525568");
    }

    #[test]
    fn runtime_messages_find_their_component() {
        assert_eq!(legacy_target("", "VRAM (GiB) on GPU0: weights 5.85"), "memory");
        assert_eq!(legacy_target("experts", "17926 tok in 60 s"), "experts");
        assert_eq!(legacy_target("", "routed experts: 49 layers x 512"), "experts");
        assert_eq!(legacy_target("", "speculation: the drafter"), "spec");
        assert_eq!(legacy_target("", "something new"), "engine");
    }

    /// The look of a load and a request, without a GPU:
    /// `cargo test --release -p blaze --lib logging::tests::demo -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn demo() {
        let rows: Vec<(String, String)> = [("model", "/models/Demo-NVFP4"), ("listen", "http://0.0.0.0:8100/v1"), ("KV cache", "786432 tokens · nvfp4_4over6 · in pinned host RAM")]
            .iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        block(Level::Info, "blaze", "blaze 0.1.0 · serve Demo-NVFP4", &rows);
        event(Level::Info, "http.listening", json!({"message":"listening on http://0.0.0.0:8100: until the model is ready, requests get 503 with the load's progress (GET /health)"}));
        let engine = stage("engine: Demo-NVFP4 on 1 GPU");
        let w = stage("backbone weights (routed experts come later) → GPU0").total(48.0, "layers");
        for i in 0..48 {
            std::thread::sleep(Duration::from_millis(25));
            w.progress((i + 1) as f64);
        }
        w.done("48 layers, 12 softmax + 36 GDN, and the LM head");
        crate::eprintln!("blaze: largest activation buffers: proj 129 MiB, partial_o 96 MiB, conv 80 MiB");
        let rows: Vec<(String, String)> = [("weights (no routed experts)", "  5.85 GiB"), ("KV pools + QSA keys", "  0.60 GiB"), ("free now", "  7.07 GiB")]
            .iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        block(Level::Info, "memory", "VRAM of GPU0 so far: 6.45 GiB in these parts", &rows);
        let e = stage("routed experts → GPU0 + GPU1 slots, pinned RAM (34 CPU expert workers)");
        crate::eprintln!("blaze: routed experts: 49 layers x 512 = 25088 experts of 2.64 MiB: GPU0 2079, GPU1 5793, RAM 17216");
        let p = stage("pinning 44.4 GiB of host RAM for 17216 RAM-tier expert slots");
        std::thread::sleep(Duration::from_millis(700));
        p.done("1.09 GiB/s");
        let total = 64.6 * (1u64 << 30) as f64;
        let r = stage("reading 64.6 GiB of experts from the checkpoint (16 threads, O_DIRECT)").total(total, "B");
        for i in 0..30 {
            std::thread::sleep(Duration::from_millis(50));
            r.progress(total * (i + 1) as f64 / 30.0);
        }
        r.done("64.6 GiB at 1.20 GB/s");
        e.done("slots: GPU0 5.44 GiB, GPU1 15.00 GiB, the rest in pinned RAM");
        engine.done("KV pool 786432 tokens, context 524288");
        event(Level::Info, "http.received", json!({"request_id":"b1aaad814c854bb98d2acd6479ba8e6b","peer":"192.168.10.134:53886","method":"POST",
            "path":"/v1/chat/completions","user_agent":"opencode/2.1.4","content_length":"1379844"}));
        event(Level::Info, "request.prefill", json!({"request_id":"b1aaad814c854bb98d2acd6479ba8e6b","choice":0,"processed_tokens":16384,
            "prompt_tokens":396108,"cached_tokens":0,"progress_pct":4.13,"chunk_tokens":2048,"chunk_ms":667.1,"chunk_tok_s":3069.9,"prefill_tok_s":3192.8,"kv_pages_free":525568}));
        event(Level::Info, "engine.stats", json!({"interval_s":5.02,"running":1,"waiting":0,"waiting_peak":0,"completed":0,"errors":0,"cancelled":0,
            "preempted":1,"kv_full":0,"prefill_tokens":16384,"prefill_tok_s":3255.3,"decode_tok_s":0.0,"output_tokens":0,"output_tok_s_wall":0.0,
            "admitted_prompt_tokens":396108,"cached_tokens":0,"step_mean_ms":630.2,"step_max_ms":702.9,"kv_usage_pct":4.1,"kv_pages_free":525568,
            "kv_pages_total":548000,"cache_entries":2,"spec_tokens_per_verify":null}));
        event(Level::Info, "request.finished", json!({"request_id":"b1aaad814c854bb98d2acd6479ba8e6b","choice":0,"finish_reason":"Stop",
            "prompt_tokens":67514,"cached_tokens":63868,"output_tokens":349,"e2e_ms":4852.9,"prefill_ms":1656.7,"decode_ms":2246.5,
            "prefill_tok_s":2200.8,"decode_tok_s":154.9,"tpot_ms":6.46}));
        event(Level::Info, "http.finished", json!({"request_id":"b1aaad814c854bb98d2acd6479ba8e6b","path":"/v1/chat/completions","status":200,
            "elapsed_ms":8507.3,"response_bytes":3743,"complete":true}));
    }

    #[test]
    fn sampling_shows_what_differs_from_neutral() {
        let p = json!({"temperature":0.75,"top_p":0.949999988079071,"top_k":20,"min_p":0.0,"presence_penalty":0.30000001192092896,
            "frequency_penalty":0.0,"seed":null,"repetition_penalty":1.0,"logit_bias":[],"min_tokens":0,"bad_words":[],"logprobs":null});
        assert_eq!(sampling(&p), "temperature=0.75 top_p=0.95 top_k=20 presence_penalty=0.3");
    }
}
