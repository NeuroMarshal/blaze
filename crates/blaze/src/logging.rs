//! Process-wide structured logging. One write per event; payloads are bounded and escaped.
use clap::ValueEnum;
use serde_json::{Value, json};
use std::ffi::{CStr, c_char};
use std::io::{IsTerminal, Write};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
    }
}

unsafe extern "C" fn cpp_log_callback(message: *const c_char) {
    if message.is_null() {
        return;
    }
    let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    legacy(format_args!("{message}"));
}
pub fn options() -> &'static Options {
    OPTIONS.get_or_init(Options::default)
}
pub fn enabled(level: Level) -> bool {
    level <= options().level
}

pub fn event(level: Level, stage: &str, fields: Value) {
    if !enabled(level) {
        return;
    }
    let opts = options();
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let name = format!("{level:?}").to_uppercase();
    let line = if opts.format == Format::Json {
        json!({"time_unix_ms": t.as_millis() as u64, "level": name, "event": stage, "fields": fields}).to_string()
    } else {
        let color = opts.color == Color::Always
            || (opts.color == Color::Auto && std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none());
        let code = match level {
            Level::Error => "31",
            Level::Warn => "33",
            Level::Info => "32",
            Level::Debug => "36",
            Level::Trace => "35",
        };
        let stamp = t.as_secs() % 86400;
        let prefix = format!("{:02}:{:02}:{:02}.{:03} {name:5} {stage}", stamp / 3600, stamp / 60 % 60, stamp % 60, t.subsec_millis());
        let prefix = if color { format!("\x1b[{code}m{prefix}\x1b[0m") } else { prefix };
        let pairs = fields
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| {
                        let v = match v {
                            Value::String(s) => format!("{:?}", s),
                            _ => v.to_string(),
                        };
                        format!("{k}={v}")
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_else(|| fields.to_string());
        format!("{prefix} {pairs}")
    };
    let _ = writeln!(std::io::stderr().lock(), "{line}");
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
    event(level, "runtime", json!({"message": message}));
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
    #[test]
    fn preview_keeps_utf8_and_bounds_chars() {
        assert_eq!(super::preview("Привет\nмир", 3), "При…");
        assert_eq!(super::preview("你好", 2), "你好");
    }
}
