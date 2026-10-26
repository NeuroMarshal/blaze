//! Serving policy and sampling defaults, independent of the inference backend.
use crate::{logging, sampler::SamplingParams};
use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct ServiceArgs {
    /// Print effective configuration (API keys redacted) and exit without loading a model
    #[arg(long)]
    pub print_config: bool,
    /// Bearer API keys; repeat the flag for rotation. BLAZE_API_KEY also supported
    #[arg(long, env = "BLAZE_API_KEY", hide_env_values = true)]
    pub api_key: Vec<String>,
    /// File with one API key per line (blank lines ignored)
    #[arg(long)]
    pub api_key_file: Option<PathBuf>,
    /// Maximum accepted generations, including queued ones; overload returns HTTP 429
    #[arg(long, default_value_t = 256)]
    pub max_queued_requests: usize,
    /// Maximum concurrent HTTP requests including their streaming bodies
    #[arg(long, default_value_t = 256)]
    pub max_concurrent_requests: usize,
    /// End-to-end request deadline, including queue/prefill/decode; 0 disables it
    #[arg(long, default_value_t = 1800.0)]
    pub request_timeout: f64,
    /// Maximum graceful shutdown drain time after SIGINT/SIGTERM, in seconds
    #[arg(long, default_value_t = 30.0)]
    pub shutdown_timeout: f64,
    /// Maximum request body in MiB
    #[arg(long, default_value_t = 64)]
    pub max_request_body_mib: usize,
    /// Allowed CORS origins, comma separated; '*' allows all, 'none' disables CORS
    #[arg(long, default_value = "*")]
    pub allowed_origins: String,
    /// Error, warn, info (important events), debug (every step), trace (every token)
    #[arg(long, value_enum, default_value = "info")]
    pub log_level: logging::Level,
    #[arg(long, value_enum, default_value = "text")]
    pub log_format: logging::Format,
    #[arg(long, value_enum, default_value = "auto")]
    pub log_color: logging::Color,
    /// Disable per-request lifecycle logs (metrics continue)
    #[arg(long)]
    pub disable_log_requests: bool,
    /// Log bounded request payload previews (control characters are escaped)
    #[arg(long, alias = "enable-log-requests")]
    pub log_inputs: bool,
    /// Log bounded response payload previews
    #[arg(long, alias = "enable-log-outputs")]
    pub log_outputs: bool,
    /// Maximum characters in payload previews
    #[arg(long, default_value_t = 1000)]
    pub max_log_len: usize,
    /// Scheduler summary interval in seconds; 0 disables periodic summaries
    #[arg(long, default_value_t = 5.0, env = "BLAZE_STATS_SECS")]
    pub log_stats_interval: f64,
    /// Per-request progress interval in seconds; 0 only logs start/first-token/end
    #[arg(long, default_value_t = 5.0)]
    pub request_log_interval: f64,
    /// Instrument prefill GPU phases (adds CUDA events); debug mode also enables it
    #[arg(long)]
    pub profile_prefill: bool,
    /// Generation defaults: auto reads model/generation_config.json; none uses Blaze defaults; or a JSON path
    #[arg(long, default_value = "auto")]
    pub generation_config: String,
    /// JSON sampling defaults over generation_config (explicit CLI and request parameters take precedence)
    #[arg(long, alias = "preferred-sampling-params")]
    pub override_generation_config: Option<String>,
    #[arg(long, allow_hyphen_values = true)]
    pub frequency_penalty: Option<f32>,
    #[arg(long)]
    pub repetition_penalty: Option<f32>,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub min_tokens: Option<usize>,
}

impl ServiceArgs {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.max_queued_requests > 0, "--max-queued-requests must be positive");
        ensure!(self.max_concurrent_requests > 0, "--max-concurrent-requests must be positive");
        ensure!(self.max_concurrent_requests <= tokio::sync::Semaphore::MAX_PERMITS, "--max-concurrent-requests is too large");
        ensure!(self.max_log_len <= 1_000_000, "--max-log-len is at most 1000000 characters");
        ensure!(
            self.frequency_penalty.is_none_or(|v| v.is_finite() && (-2.0..=2.0).contains(&v)),
            "--frequency-penalty must be finite and in [-2,2]"
        );
        ensure!(self.repetition_penalty.is_none_or(|v| v.is_finite() && v > 0.), "--repetition-penalty must be finite and positive");
        ensure!((1..=4096).contains(&self.max_request_body_mib), "--max-request-body-mib must be in 1..=4096");
        for (k, x) in [
            ("request-timeout", self.request_timeout),
            ("shutdown-timeout", self.shutdown_timeout),
            ("log-stats-interval", self.log_stats_interval),
            ("request-log-interval", self.request_log_interval),
        ] {
            ensure!(x.is_finite() && (0.0..=31_536_000.0).contains(&x), "--{k} must be finite, nonnegative, and at most one year");
        }
        Ok(())
    }

    pub fn logging(&self) -> logging::Options {
        logging::Options {
            level: self.log_level,
            format: self.log_format,
            color: self.log_color,
            requests: !self.disable_log_requests,
            inputs: self.log_inputs,
            outputs: self.log_outputs,
            max_len: self.max_log_len,
            stats_secs: self.log_stats_interval,
            progress_secs: self.request_log_interval,
            profile_prefill: self.profile_prefill || self.log_level >= logging::Level::Debug,
        }
    }

    pub fn keys(&self) -> Result<Vec<String>> {
        let mut keys = self.api_key.clone();
        if let Some(path) = &self.api_key_file {
            keys.extend(
                std::fs::read_to_string(path)
                    .with_context(|| format!("API key file {}", path.display()))?
                    .lines()
                    .map(str::trim)
                    .filter(|x| !x.is_empty())
                    .map(str::to_owned),
            );
        }
        ensure!(
            keys.iter().all(|k| !k.trim().is_empty() && !k.chars().any(char::is_control)),
            "API keys must be nonempty and contain no control characters"
        );
        keys.sort();
        keys.dedup();
        Ok(keys)
    }

    pub fn sampling(&self, model: &Path, overrides: &Value) -> Result<SamplingParams> {
        let mut defaults = json!({"temperature": 0.8, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.5, "frequency_penalty": 0.0, "repetition_penalty": 1.0, "min_tokens": 0});
        let path = match self.generation_config.as_str() {
            "auto" => Some(model.join("generation_config.json")).filter(|p| p.exists()),
            "none" => None,
            value => {
                let p = PathBuf::from(value);
                Some(if p.is_dir() { p.join("generation_config.json") } else { p })
            }
        };
        if let Some(path) = path {
            let value: Value =
                serde_json::from_str(&std::fs::read_to_string(&path).with_context(|| format!("generation config {}", path.display()))?)?;
            merge_sampling(&mut defaults, &value, false)?;
        }
        if let Some(s) = &self.override_generation_config {
            merge_sampling(&mut defaults, &serde_json::from_str(s).context("--override-generation-config must be a JSON object")?, true)?;
        }
        merge_sampling(&mut defaults, overrides, true)?;
        if let Some(x) = self.frequency_penalty {
            defaults["frequency_penalty"] = json!(x);
        }
        if let Some(x) = self.repetition_penalty {
            defaults["repetition_penalty"] = json!(x);
        }
        if let Some(x) = self.seed {
            defaults["seed"] = json!(x);
        }
        if let Some(x) = self.min_tokens {
            defaults["min_tokens"] = json!(x);
        }
        validate_sampling(&defaults).map_err(anyhow::Error::msg)?;
        Ok(SamplingParams {
            temperature: defaults["temperature"].as_f64().unwrap() as f32,
            top_p: defaults["top_p"].as_f64().unwrap() as f32,
            top_k: defaults["top_k"].as_i64().unwrap() as i32,
            min_p: defaults["min_p"].as_f64().unwrap() as f32,
            presence_penalty: defaults["presence_penalty"].as_f64().unwrap() as f32,
            frequency_penalty: defaults["frequency_penalty"].as_f64().unwrap() as f32,
            repetition_penalty: defaults["repetition_penalty"].as_f64().unwrap() as f32,
            min_tokens: defaults["min_tokens"].as_u64().unwrap() as usize,
            seed: defaults.get("seed").and_then(Value::as_u64),
            ..Default::default()
        })
    }
}

const SAMPLING_KEYS: &[&str] =
    &["temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty", "min_tokens", "seed"];

fn merge_sampling(out: &mut Value, value: &Value, strict: bool) -> Result<()> {
    let obj = value.as_object().context("generation defaults must be a JSON object")?;
    for (k, v) in obj {
        if SAMPLING_KEYS.contains(&k.as_str()) {
            if !v.is_null() {
                out[k] = v.clone();
            }
        } else if strict {
            bail!("unsupported sampling default '{k}'");
        }
    }
    Ok(())
}

/// Used for defaults and requests alike; wrong types must never silently fall back to defaults.
#[derive(Debug)]
pub struct SamplingError {
    pub field: &'static str,
    pub message: String,
}

impl std::fmt::Display for SamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

pub fn validate_sampling(value: &Value) -> std::result::Result<(), SamplingError> {
    let error = |field, message| SamplingError { field, message };
    for k in ["temperature", "top_p", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty"] {
        let Some(v) = value.get(k).filter(|v| !v.is_null()) else { continue };
        let x = v
            .as_f64()
            .filter(|x| x.is_finite() && (*x as f32).is_finite())
            .ok_or_else(|| error(k, format!("{k} must be a finite number")))?;
        let valid = match k {
            "temperature" => x >= 0.,
            "top_p" => x > 0. && x <= 1.,
            "min_p" => (0. ..=1.).contains(&x),
            "repetition_penalty" => x > 0.,
            _ => (-2. ..=2.).contains(&x),
        };
        if !valid {
            return Err(error(k, format!("{k} is outside its supported range")));
        }
    }
    if let Some(v) = value.get("top_k").filter(|v| !v.is_null()) {
        if v.as_i64().is_none_or(|n| n < -1 || n > i32::MAX as i64) {
            return Err(error("top_k", "top_k must be an integer in [-1, 2147483647] (0 and -1 disable the cutoff)".into()));
        }
    }
    for k in ["seed", "min_tokens"] {
        if value.get(k).is_some_and(|v| !v.is_null() && v.as_u64().is_none()) {
            return Err(error(k, format!("{k} must be a nonnegative integer")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(clap::Parser)]
    struct TestCli {
        #[command(flatten)]
        args: ServiceArgs,
    }
    #[test]
    fn invalid_types_never_become_defaults() {
        for v in [json!({"temperature":"cold"}), json!({"top_k":1.2}), json!({"seed":-1}), json!({"min_p":2}), json!({"temperature":1e300})]
        {
            assert!(validate_sampling(&v).is_err(), "{v}");
        }
        assert!(validate_sampling(&json!({"temperature":0, "top_k":-1, "seed":0})).is_ok());
        assert_eq!(validate_sampling(&json!({"temperature":-1})).unwrap_err().field, "temperature");
    }
    #[test]
    fn unsupported_model_metadata_is_ignored_but_overrides_are_strict() {
        let mut out = json!({"temperature":1.0});
        merge_sampling(&mut out, &json!({"temperature":0.7,"eos_token_id":[1,2]}), false).unwrap();
        assert_eq!(out["temperature"], 0.7);
        assert!(merge_sampling(&mut out, &json!({"temperatur":0.7}), true).is_err());
    }
    #[test]
    fn defaults_follow_model_then_json_then_cli_and_preserve_zeroes() {
        use clap::Parser;
        let dir = std::env::temp_dir().join(format!("blaze-generation-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("generation_config.json"), r#"{"temperature":0.6,"top_p":0.9,"presence_penalty":0.2,"eos_token_id":123}"#)
            .unwrap();
        let cli = TestCli::parse_from([
            "test",
            "--override-generation-config",
            r#"{"temperature":0.7,"top_p":0.8}"#,
            "--frequency-penalty",
            "-0.1",
            "--seed",
            "0",
        ]);
        let defaults = cli.args.sampling(&dir, &json!({"temperature":0.0,"presence_penalty":0.0})).unwrap();
        assert_eq!(defaults.temperature, 0.0);
        assert_eq!(defaults.top_p, 0.8);
        assert_eq!(defaults.presence_penalty, 0.0);
        assert_eq!(defaults.frequency_penalty, -0.1);
        assert_eq!(defaults.seed, Some(0));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
