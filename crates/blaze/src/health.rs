//! Scheduler health: actual prefill work, decode throughput, KV pressure and queue activity.
//! The default interval is five seconds; phase-specific baselines detect latency outliers.

use std::time::{Duration, Instant};

/// A step this many times the typical one, and at least SLOW_STEP_SECS, is logged.
const SLOW_STEP_FACTOR: f64 = 30.0;
const SLOW_STEP_SECS: f64 = 5.0;
/// Most slow-step lines per interval.
const SLOW_STEP_LINES: u32 = 5;

#[derive(Default, Clone, Copy)]
pub struct Interval {
    pub requests: u64,
    pub errors: u64,
    pub cancelled: u64,
    pub lengths: u64,
    pub kv_full: u64,
    pub preempted: u64,
    pub tokens: u64,
    pub prompt: u64,
    pub cached: u64,
    pub prefill_work: u64,
    pub prefill_secs: f64,
    pub decode_tokens: u64,
    pub decode_secs: f64,
    steps: u64,
    step_secs: f64,
    step_max: f64,
    waiting_max: usize,
}

/// What a step held, for the slow-step and step-failure lines.
#[derive(Clone, Copy)]
pub struct StepShape {
    pub decoding: usize,
    pub prefilling: usize,
    /// the first prefilling request: tokens read, prompt length
    pub prefill: Option<(usize, usize)>,
    pub spec: bool,
}

impl std::fmt::Display for StepShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} decoding, {} prefilling", self.decoding, self.prefilling)?;
        if let Some((fed, len)) = self.prefill {
            write!(f, " (the next at {fed} of {len} prompt tokens)")?;
        }
        if self.spec {
            write!(f, ", speculation on")?;
        }
        Ok(())
    }
}

pub struct Health {
    every: Option<Duration>,
    pub req_log: bool,
    since: Instant,
    pub cur: Interval,
    spec0: (usize, usize),
    /// decayed mean step time (s)
    mean: [f64;2],
    slow_lines: u32,
    next_id: u64,
}

impl Health {
    pub fn from_env() -> Self {
        let secs = crate::logging::options().stats_secs;
        Self {
            every: (secs > 0.0).then(|| Duration::from_secs_f64(secs)),
            req_log: crate::logging::options().requests,
            since: Instant::now(),
            cur: Interval::default(),
            spec0: (0, 0),
            mean: [0.0;2],
            slow_lines: 0,
            next_id: 0,
        }
    }

    pub fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// A step of `secs` that held `shape`.
    pub fn step(&mut self, secs: f64, shape: StepShape) {
        let c = &mut self.cur;
        c.steps += 1;
        c.step_secs += secs;
        c.step_max = c.step_max.max(secs);
        let mean=&mut self.mean[usize::from(shape.prefilling>0)];
        if *mean > 0.0 && secs > SLOW_STEP_SECS && secs > SLOW_STEP_FACTOR * *mean && self.slow_lines < SLOW_STEP_LINES {
            self.slow_lines += 1;
            crate::logging::event(crate::logging::Level::Warn,"scheduler.latency_outlier",serde_json::json!({"elapsed_ms":secs*1e3,"phase_mean_ms":*mean*1e3,"shape":shape.to_string()}));
        }
        *mean = if *mean == 0.0 { secs } else { 0.98 * *mean + 0.02 * secs };
    }

    /// End of a scheduler round: the interval's line when it is due and anything happened.
    pub fn tick(&mut self, running: usize, waiting: usize, kv_pages: (usize, usize), snapshots: usize, spec: (usize, usize)) {
        self.cur.waiting_max = self.cur.waiting_max.max(waiting);
        let Some(every) = self.every else { return };
        let dt = self.since.elapsed();
        if dt < every {
            return;
        }
        let c = std::mem::take(&mut self.cur);
        let (verifies, emitted) = (spec.0 - self.spec0.0, spec.1 - self.spec0.1);
        self.spec0 = spec;
        self.since = Instant::now();
        self.slow_lines = 0;
        if c.steps == 0 && c.requests == 0 && c.errors == 0 && running == 0 && waiting == 0 {
            return;
        }
        let secs = dt.as_secs_f64();
        crate::logging::event(crate::logging::Level::Info, "engine.stats", serde_json::json!({
            "interval_s":secs,"running":running,"waiting":waiting,"waiting_peak":c.waiting_max,
            "completed":c.requests,"errors":c.errors,"cancelled":c.cancelled,"preempted":c.preempted,"kv_full":c.kv_full,
            "prefill_tokens":c.prefill_work,"prefill_tok_s":c.prefill_work as f64/c.prefill_secs.max(1e-9),
            "decode_tok_s":c.decode_tokens as f64/c.decode_secs.max(1e-9),
            "output_tokens":c.tokens,"output_tok_s_wall":c.tokens as f64/secs,
            "admitted_prompt_tokens":c.prompt,"cached_tokens":c.cached,
            "step_mean_ms":1e3*c.step_secs/c.steps.max(1) as f64,"step_max_ms":c.step_max*1e3,
            "kv_usage_pct":100.0*(kv_pages.1.saturating_sub(kv_pages.0)) as f64/kv_pages.1.max(1) as f64,
            "kv_pages_free":kv_pages.0,"kv_pages_total":kv_pages.1,"cache_entries":snapshots,
            "spec_tokens_per_verify":if verifies>0 {Some(emitted as f64/verifies as f64)}else{None}
        }));
    }
}
