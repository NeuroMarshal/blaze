//! The scheduler's health log on stderr: every BLAZE_STATS_SECS seconds (default 60, 0: off) a line with what the
//! interval did, when it did anything; a slow step with what it held; with BLAZE_REQ_LOG=1 a line per request when it
//! is admitted and when it finishes.

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
    pub tokens: u64,
    pub prompt: u64,
    pub cached: u64,
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
    mean: f64,
    slow_lines: u32,
    next_id: u64,
}

impl Health {
    pub fn from_env() -> Self {
        let secs = std::env::var("BLAZE_STATS_SECS").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(60);
        Self {
            every: (secs > 0).then(|| Duration::from_secs(secs)),
            req_log: std::env::var_os("BLAZE_REQ_LOG").is_some_and(|v| v != "0"),
            since: Instant::now(),
            cur: Interval::default(),
            spec0: (0, 0),
            mean: 0.0,
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
        if self.mean > 0.0 && secs > SLOW_STEP_SECS && secs > SLOW_STEP_FACTOR * self.mean && self.slow_lines < SLOW_STEP_LINES {
            self.slow_lines += 1;
            eprintln!("blaze: slow step of {secs:.2} s (typical {:.1} ms): {shape}", self.mean * 1e3);
        }
        self.mean = if self.mean == 0.0 { secs } else { 0.98 * self.mean + 0.02 * secs };
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
        let mut line = format!(
            "blaze stats: {secs:.0} s: {} requests done ({} errors, {} cancelled, {} at length, {} KV full), {} tokens out \
             ({:.1} tok/s), prompts {} tokens ({:.0}% cached), {} steps (mean {:.1} ms, max {:.0} ms), KV {} of {} pages \
             free, {snapshots} prompt-cache snapshots, running {running}, waiting {waiting} (max {})",
            c.requests,
            c.errors,
            c.cancelled,
            c.lengths,
            c.kv_full,
            c.tokens,
            c.tokens as f64 / secs,
            c.prompt,
            100.0 * c.cached as f64 / c.prompt.max(1) as f64,
            c.steps,
            1e3 * c.step_secs / c.steps.max(1) as f64,
            1e3 * c.step_max,
            kv_pages.0,
            kv_pages.1,
            c.waiting_max,
        );
        if verifies > 0 {
            line += &format!(", speculation {:.2} tokens a verify", emitted as f64 / verifies as f64);
        }
        eprintln!("{line}");
    }
}
