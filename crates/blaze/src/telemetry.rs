//! Lock-free observations; fixed buckets keep both memory and label cardinality bounded.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

const BOUNDS: [f64; 16] = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1., 2.5, 5., 10., 30., 60., 300., 1800.];

pub struct Histogram {
    counts: [AtomicU64; 17],
    micros: AtomicU64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self { counts: std::array::from_fn(|_| AtomicU64::new(0)), micros: AtomicU64::new(0) }
    }
}

impl Histogram {
    pub fn observe(&self, seconds: f64) {
        if !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        self.counts[BOUNDS.partition_point(|&bound| bound < seconds)].fetch_add(1, Relaxed);
        self.micros.fetch_add((seconds * 1e6) as u64, Relaxed);
    }

    pub fn render(&self, name: &str, help: &str, out: &mut String) {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
        let mut total = 0;
        for (i, count) in self.counts.iter().enumerate() {
            total += count.load(Relaxed);
            let bound = BOUNDS.get(i).map_or_else(|| "+Inf".to_string(), ToString::to_string);
            out.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {total}\n"));
        }
        out.push_str(&format!("{name}_count {total}\n{name}_sum {}\n", self.micros.load(Relaxed) as f64 / 1e6));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn buckets_are_cumulative_with_inclusive_bounds() {
        let h = super::Histogram::default();
        for x in [0.001, 0.1, 2000., f64::NAN, -1.] {
            h.observe(x);
        }
        let mut s = String::new();
        h.render("latency", "Latency", &mut s);
        assert!(s.contains("latency_bucket{le=\"0.001\"} 1\n"));
        assert!(s.contains("latency_bucket{le=\"0.1\"} 2\n"));
        assert!(s.contains("latency_bucket{le=\"+Inf\"} 3\nlatency_count 3\n"));
    }
}
