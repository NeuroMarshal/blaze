//! NVFP4 attention tuning, cached per GPU and query length. Choices stay fixed as context grows
//! so captured CUDA graphs remain valid; BLAZE_TUNE=0 keeps the built-in rules.

use std::ffi::c_int;
use std::path::PathBuf;

/// Bump when the attention kernels change: older entries are measured again.
const VERSION: u32 = 1;

/// The choice for a call of `q_len` rows of one request: the verify mode (blz_mha_set_verify) and its splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Choice {
    pub mode: c_int,
    pub splits: usize,
}

impl Choice {
    pub fn describe(&self) -> &'static str {
        match self.mode {
            0 => "register",
            1 => "shared fp32",
            2 => "shared fp16",
            3 => "pair fp32",
            _ => "pair fp16",
        }
    }
}

/// The cache file of this device (one per rank: the ranks measure in parallel on their own GPUs).
pub fn path(rank: usize) -> Option<PathBuf> {
    let (major, minor) = crate::gpu::compute_capability();
    let sms = crate::gpu::multiprocessors();
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join(".cache/blaze").join(format!("attn-sm{major}{minor}-{sms}sm-r{rank}.json")))
}

/// The entry of a geometry: `key` names heads, format, accumulation and the measured context.
pub fn key(heads: usize, kv_heads: usize, head_dim: usize, format: c_int, f16: bool, decode_f16: bool, ctx: usize) -> String {
    format!("v{VERSION}:h{heads}:kv{kv_heads}:d{head_dim}:f{format}:acc{}{}:ctx{ctx}", f16 as u8, decode_f16 as u8)
}

/// The table of `key` (index q_len; None: the built-in rule), if it covers lengths up to `max_q`.
pub fn load(rank: usize, key: &str, max_q: usize) -> Option<Vec<Option<Choice>>> {
    let text = std::fs::read_to_string(path(rank)?).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&text).ok()?;
    let rows = doc.get(key)?.as_array()?;
    let mut table = vec![None; max_q + 1];
    for r in rows {
        let (Some(q), Some(mode), Some(splits)) = (r[0].as_u64(), r[1].as_i64(), r[2].as_u64()) else { continue };
        if let Some(slot) = table.get_mut(q as usize) {
            *slot = Some(Choice { mode: mode as c_int, splits: splits as usize });
        }
    }
    table[1..].iter().all(Option::is_some).then_some(table)
}

/// Add `table` under `key` to the cache file (other geometries' entries stay).
pub fn store(rank: usize, key: &str, table: &[Option<Choice>]) {
    let Some(p) = path(rank) else { return };
    let mut doc = std::fs::read_to_string(&p)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    let rows: Vec<serde_json::Value> = table
        .iter()
        .enumerate()
        .filter_map(|(q, c)| c.map(|c| serde_json::json!([q, c.mode, c.splits])))
        .collect();
    doc[key] = serde_json::Value::Array(rows);
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(p, serde_json::to_string_pretty(&doc).unwrap_or_default());
}

/// Splits to try around the occupancy rule's `base`: half, the rule, twice (within 1..=64).
pub fn split_candidates(base: usize) -> Vec<usize> {
    let mut c: Vec<usize> = [base / 2, base, base * 2].iter().map(|&s| s.clamp(1, 64)).collect();
    c.dedup();
    c
}
