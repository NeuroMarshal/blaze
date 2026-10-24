//! GEMV launch tuning cached per GPU, weight shape and input-row bucket.
//! Measurements cycle through layers so large weights are read from DRAM; BLAZE_TUNE=0 disables tuning.

use crate::gpu::{DevBuf, Stream, Timer};
use anyhow::Result;
use blaze_sys as sys;
use std::collections::HashMap;
use std::ffi::c_int;
use std::path::PathBuf;
use std::sync::RwLock;

/// Representative rows of the buckets (a call of m rows takes the first bucket >= m).
pub const BUCKETS: [usize; 10] = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64];
/// Bump when kernel variants change: older cache files are ignored.
const VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    Nvfp4,
    Mxfp8,
    Bf16,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Kernel::Nvfp4 => "nvfp4",
            Kernel::Mxfp8 => "mxfp8",
            Kernel::Bf16 => "bf16",
        }
    }
}

/// BF16 entries (variant, parameter): the skinny kernel (columns per block), the wide GEMV, cuBLAS, the tensor-core
/// GEMV (split * 8 + tiles).
pub const BF16_SKINNY: c_int = 1;
pub const BF16_WIDE: c_int = 2;
pub const BF16_CUBLAS: c_int = 3;
pub const BF16_MMA: c_int = 4;

/// Whether blz_bf16_gemv_mma_cfg takes a call (the alignment of x and w is checked by the caller).
pub fn mma_fits(rows: usize, k: usize, ldx: usize) -> bool {
    rows <= sys::BLZ_MXFP8_MMA_MAX_ROWS && k % 128 == 0 && ldx % 8 == 0
}

/// Whether blz_skinny_gemm_cfg takes a call (the alignment of x and w is checked by the caller).
pub fn skinny_fits(rows: usize, k: usize, ldx: usize) -> bool {
    rows <= sys::BLZ_SKINNY_EX_MAX_ROWS && k % 8 == 0 && ldx % 8 == 0
}

/// Whether blz_gemv_wide takes a call.
pub fn wide_fits(rows: usize, k: usize, ldx: usize) -> bool {
    rows <= sys::BLZ_GEMV_WIDE_MAX_ROWS && k % 256 == 0 && k <= sys::BLZ_GEMV_WIDE_MAX_K && rows * k * 2 <= sys::BLZ_GEMV_WIDE_MAX_SMEM
        && ldx % 8 == 0
}

/// A weight of one shape to measure: its device buffers (codes, scales, NVFP4 alpha; BF16: the weight in codes).
#[derive(Clone, Copy)]
pub struct Weight {
    pub kernel: Kernel,
    /// NVFP4: the FP16-accumulate GEMV may take it (block scales <= 1)
    pub f16: bool,
    pub n: usize,
    pub k: usize,
    pub codes: *const u8,
    pub scales: *const u8,
    pub alpha: *const f32,
}

fn bucket(m: usize) -> usize {
    BUCKETS.iter().position(|&b| b >= m).unwrap_or(BUCKETS.len() - 1)
}

/// One launch of configuration (a, b): NVFP4 / MXFP8 (split, tiles; NVFP4 tiles + 8: the FP16-accumulate GEMV, its
/// input conversion included, into `x16`), BF16 (variant, columns per block).
#[allow(clippy::too_many_arguments)]
fn launch(w: &Weight, gemm: *mut sys::blz_gemm, x: *const u16, x16: (*mut u16, *mut i32), y: *mut u16, rows: usize, a: c_int, b: c_int,
          s: &Stream) -> c_int {
    let (n, k) = (w.n as c_int, w.k as c_int);
    unsafe {
        if w.kernel == Kernel::Nvfp4 && b > 8 {
            let e = sys::blz_nvfp4_f16_rows(x, w.k as i64, rows as c_int, k, x16.0, x16.1, s.0);
            if e != 0 {
                return e;
            }
            return sys::blz_nvfp4_gemv_mma_f16_cfg(x16.0, x16.1, w.codes, w.scales, w.alpha, y, w.n as i64, rows as c_int, n, k, a, b - 8, s.0);
        }
        match w.kernel {
            Kernel::Nvfp4 => sys::blz_nvfp4_gemv_mma_cfg(x, w.k as i64, w.codes, w.scales, w.alpha, y, w.n as i64, rows as c_int, n, k,
                                                        a, b, s.0),
            Kernel::Mxfp8 => sys::blz_mxfp8_gemv_mma_cfg(x, w.k as i64, w.codes, w.scales, y, w.n as i64, rows as c_int, n, k, a, b, s.0),
            Kernel::Bf16 => match a {
                BF16_SKINNY => sys::blz_skinny_gemm_cfg(x.cast(), w.k as i64, w.codes.cast(), y.cast(), w.n as i64, rows as c_int, n, k,
                                                        sys::BLZ_BF16, b, s.0),
                BF16_WIDE => sys::blz_gemv_wide(x.cast(), w.k as i64, w.codes.cast(), y.cast(), w.n as i64, rows as c_int, n, k, s.0),
                BF16_MMA => sys::blz_bf16_gemv_mma_cfg(x, w.k as i64, w.codes.cast(), y, w.n as i64, rows as c_int, n, k, b / 8, b % 8, s.0),
                _ => sys::blz_gemm_bf16(gemm, x, w.k as i64, w.codes.cast(), y, w.n as i64, rows as c_int, n, k, s.0),
            },
        }
    }
}

/// Whether the FP16-accumulate NVFP4 GEMV is among the candidates (BLAZE_NVFP4_F16=0: never).
fn f16_enabled() -> bool {
    std::env::var("BLAZE_NVFP4_F16").map(|v| v != "0").unwrap_or(true)
}

/// The configurations to measure for `rows` rows of a weight of `kernel` with `k` columns (`f16`: NVFP4 with block
/// scales <= 1, whose FP16-accumulate launches count too).
fn candidates(kernel: Kernel, rows: usize, k: usize, f16: bool) -> Vec<(c_int, c_int)> {
    match kernel {
        Kernel::Nvfp4 | Kernel::Mxfp8 => {
            let mut v = Vec::new();
            for split in [2, 4, 8] {
                for tiles in [1, 2, 4] {
                    let ok = unsafe {
                        match kernel {
                            Kernel::Nvfp4 => sys::blz_nvfp4_mma_cfg_valid(rows as c_int, split, tiles) != 0,
                            _ => sys::blz_mxfp8_mma_cfg_valid(rows as c_int, split, tiles) != 0,
                        }
                    };
                    if ok {
                        v.push((split, tiles));
                        if kernel == Kernel::Nvfp4 && f16 && rows > 1 && f16_enabled() {
                            v.push((split, tiles + 8));
                        }
                    }
                }
            }
            v
        }
        Kernel::Bf16 => {
            let mut v = vec![(BF16_CUBLAS, 0)];
            if skinny_fits(rows, k, k) {
                v.extend([(BF16_SKINNY, 1), (BF16_SKINNY, 2), (BF16_SKINNY, 4)]);
            }
            if wide_fits(rows, k, k) {
                v.push((BF16_WIDE, 0));
            }
            if mma_fits(rows, k, k) {
                for split in [2, 4, 8] {
                    for tiles in [1, 2, 4] {
                        if unsafe { sys::blz_mxfp8_mma_cfg_valid(rows as c_int, split, tiles) } != 0 {
                            v.push((BF16_MMA, split * 8 + tiles));
                        }
                    }
                }
            }
            v
        }
    }
}

type Key = (Kernel, usize, usize, usize);

pub struct Tuning {
    table: RwLock<HashMap<Key, (c_int, c_int)>>,
    path: Option<PathBuf>,
    enabled: bool,
}

impl Tuning {
    /// The cache of this device (`rank`: one file per rank, the ranks tune in parallel).
    pub fn open(rank: usize) -> Self {
        let enabled = std::env::var("BLAZE_TUNE").map(|v| v != "0").unwrap_or(true);
        let (mut major, mut minor, mut sms, mut dev) = (0, 0, 0, 0);
        unsafe {
            unsafe extern "C" {
                fn cudaGetDevice(device: *mut c_int) -> c_int;
            }
            cudaGetDevice(&mut dev);
            sys::cudaDeviceGetAttribute(&mut major, sys::ATTR_COMPUTE_CAPABILITY_MAJOR, dev);
            sys::cudaDeviceGetAttribute(&mut minor, sys::ATTR_COMPUTE_CAPABILITY_MINOR, dev);
            sys::cudaDeviceGetAttribute(&mut sms, sys::ATTR_MULTIPROCESSOR_COUNT, dev);
        }
        let path = std::env::var_os("HOME").map(|h| {
            PathBuf::from(h).join(".cache/blaze").join(format!("tune-sm{major}{minor}-{sms}sm-r{rank}.json"))
        });
        let mut table = HashMap::new();
        if let Some(p) = &path
            && let Ok(text) = std::fs::read_to_string(p)
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
            && v["version"].as_u64() == Some(VERSION as u64)
            && let Some(entries) = v["entries"].as_object()
        {
            for (key, val) in entries {
                let parts: Vec<&str> = key.split(':').collect();
                let kernel = match parts.first() {
                    Some(&"nvfp4") => Kernel::Nvfp4,
                    Some(&"mxfp8") => Kernel::Mxfp8,
                    Some(&"bf16") => Kernel::Bf16,
                    _ => continue,
                };
                let (Some(n), Some(k), Some(b)) = (parts.get(1).and_then(|x| x.parse().ok()), parts.get(2).and_then(|x| x.parse().ok()),
                                                   parts.get(3).and_then(|x| x.parse::<usize>().ok()))
                else {
                    continue;
                };
                let (Some(split), Some(tiles)) = (val[0].as_i64(), val[1].as_i64()) else { continue };
                table.insert((kernel, n, k, b), (split as c_int, tiles as c_int));
            }
        }
        Self { table: RwLock::new(table), path, enabled }
    }

    /// The configuration of a call (0, 0: the kernel's default).
    pub fn get(&self, kernel: Kernel, n: usize, k: usize, m: usize) -> (c_int, c_int) {
        if !self.enabled {
            return (0, 0);
        }
        self.table.read().unwrap().get(&(kernel, n, k, bucket(m))).copied().unwrap_or((0, 0))
    }

    /// Measure the shapes of `weights` not in the table (every bucket, every valid configuration; each timed over all
    /// weights of the shape in turn) and save the table. `gemm`: the cuBLAS handle BF16 candidates use. Returns the
    /// shapes measured.
    pub fn tune(&self, weights: &[Weight], gemm: *mut sys::blz_gemm, s: &Stream) -> Result<usize> {
        if !self.enabled {
            return Ok(0);
        }
        let mut shapes: HashMap<(Kernel, usize, usize), Vec<Weight>> = HashMap::new();
        for w in weights {
            let fits = match w.kernel {
                Kernel::Nvfp4 | Kernel::Mxfp8 => w.k % 128 == 0,
                Kernel::Bf16 => w.k % 8 == 0 && w.codes as usize % 16 == 0,
            };
            if fits {
                shapes.entry((w.kernel, w.n, w.k)).or_default().push(*w);
            }
        }
        let missing: Vec<_> = {
            let t = self.table.read().unwrap();
            shapes.into_iter().filter(|(key, _)| (0..BUCKETS.len()).any(|b| !t.contains_key(&(key.0, key.1, key.2, b)))).collect()
        };
        if missing.is_empty() {
            return Ok(0);
        }
        let max_k = missing.iter().map(|(k, _)| k.2).max().unwrap();
        let max_n = missing.iter().map(|(k, _)| k.1).max().unwrap();
        let rows_max = *BUCKETS.last().unwrap();
        let x = DevBuf::zeroed(rows_max * max_k * 2, s)?;
        let y = DevBuf::zeroed(rows_max * max_n * 2, s)?;
        let (x16, sh) = (DevBuf::zeroed(rows_max * max_k * 2, s)?, DevBuf::zeroed(rows_max * 4, s)?);
        let x16 = (x16.ptr::<u16>(), sh.ptr::<i32>());
        let timer = Timer::new()?;
        let mut found = Vec::new();
        for ((kernel, n, k), ws) in &missing {
            // at most 16 weights in turn: enough to keep them out of L2, few enough to be quick
            let ws = &ws[..ws.len().min(16)];
            let rounds = (32 / ws.len()).clamp(2, 8);
            for (b, &rows) in BUCKETS.iter().enumerate() {
                let mut best = (f32::INFINITY, 0, 0);
                let f16 = ws.iter().all(|w| w.f16);
                for (ca, cb) in candidates(*kernel, rows, *k, f16) {
                    let mut err = 0;
                    for w in ws {
                        err |= launch(w, gemm, x.ptr(), x16, y.ptr(), rows, ca, cb, s);
                    }
                    if err != 0 {
                        continue;
                    }
                    timer.start(s)?;
                    for _ in 0..rounds {
                        for w in ws {
                            launch(w, gemm, x.ptr(), x16, y.ptr(), rows, ca, cb, s);
                        }
                    }
                    let ms = timer.stop(s)?;
                    if ms < best.0 {
                        best = (ms, ca, cb);
                    }
                }
                if best.1 > 0 {
                    found.push(((*kernel, *n, *k, b), (best.1, best.2)));
                }
            }
        }
        let shapes_tuned = missing.len();
        {
            let mut t = self.table.write().unwrap();
            t.extend(found);
        }
        self.save();
        Ok(shapes_tuned)
    }

    fn save(&self) {
        let Some(p) = &self.path else { return };
        let t = self.table.read().unwrap();
        let mut entries = serde_json::Map::new();
        let mut keys: Vec<_> = t.keys().copied().collect();
        keys.sort_by_key(|k| (k.0.name(), k.1, k.2, k.3));
        for key in keys {
            let v = t[&key];
            entries.insert(format!("{}:{}:{}:{}", key.0.name(), key.1, key.2, key.3), serde_json::json!([v.0, v.1]));
        }
        let doc = serde_json::json!({"version": VERSION, "buckets": BUCKETS, "entries": entries});
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(p, serde_json::to_string_pretty(&doc).unwrap_or_default());
    }
}
