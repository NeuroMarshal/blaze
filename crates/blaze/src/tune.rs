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
    /// the FP16-accumulate NVFP4 GEMV: its best launch where it beats the BF16 kernel's best, else (0, 0)
    Nvfp4F16,
    /// NVFP4 calls of more rows than one tensor-core launch takes (PASS_BUCKETS): (1, 0) passes of the GEMV beat
    /// widening the weights for cuBLAS, (0, 0) they do not
    Nvfp4Pass,
    Mxfp8,
    Bf16,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Kernel::Nvfp4 => "nvfp4",
            Kernel::Nvfp4F16 => "nvfp4f16",
            Kernel::Nvfp4Pass => "nvfp4pass",
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

/// Rows of the pass buckets (Kernel::Nvfp4Pass; a call takes the first bucket >= its rows) and the most rows passes
/// are measured for (beyond: widening + cuBLAS, as a prefill chunk).
pub const PASS_BUCKETS: [usize; 4] = [96, 128, 192, 256];
pub const PASS_MAX_ROWS: usize = 256;

fn pass_bucket(m: usize) -> usize {
    PASS_BUCKETS.iter().position(|&b| b >= m).unwrap_or(PASS_BUCKETS.len() - 1)
}

/// One launch of configuration (a, b): NVFP4 / MXFP8 (split, tiles; NVFP4 tiles + 8: the FP16-accumulate GEMV, its
/// input conversion included, into `x16`), BF16 (variant, columns per block).
#[allow(clippy::too_many_arguments)]
fn launch(w: &Weight, gemm: *mut sys::blz_gemm, x: *const u16, x16: (*mut u16, *mut i32), y: *mut u16, rows: usize, a: c_int, b: c_int,
          s: &Stream) -> c_int {
    let (n, k) = (w.n as c_int, w.k as c_int);
    unsafe {
        match w.kernel {
            Kernel::Nvfp4F16 => {
                let e = sys::blz_nvfp4_f16_rows(x, w.k as i64, rows as c_int, k, x16.0, x16.1, s.0);
                if e != 0 { e } else { sys::blz_nvfp4_gemv_mma_f16_cfg(x16.0, x16.1, w.codes, w.scales, w.alpha, y, w.n as i64, rows as c_int, n, k, a, b, s.0) }
            }
            Kernel::Nvfp4 | Kernel::Nvfp4Pass => sys::blz_nvfp4_gemv_mma_cfg(x, w.k as i64, w.codes, w.scales, w.alpha, y, w.n as i64, rows as c_int, n, k,
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

/// Whether the FP16-accumulate NVFP4 GEMV may run (BLAZE_NVFP4_F16=0: never).
pub fn f16_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BLAZE_NVFP4_F16").map(|v| v != "0").unwrap_or(true))
}

/// The configurations to measure for `rows` rows of a weight of `kernel` with `k` columns.
fn candidates(kernel: Kernel, rows: usize, k: usize) -> Vec<(c_int, c_int)> {
    match kernel {
        Kernel::Nvfp4 | Kernel::Nvfp4F16 | Kernel::Nvfp4Pass | Kernel::Mxfp8 => {
            let mut v = Vec::new();
            for split in [2, 4, 8] {
                for tiles in [1, 2, 4] {
                    let ok = unsafe {
                        match kernel {
                            Kernel::Nvfp4 | Kernel::Nvfp4F16 | Kernel::Nvfp4Pass => sys::blz_nvfp4_mma_cfg_valid(rows as c_int, split, tiles) != 0,
                            _ => sys::blz_mxfp8_mma_cfg_valid(rows as c_int, split, tiles) != 0,
                        }
                    };
                    if ok {
                        v.push((split, tiles));
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
                    Some(&"nvfp4f16") => Kernel::Nvfp4F16,
                    Some(&"nvfp4pass") => Kernel::Nvfp4Pass,
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
        let b = if kernel == Kernel::Nvfp4Pass { pass_bucket(m) } else { bucket(m) };
        self.table.read().unwrap().get(&(kernel, n, k, b)).copied().unwrap_or((0, 0))
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
                Kernel::Nvfp4 | Kernel::Nvfp4F16 | Kernel::Nvfp4Pass | Kernel::Mxfp8 => w.k % 128 == 0,
                Kernel::Bf16 => w.k % 8 == 0 && w.codes as usize % 16 == 0,
            };
            if fits {
                shapes.entry((w.kernel, w.n, w.k)).or_default().push(*w);
                // the FP16-accumulate route of weights it may take, measured against the BF16 kernel's best
                if w.kernel == Kernel::Nvfp4 && w.f16 && f16_enabled() {
                    shapes.entry((Kernel::Nvfp4F16, w.n, w.k)).or_default().push(Weight { kernel: Kernel::Nvfp4F16, ..*w });
                }
            }
        }
        let nvfp4: Vec<((usize, usize), Vec<Weight>)> =
            shapes.iter().filter(|(key, _)| key.0 == Kernel::Nvfp4).map(|(key, v)| ((key.1, key.2), v.clone())).collect();
        let missing: Vec<_> = {
            let t = self.table.read().unwrap();
            shapes.into_iter().filter(|(key, _)| (0..BUCKETS.len()).any(|b| !t.contains_key(&(key.0, key.1, key.2, b)))).collect()
        };
        if missing.is_empty() {
            return self.tune_passes(&nvfp4, gemm, s);
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
        // BF16-kernel shapes first: the FP16 route of a shape is kept only where it beats their best time
        let mut missing = missing;
        missing.sort_by_key(|(key, _)| key.0 == Kernel::Nvfp4F16);
        let mut best_bf: HashMap<(usize, usize, usize), f32> = HashMap::new();
        for ((kernel, n, k), ws) in &missing {
            // at most 16 weights in turn: enough to keep them out of L2, few enough to be quick
            let ws = &ws[..ws.len().min(16)];
            let rounds = (32 / ws.len()).clamp(2, 8);
            for (b, &rows) in BUCKETS.iter().enumerate() {
                let mut best = (f32::INFINITY, 0, 0);
                for (ca, cb) in candidates(*kernel, rows, *k) {
                    if *kernel == Kernel::Nvfp4F16 && rows < 2 {
                        break;
                    }
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
                match *kernel {
                    Kernel::Nvfp4F16 => {
                        // where the BF16 kernel's best is not known (cached from before), measure it now
                        let bf = match best_bf.get(&(*n, *k, b)) {
                            Some(&t) => t,
                            None => {
                                let mut t = f32::INFINITY;
                                let cfg = self.get(Kernel::Nvfp4, *n, *k, rows);
                                let w0: Vec<Weight> = ws.iter().map(|w| Weight { kernel: Kernel::Nvfp4, ..*w }).collect();
                                for w in &w0 {
                                    launch(w, gemm, x.ptr(), x16, y.ptr(), rows, cfg.0, cfg.1, s);
                                }
                                timer.start(s)?;
                                for _ in 0..rounds {
                                    for w in &w0 {
                                        launch(w, gemm, x.ptr(), x16, y.ptr(), rows, cfg.0, cfg.1, s);
                                    }
                                }
                                t = t.min(timer.stop(s)?);
                                t
                            }
                        };
                        let keep = best.1 > 0 && best.0 < bf;
                        found.push(((*kernel, *n, *k, b), if keep { (best.1, best.2) } else { (0, 0) }));
                    }
                    _ => {
                        if *kernel == Kernel::Nvfp4 {
                            best_bf.insert((*n, *k, b), best.0);
                        }
                        if best.1 > 0 {
                            found.push(((*kernel, *n, *k, b), (best.1, best.2)));
                        }
                    }
                }
            }
        }
        let shapes_tuned = missing.len();
        {
            let mut t = self.table.write().unwrap();
            t.extend(found);
        }
        self.save();
        Ok(shapes_tuned + self.tune_passes(&nvfp4, gemm, s)?)
    }

    /// One NVFP4 call of `rows` <= 64 rows as Linear::run makes it with this table (FP16 or FP32 accumulation).
    #[allow(clippy::too_many_arguments)]
    fn mma_rows(&self, w: &Weight, x: *const u16, x16: (*mut u16, *mut i32), y: *mut u16, rows: usize, s: &Stream) -> c_int {
        let (n, k) = (w.n, w.k);
        let f = if w.f16 && f16_enabled() { self.get(Kernel::Nvfp4F16, n, k, rows) } else { (0, 0) };
        if f.0 > 0 {
            launch(&Weight { kernel: Kernel::Nvfp4F16, ..*w }, std::ptr::null_mut(), x, x16, y, rows, f.0, f.1, s)
        } else {
            let c = self.get(Kernel::Nvfp4, n, k, rows);
            launch(&Weight { kernel: Kernel::Nvfp4, ..*w }, std::ptr::null_mut(), x, x16, y, rows, c.0, c.1, s)
        }
    }

    /// Kernel::Nvfp4Pass of the NVFP4 shapes not measured yet: for each pass bucket, passes of up to 64 rows (each with
    /// its own entry) against widening the weights + cuBLAS. Returns the shapes measured.
    fn tune_passes(&self, nvfp4: &[((usize, usize), Vec<Weight>)], gemm: *mut sys::blz_gemm, s: &Stream) -> Result<usize> {
        let missing: Vec<_> = {
            let t = self.table.read().unwrap();
            nvfp4.iter().filter(|((n, k), _)| (0..PASS_BUCKETS.len()).any(|b| !t.contains_key(&(Kernel::Nvfp4Pass, *n, *k, b)))).collect()
        };
        if missing.is_empty() {
            return Ok(0);
        }
        let max_k = missing.iter().map(|((_, k), _)| *k).max().unwrap();
        let max_n = missing.iter().map(|((n, _), _)| *n).max().unwrap();
        let max_nk = missing.iter().map(|((n, k), _)| n * k).max().unwrap();
        let x = DevBuf::zeroed(PASS_MAX_ROWS * max_k * 2, s)?;
        let y = DevBuf::zeroed(PASS_MAX_ROWS * max_n * 2, s)?;
        let wide = DevBuf::zeroed(max_nk * 2, s)?;
        let step = sys::BLZ_NVFP4_MMA_MAX_ROWS;
        let (x16b, shb) = (DevBuf::zeroed(step * max_k * 2, s)?, DevBuf::zeroed(step * 4, s)?);
        let x16 = (x16b.ptr::<u16>(), shb.ptr::<i32>());
        let timer = Timer::new()?;
        let mut found = Vec::new();
        for ((n, k), ws) in &missing {
            let ws = &ws[..ws.len().min(8)];
            let rounds = (16 / ws.len()).clamp(2, 4);
            for (b, &rows) in PASS_BUCKETS.iter().enumerate() {
                let passes = |w: &Weight| {
                    let mut e = 0;
                    for r0 in (0..rows).step_by(step) {
                        let r = step.min(rows - r0);
                        e |= self.mma_rows(w, unsafe { x.ptr::<u16>().add(r0 * k) }, x16, unsafe { y.ptr::<u16>().add(r0 * n) }, r, s);
                    }
                    e
                };
                let widen = |w: &Weight| unsafe {
                    sys::blz_nvfp4_weight_dequant(w.codes, w.scales, w.alpha, *n as c_int, *k as c_int, wide.ptr(), s.0)
                        | sys::blz_gemm_bf16(gemm, x.ptr(), *k as i64, wide.ptr(), y.ptr(), *n as i64, rows as c_int, *n as c_int, *k as c_int, s.0)
                };
                let time = |f: &dyn Fn(&Weight) -> c_int| -> Result<f32> {
                    let mut err = 0;
                    for w in ws {
                        err |= f(w);
                    }
                    if err != 0 {
                        return Ok(f32::INFINITY);
                    }
                    timer.start(s)?;
                    for _ in 0..rounds {
                        for w in ws {
                            f(w);
                        }
                    }
                    timer.stop(s)
                };
                let (tp, tw) = (time(&passes)?, time(&widen)?);
                found.push(((Kernel::Nvfp4Pass, *n, *k, b), if tp < tw { (1, 0) } else { (0, 0) }));
            }
        }
        let tuned = missing.len();
        self.table.write().unwrap().extend(found);
        self.save();
        Ok(tuned)
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
