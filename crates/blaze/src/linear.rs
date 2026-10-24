//! Checkpoint linear layers, grouped by shared input and storage format.
//! Tensor-parallel shards are cut before upload.

use crate::gpu::{DevBuf, Stream, check};
use crate::quant::{Format, QuantConfig, WeightNames};
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Result, bail};
use blaze_sys as sys;
use std::borrow::Cow;
use std::ffi::c_int;
use std::ops::Range;

/// The part of a module this rank keeps.
#[derive(Clone, Debug)]
pub enum Cut {
    All,
    /// Output rows (column parallel), several ranges concatenated in order.
    Rows(Vec<Range<usize>>),
    /// Input columns (row parallel: the outputs are partial sums to all-reduce).
    Cols(Range<usize>),
}

enum Store {
    Bf16(DevBuf),
    /// `f16`: every block scale <= 1, so the FP16-accumulate GEMV keeps its partial sums finite
    Nvfp4 { codes: DevBuf, scales: DevBuf, alpha: DevBuf, f16: bool },
    /// E4M3 codes [n, k], optional FP32 scales per [block_n, 128] block, FP32 multiplier per row.
    Fp8 { codes: DevBuf, block_scale: Option<DevBuf>, block_n: usize, alpha: DevBuf },
    /// MXFP8: E4M3 codes [n, k], E8M0 scales [n, k / 32].
    Mxfp8 { codes: DevBuf, scales: DevBuf },
}

/// How a module is kept on the GPU; consecutive modules of one kind share a part.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Nvfp4,
    Fp8Channel,
    Fp8Block(usize),
    Mxfp8,
    Bf16,
}

struct Part {
    store: Store,
    /// First output column of the part.
    col: usize,
    n: usize,
}

pub struct Linear {
    parts: Vec<Part>,
    pub n: usize,
    pub k: usize,
}

/// What every linear launch needs: the cuBLAS handle and the scratch matrix NVFP4 weights are widened into for
/// inputs longer than the GEMV serves.
pub struct LinCtx {
    pub gemm: *mut sys::blz_gemm,
    pub scratch: *mut u16,
    pub scratch_elems: usize,
    /// launch configurations of the tensor-core GEMVs measured for this device (tune.rs)
    pub tune: std::sync::Arc<crate::tune::Tuning>,
}

/// Rows `rows` of a row-major tensor with `row_bytes` per row, bytes `cols` of each.
fn cut_rows<'a>(raw: &'a [u8], row_bytes: usize, rows: &[Range<usize>], cols: Range<usize>) -> Cow<'a, [u8]> {
    if cols.start == 0 && cols.end == row_bytes && rows.len() == 1 {
        return Cow::Borrowed(&raw[rows[0].start * row_bytes..rows[0].end * row_bytes]);
    }
    let mut v = Vec::with_capacity(rows.iter().map(|r| r.len()).sum::<usize>() * cols.len());
    for r in rows {
        for i in r.clone() {
            v.extend_from_slice(&raw[i * row_bytes + cols.start..i * row_bytes + cols.end]);
        }
    }
    Cow::Owned(v)
}

struct Module {
    name: String,
    format: Format,
    names: WeightNames,
    rows: Vec<Range<usize>>,
    cols: Range<usize>,
    /// Rows kept.
    n: usize,
}

impl Module {
    fn kind(&self) -> Kind {
        match &self.format {
            Format::Nvfp4 { .. } => Kind::Nvfp4,
            Format::Fp8Channel { .. } => Kind::Fp8Channel,
            Format::Fp8Block { block_n, block_k: 128, inverse: false, .. } => Kind::Fp8Block(*block_n),
            Format::Mxfp8 => Kind::Mxfp8,
            _ => Kind::Bf16,
        }
    }
}

fn to_f32(raw: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        Dtype::Bf16 => raw.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dtype::F16 => raw.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        other => bail!("scales in {other:?}"),
    })
}

/// An NVFP4 weight part for the tensor-core GEMV.
pub(crate) struct Nvfp4Ref {
    pub codes: *const u8,
    pub scales: *const u8,
    pub alpha: *const f32,
    pub f16: bool,
    pub n: usize,
    pub k: usize,
}

/// `m` <= BLZ_NVFP4_MMA_MAX_ROWS rows through the tensor-core GEMV the tuner picked for them (FP16 or FP32
/// accumulation); `x16`: the FP16 inputs if an earlier call already made them for the same rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn nvfp4_mma(ctx: &LinCtx, w: &Nvfp4Ref, x: *const u16, ldx: usize, m: usize, o: *mut u16, ldo: usize, s: &Stream,
                        x16: &mut Option<(*const u16, *mut u16, *mut i32)>) -> Result<()> {
    let (n, k) = (w.n, w.k);
    let (s16, t16) = if w.f16 && crate::tune::f16_enabled() { ctx.tune.get(crate::tune::Kernel::Nvfp4F16, n, k, m) } else { (0, 0) };
    if s16 > 0 && m * k + 2 * m + 8 <= ctx.scratch_elems {
        let (xh, sh) = match *x16 {
            Some((key, xh, sh)) if key == x => (xh, sh),
            _ => {
                let v = (ctx.scratch, unsafe { ctx.scratch.add((m * k).next_multiple_of(8)) }.cast::<i32>());
                check(unsafe { sys::blz_nvfp4_f16_rows(x, ldx as i64, m as c_int, k as c_int, v.0, v.1, s.0) }, "f16 rows")?;
                *x16 = Some((x, v.0, v.1));
                v
            }
        };
        return check(
            unsafe {
                sys::blz_nvfp4_gemv_mma_f16_cfg(xh, sh, w.codes, w.scales, w.alpha, o, ldo as i64, m as c_int, n as c_int, k as c_int, s16,
                                                t16, s.0)
            },
            "blz_nvfp4_gemv_mma_f16",
        );
    }
    let (split, tiles) = ctx.tune.get(crate::tune::Kernel::Nvfp4, n, k, m);
    check(
        unsafe {
            sys::blz_nvfp4_gemv_mma_cfg(x, ldx as i64, w.codes, w.scales, w.alpha, o, ldo as i64, m as c_int, n as c_int, k as c_int, split,
                                        tiles, s.0)
        },
        "blz_nvfp4_gemv_mma",
    )
}

impl Linear {
    /// Load `modules` (name without `.weight`, cut) stacked in order.
    pub fn load(ckpt: &Checkpoint, quant: &QuantConfig, modules: &[(String, Cut)], s: &Stream) -> Result<Self> {
        let mut mods = Vec::with_capacity(modules.len());
        let mut k_local = None;
        for (name, cut) in modules {
            let (scheme, names) = quant.resolve(ckpt, name)?;
            let info = ckpt.info(&names.weight)?;
            let full_n = info.shape[0];
            let full_k = match scheme.format {
                Format::Nvfp4 { group, .. } => {
                    if group != 16 {
                        bail!("{name}: NVFP4 group {group} (16 supported)");
                    }
                    info.shape[1] * 2
                }
                _ => info.shape[1..].iter().product(),
            };
            let (rows, cols) = match cut {
                Cut::All => (vec![0..full_n], 0..full_k),
                Cut::Rows(r) => (r.clone(), 0..full_k),
                Cut::Cols(c) => (vec![0..full_n], c.clone()),
            };
            if rows.iter().any(|r| r.end > full_n) || cols.end > full_k {
                bail!("{name}: cut {cut:?} outside [{full_n}, {full_k}]");
            }
            let k = cols.len();
            if *k_local.get_or_insert(k) != k {
                bail!("stacked modules {modules:?} have different input sizes");
            }
            let n = rows.iter().map(|r| r.len()).sum();
            mods.push(Module { name: name.clone(), format: scheme.format, names, rows, cols, n });
        }
        let k = k_local.unwrap();
        let mut parts = Vec::new();
        let mut col = 0;
        let mut i = 0;
        while i < mods.len() {
            let mut j = i + 1;
            while j < mods.len() && mods[j].kind() == mods[i].kind() {
                j += 1;
            }
            let group = &mods[i..j];
            let n: usize = group.iter().map(|m| m.n).sum();
            let store = match mods[i].kind() {
                Kind::Nvfp4 => Self::load_nvfp4(ckpt, group, n, k)?,
                Kind::Fp8Channel | Kind::Fp8Block(_) if k % 128 == 0 => Self::load_fp8(ckpt, group, n, k, mods[i].kind())?,
                Kind::Mxfp8 if k % 128 == 0 => Self::load_mxfp8(ckpt, group, n, k)?,
                _ => Self::load_16bit(ckpt, group, n, k, s)?,
            };
            parts.push(Part { store, col, n });
            col += n;
            i = j;
        }
        Ok(Self { parts, n: col, k })
    }

    fn load_nvfp4(ckpt: &Checkpoint, group: &[Module], n: usize, k: usize) -> Result<Store> {
        if k % 32 != 0 {
            bail!("{}: NVFP4 input size {k} is not a multiple of 32", group[0].name);
        }
        let codes = DevBuf::new(n * k / 2)?;
        let scales = DevBuf::new(n * k / 16)?;
        let mut alpha = Vec::with_capacity(n);
        let mut row = 0;
        let mut f16 = true;
        for m in group {
            let Format::Nvfp4 { global_inverse, .. } = m.format else { unreachable!() };
            if m.cols.start % 32 != 0 {
                bail!("{}: column cut {:?} is not aligned to 32", m.name, m.cols);
            }
            let full_k = ckpt.info(&m.names.weight)?.shape[1] * 2;
            let c = cut_rows(ckpt.bytes(&m.names.weight)?, full_k / 2, &m.rows, m.cols.start / 2..m.cols.end / 2);
            let sc = cut_rows(ckpt.bytes(m.names.scale.as_ref().unwrap())?, full_k / 16, &m.rows, m.cols.start / 16..m.cols.end / 16);
            codes.upload_at(row * k / 2, &c)?;
            scales.upload_at(row * k / 16, &sc)?;
            // positive E4M3 bytes order as their values; 1.0 is 0x38
            f16 &= sc.iter().all(|&b| b & 0x7f <= 0x38);
            let g_name = m.names.global.as_ref().unwrap();
            let g = f32::from_le_bytes(ckpt.bytes(g_name)?[..4].try_into()?);
            let a = if global_inverse { 1.0 / g } else { g };
            alpha.extend(std::iter::repeat_n(a, m.n));
            row += m.n;
        }
        Ok(Store::Nvfp4 { codes, scales, alpha: DevBuf::from_slice(&alpha)?, f16 })
    }

    fn load_fp8(ckpt: &Checkpoint, group: &[Module], n: usize, k: usize, kind: Kind) -> Result<Store> {
        let codes = DevBuf::new(n * k)?;
        let block_n = if let Kind::Fp8Block(bn) = kind { bn } else { 0 };
        let mut alpha = Vec::with_capacity(n);
        let mut blocks = Vec::new();
        let mut row = 0;
        for m in group {
            let info = ckpt.info(&m.names.weight)?;
            let full_k = info.shape[1];
            let raw = cut_rows(ckpt.bytes(&m.names.weight)?, full_k, &m.rows, m.cols.clone());
            codes.upload_at(row * k, &raw)?;
            let si = ckpt.info(m.names.scale.as_ref().unwrap())?;
            let sc = to_f32(ckpt.bytes(m.names.scale.as_ref().unwrap())?, si.dtype)?;
            match m.format {
                Format::Fp8Channel { per_tensor: true } => alpha.extend(std::iter::repeat_n(sc[0], m.n)),
                Format::Fp8Channel { per_tensor: false } => {
                    for r in &m.rows {
                        alpha.extend_from_slice(&sc[r.clone()]);
                    }
                }
                _ => {
                    // [ceil(N / block_n), ceil(K / 128)] scales: whole blocks of rows and columns only
                    if m.rows.iter().any(|r| r.start % block_n != 0 || r.end % block_n != 0) || m.cols.start % 128 != 0 {
                        bail!("{}: FP8 block cut {:?} / {:?} not aligned to the {}x128 blocks", m.name, m.rows, m.cols, block_n);
                    }
                    let kb_full = full_k.div_ceil(128);
                    for r in &m.rows {
                        for br in r.start / block_n..r.end / block_n {
                            blocks.extend_from_slice(&sc[br * kb_full + m.cols.start / 128..br * kb_full + m.cols.end.div_ceil(128)]);
                        }
                    }
                    alpha.extend(std::iter::repeat_n(1.0f32, m.n));
                }
            }
            row += m.n;
        }
        let block_scale = if block_n > 0 { Some(DevBuf::from_slice(&blocks)?) } else { None };
        Ok(Store::Fp8 { codes, block_scale, block_n: block_n.max(1), alpha: DevBuf::from_slice(&alpha)? })
    }

    fn load_mxfp8(ckpt: &Checkpoint, group: &[Module], n: usize, k: usize) -> Result<Store> {
        let codes = DevBuf::new(n * k)?;
        let scales = DevBuf::new(n * k / 32)?;
        let mut row = 0;
        for m in group {
            if m.cols.start % 32 != 0 || m.cols.end % 32 != 0 {
                bail!("{}: MXFP8 column cut {:?} not aligned to the 32-value blocks", m.name, m.cols);
            }
            let full_k = ckpt.info(&m.names.weight)?.shape[1];
            codes.upload_at(row * k, &cut_rows(ckpt.bytes(&m.names.weight)?, full_k, &m.rows, m.cols.clone()))?;
            let sc = cut_rows(ckpt.bytes(m.names.scale.as_ref().unwrap())?, full_k / 32, &m.rows, m.cols.start / 32..m.cols.end / 32);
            scales.upload_at(row * k / 32, &sc)?;
            row += m.n;
        }
        Ok(Store::Mxfp8 { codes, scales })
    }

    fn load_16bit(ckpt: &Checkpoint, group: &[Module], n: usize, k: usize, s: &Stream) -> Result<Store> {
        let w = DevBuf::new(n * k * 2)?;
        let mut row = 0;
        for m in group {
            let info = ckpt.info(&m.names.weight)?;
            let full_k: usize = info.shape[1..].iter().product();
            let dst: *mut u16 = w.at(row * k);
            match &m.format {
                Format::Dense(d @ (Dtype::Bf16 | Dtype::F16 | Dtype::F32)) => {
                    let es = d.size();
                    let raw = cut_rows(ckpt.bytes(&m.names.weight)?, full_k * es, &m.rows, m.cols.start * es..m.cols.end * es);
                    match d {
                        Dtype::Bf16 => w.upload_at(row * k * 2, &raw)?,
                        Dtype::F16 => {
                            let v: Vec<u16> = raw
                                .chunks_exact(2)
                                .map(|c| half::bf16::from_f32(half::f16::from_le_bytes([c[0], c[1]]).to_f32()).to_bits())
                                .collect();
                            w.upload_at(row * k * 2, &v)?
                        }
                        _ => {
                            let v: Vec<u16> =
                                raw.chunks_exact(4).map(|c| half::bf16::from_f32(f32::from_le_bytes(c.try_into().unwrap())).to_bits()).collect();
                            w.upload_at(row * k * 2, &v)?
                        }
                    }
                }
                Format::Fp8Block { block_n, block_k, scale_dtype, inverse } => {
                    if m.rows.len() != 1 || m.rows[0] != (0..info.shape[0]) || m.cols != (0..full_k) {
                        bail!("{}: FP8 block weights cannot be cut for tensor parallelism yet", m.name);
                    }
                    if *inverse {
                        bail!("{}: weight_scale_inv checkpoints are not supported yet", m.name);
                    }
                    let scale_name = m.names.scale.as_ref().unwrap();
                    let (wq, sc) = (DevBuf::from_slice(ckpt.bytes(&m.names.weight)?)?, DevBuf::from_slice(ckpt.bytes(scale_name)?)?);
                    let is_f32 = match scale_dtype {
                        Dtype::F32 => 1,
                        Dtype::Bf16 => 0,
                        other => bail!("{}: fp8 block scales in {other:?}", m.name),
                    };
                    check(
                        unsafe {
                            sys::blz_fp8_block_dequant(wq.ptr(), sc.ptr(), is_f32, m.n as c_int, k as c_int, *block_n as c_int,
                                                       *block_k as c_int, dst, s.0)
                        },
                        "blz_fp8_block_dequant",
                    )?;
                    s.sync()?;
                }
                Format::Fp8Channel { per_tensor } => {
                    if m.rows.len() != 1 || m.rows[0] != (0..info.shape[0]) || m.cols != (0..full_k) {
                        bail!("{}: FP8 weights cannot be cut for tensor parallelism yet", m.name);
                    }
                    let scale_name = m.names.scale.as_ref().unwrap();
                    let si = ckpt.info(scale_name)?;
                    let (wq, sc) = (DevBuf::from_slice(ckpt.bytes(&m.names.weight)?)?, DevBuf::from_slice(ckpt.bytes(scale_name)?)?);
                    let (bn, is_f32) = (if *per_tensor { m.n } else { 1 }, (si.dtype == Dtype::F32) as c_int);
                    check(
                        unsafe { sys::blz_fp8_block_dequant(wq.ptr(), sc.ptr(), is_f32, m.n as c_int, k as c_int, bn as c_int, k as c_int, dst, s.0) },
                        "blz_fp8_block_dequant",
                    )?;
                    s.sync()?;
                }
                other => bail!("{}: {} weights in a 16-bit part", m.name, other.describe()),
            }
            row += m.n;
        }
        Ok(Store::Bf16(w))
    }

    /// The parts the tensor-core GEMVs serve, for the tuner (tune.rs).
    pub fn tune_weights(&self) -> Vec<crate::tune::Weight> {
        let mut v = Vec::new();
        for p in &self.parts {
            match &p.store {
                Store::Nvfp4 { codes, scales, alpha, f16 } => v.push(crate::tune::Weight {
                    f16: *f16,
                    kernel: crate::tune::Kernel::Nvfp4,
                    n: p.n,
                    k: self.k,
                    codes: codes.ptr::<u8>() as *const u8,
                    scales: scales.ptr::<u8>() as *const u8,
                    alpha: alpha.ptr::<f32>() as *const f32,
                }),
                Store::Mxfp8 { codes, scales } => v.push(crate::tune::Weight {
                    f16: false,
                    kernel: crate::tune::Kernel::Mxfp8,
                    n: p.n,
                    k: self.k,
                    codes: codes.ptr::<u8>() as *const u8,
                    scales: scales.ptr::<u8>() as *const u8,
                    alpha: std::ptr::null(),
                }),
                Store::Bf16(w) => v.push(crate::tune::Weight {
                    f16: false,
                    kernel: crate::tune::Kernel::Bf16,
                    n: p.n,
                    k: self.k,
                    codes: w.ptr::<u8>() as *const u8,
                    scales: std::ptr::null(),
                    alpha: std::ptr::null(),
                }),
                _ => {}
            }
        }
        v
    }

    /// Device memory the layer reads, in reading order: (address, bytes).
    pub fn regions(&self) -> Vec<(*const u8, usize)> {
        let mut v = Vec::new();
        for p in &self.parts {
            match &p.store {
                Store::Nvfp4 { codes, scales, .. } => {
                    v.push((codes.ptr::<u8>() as *const u8, codes.bytes()));
                    v.push((scales.ptr::<u8>() as *const u8, scales.bytes()));
                }
                Store::Bf16(w) => v.push((w.ptr::<u8>() as *const u8, w.bytes())),
                Store::Fp8 { codes, .. } => v.push((codes.ptr::<u8>() as *const u8, codes.bytes())),
                Store::Mxfp8 { codes, scales } => {
                    v.push((codes.ptr::<u8>() as *const u8, codes.bytes()));
                    v.push((scales.ptr::<u8>() as *const u8, scales.bytes()));
                }
            }
        }
        v
    }

    /// Elements of the widest NVFP4 part (the scratch an input longer than the GEMV needs).
    pub fn scratch_elems(&self) -> usize {
        self.parts.iter().filter(|p| !matches!(p.store, Store::Bf16(_))).map(|p| p.n * self.k).max().unwrap_or(0)
    }

    /// out[m, 0..n] (row stride ldo) = x[m, 0..k] (row stride ldx) . W^T.
    pub fn run(&self, ctx: &LinCtx, x: *const u16, ldx: usize, m: usize, out: *mut u16, ldo: usize, s: &Stream) -> Result<()> {
        let k = self.k;
        // the FP16 inputs of the FP16-accumulate GEMV, made once per input rows for every part that takes it (in the
        // scratch the prefill's widened weights use; tensor-core calls never widen)
        let mut x16: Option<(*const u16, *mut u16, *mut i32)> = None;
        for p in &self.parts {
            let o = unsafe { out.add(p.col) };
            match &p.store {
                Store::Nvfp4 { codes, scales, alpha, f16 } => {
                    let w = Nvfp4Ref { codes: codes.ptr(), scales: scales.ptr(), alpha: alpha.ptr(), f16: *f16, n: p.n, k };
                    let passes = m > sys::BLZ_NVFP4_MMA_MAX_ROWS
                        && k % 128 == 0
                        && m <= crate::tune::PASS_MAX_ROWS
                        && ctx.tune.get(crate::tune::Kernel::Nvfp4Pass, p.n, k, m).0 == 1;
                    if m <= sys::BLZ_NVFP4_MMA_MAX_ROWS && k % 128 == 0 {
                        nvfp4_mma(ctx, &w, x, ldx, m, o, ldo, s, &mut x16)?;
                    } else if passes {
                        // more rows than one launch takes: passes of the tensor-core GEMV, each reading the weights
                        // once (0.5 byte a value) where widening them reads and writes 4.5 (the tuner measured both)
                        let step = sys::BLZ_NVFP4_MMA_MAX_ROWS;
                        for r0 in (0..m).step_by(step) {
                            let rows = step.min(m - r0);
                            nvfp4_mma(ctx, &w, unsafe { x.add(r0 * ldx) }, ldx, rows, unsafe { o.add(r0 * ldo) }, ldo, s, &mut x16)?;
                        }
                    } else if m <= sys::BLZ_NVFP4_GEMV_MAX_ROWS {
                        check(
                            unsafe {
                                sys::blz_nvfp4_gemv(x, ldx as i64, codes.ptr(), scales.ptr(), alpha.ptr(), o, ldo as i64, m as c_int,
                                                    p.n as c_int, k as c_int, s.0)
                            },
                            "blz_nvfp4_gemv",
                        )?;
                    } else {
                        if p.n * k > ctx.scratch_elems {
                            bail!("NVFP4 scratch of {} elements for a {}x{} part", ctx.scratch_elems, p.n, k);
                        }
                        check(
                            unsafe {
                                sys::blz_nvfp4_weight_dequant(codes.ptr(), scales.ptr(), alpha.ptr(), p.n as c_int, k as c_int,
                                                              ctx.scratch, s.0)
                            },
                            "blz_nvfp4_weight_dequant",
                        )?;
                        check(
                            unsafe {
                                sys::blz_gemm_bf16(ctx.gemm, x, ldx as i64, ctx.scratch, o, ldo as i64, m as c_int, p.n as c_int,
                                                   k as c_int, s.0)
                            },
                            "blz_gemm_bf16",
                        )?;
                    }
                }
                Store::Fp8 { codes, block_scale, block_n, alpha } => {
                    let bs: *const f32 = block_scale.as_ref().map_or(std::ptr::null(), |b| b.ptr::<f32>() as *const f32);
                    if m <= sys::BLZ_FP8_MMA_MAX_ROWS {
                        check(
                            unsafe {
                                sys::blz_fp8_gemv_mma(x, ldx as i64, codes.ptr(), bs, *block_n as c_int, alpha.ptr(), o, ldo as i64,
                                                      m as c_int, p.n as c_int, k as c_int, s.0)
                            },
                            "blz_fp8_gemv_mma",
                        )?;
                    } else {
                        if p.n * k > ctx.scratch_elems {
                            bail!("FP8 scratch of {} elements for a {}x{} part", ctx.scratch_elems, p.n, k);
                        }
                        check(
                            unsafe {
                                sys::blz_fp8_weight_dequant(codes.ptr(), bs, *block_n as c_int, alpha.ptr(), p.n as c_int, k as c_int,
                                                            ctx.scratch, s.0)
                            },
                            "blz_fp8_weight_dequant",
                        )?;
                        check(
                            unsafe {
                                sys::blz_gemm_bf16(ctx.gemm, x, ldx as i64, ctx.scratch, o, ldo as i64, m as c_int, p.n as c_int,
                                                   k as c_int, s.0)
                            },
                            "blz_gemm_bf16",
                        )?;
                    }
                }
                Store::Mxfp8 { codes, scales } => {
                    if m <= sys::BLZ_MXFP8_MMA_MAX_ROWS {
                        let (split, tiles) = ctx.tune.get(crate::tune::Kernel::Mxfp8, p.n, k, m);
                        check(
                            unsafe {
                                sys::blz_mxfp8_gemv_mma_cfg(x, ldx as i64, codes.ptr(), scales.ptr(), o, ldo as i64, m as c_int, p.n as c_int,
                                                            k as c_int, split, tiles, s.0)
                            },
                            "blz_mxfp8_gemv_mma",
                        )?;
                    } else {
                        if p.n * k > ctx.scratch_elems {
                            bail!("MXFP8 scratch of {} elements for a {}x{} part", ctx.scratch_elems, p.n, k);
                        }
                        check(unsafe { sys::blz_mxfp8_weight_dequant(codes.ptr(), scales.ptr(), p.n as c_int, k as c_int, ctx.scratch, s.0) },
                              "blz_mxfp8_weight_dequant")?;
                        check(
                            unsafe {
                                sys::blz_gemm_bf16(ctx.gemm, x, ldx as i64, ctx.scratch, o, ldo as i64, m as c_int, p.n as c_int,
                                                   k as c_int, s.0)
                            },
                            "blz_gemm_bf16",
                        )?;
                    }
                }
                Store::Bf16(w) => {
                    use crate::tune::{BF16_CUBLAS, BF16_MMA, BF16_SKINNY, BF16_WIDE, Kernel, mma_fits, skinny_fits, wide_fits};
                    let aligned = x as usize % 16 == 0;
                    let skinny = aligned && skinny_fits(m, k, ldx);
                    let mma = aligned && mma_fits(m, k, ldx);
                    let wide = wide_fits(m, k, ldx);
                    // the kernel measured fastest on this GPU for the shape and rows (tune.rs); untuned: the wide GEMV for
                    // an LM head of a few rows, the skinny kernel for a small n and a row or two, the tensor-core GEMV up
                    // to 64 rows, cuBLAS otherwise
                    let (variant, cpb) = match ctx.tune.get(Kernel::Bf16, p.n, k, m) {
                        (BF16_SKINNY, c) if skinny => (BF16_SKINNY, c),
                        (BF16_WIDE, _) if wide => (BF16_WIDE, 0),
                        (BF16_MMA, c) if mma => (BF16_MMA, c),
                        (BF16_CUBLAS, _) => (BF16_CUBLAS, 0),
                        _ if wide && p.n > 4096 => (BF16_WIDE, 0),
                        _ if skinny && p.n <= 4096 && m <= 2 => (BF16_SKINNY, 0),
                        _ if mma && p.n <= 4096 => (BF16_MMA, 0),
                        _ => (BF16_CUBLAS, 0),
                    };
                    if variant == BF16_MMA {
                        check(
                            unsafe {
                                sys::blz_bf16_gemv_mma_cfg(x, ldx as i64, w.ptr(), o, ldo as i64, m as c_int, p.n as c_int, k as c_int,
                                                           cpb / 8, cpb % 8, s.0)
                            },
                            "blz_bf16_gemv_mma",
                        )?;
                    } else if variant == BF16_WIDE {
                        check(
                            unsafe {
                                sys::blz_gemv_wide(x.cast(), ldx as i64, w.ptr::<u16>().cast(), o.cast(), ldo as i64, m as c_int,
                                                   p.n as c_int, k as c_int, s.0)
                            },
                            "blz_gemv_wide",
                        )?;
                    } else if variant == BF16_SKINNY {
                        check(
                            unsafe {
                                sys::blz_skinny_gemm_cfg(x.cast(), ldx as i64, w.ptr::<u16>().cast(), o.cast(), ldo as i64, m as c_int,
                                                         p.n as c_int, k as c_int, sys::BLZ_BF16, cpb, s.0)
                            },
                            "blz_skinny_gemm",
                        )?;
                    } else {
                        check(
                            unsafe {
                                sys::blz_gemm_bf16(ctx.gemm, x, ldx as i64, w.ptr(), o, ldo as i64, m as c_int, p.n as c_int, k as c_int,
                                                   s.0)
                            },
                            "blz_gemm_bf16",
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}
