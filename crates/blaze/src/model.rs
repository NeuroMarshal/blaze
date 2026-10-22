//! Qwen3.5 / 3.6 / 3.8 model graphs with Gated DeltaNet, softmax attention and dense or MoE MLPs.
//! Projections with a shared input are stacked at load time.

use crate::config::{LayerKind, MlpConfig, ModelConfig, RopeScaling};
use crate::gpu::{DevBuf, Event, HostMapped, Stream, check, stream_after};
use std::cell::Cell;
use crate::linear::{Cut, LinCtx, Linear};
use crate::quant::QuantConfig;
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Context, Result, anyhow, bail};
use blaze_sys as sys;
use std::ffi::{CString, c_int};
use std::ops::Range;
use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::Arc;

/// This rank of a tensor-parallel group: heads, GDN heads, MLP rows and vocabulary are split `world` ways; the
/// row-parallel projections (o_proj, out_proj, down_proj) are summed over the group with the host all-reduce.
pub struct Tp {
    pub rank: usize,
    pub world: usize,
    pub ar: *mut sys::blz_host_ar,
    pub ar_bytes: usize,
}

/// Head counts and sizes this rank computes.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub heads: usize,
    pub kv_heads: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub inter: usize,
    pub vocab: usize,
}

fn part(total: usize, rank: usize, world: usize) -> Range<usize> {
    let n = total / world;
    rank * n..(rank + 1) * n
}

fn load_vec(ckpt: &Checkpoint, name: &str) -> Result<DevBuf> {
    load_rows(ckpt, name, None)
}

/// A bf16 tensor, or rows `rows` of it (first dimension).
fn load_rows(ckpt: &Checkpoint, name: &str, rows: Option<&[Range<usize>]>) -> Result<DevBuf> {
    let t = ckpt.info(name)?;
    if t.dtype != Dtype::Bf16 {
        bail!("{name}: expected bf16, got {:?}", t.dtype);
    }
    let raw = ckpt.bytes(name)?;
    match rows {
        None => DevBuf::from_slice(raw),
        Some(rows) => {
            let row_bytes = raw.len() / t.shape[0];
            let mut v = Vec::new();
            for r in rows {
                v.extend_from_slice(&raw[r.start * row_bytes..r.end * row_bytes]);
            }
            DevBuf::from_slice(&v)
        }
    }
}

pub struct Attn {
    qkv: Linear,
    o: Linear,
    q_norm: DevBuf,
    k_norm: DevBuf,
}

pub struct Gdn {
    in_proj: Linear,
    out: Linear,
    conv_w: DevBuf,
    a_log: DevBuf,
    dt_bias: DevBuf,
    norm: DevBuf,
}

pub enum Mlp {
    Moe { gate: Linear, shared_gu: Linear, shared_down: Linear, has_shared_gate: bool },
    Dense { gu: Linear, down: Linear },
}

pub struct Layer {
    kind: LayerKind,
    input_norm: DevBuf,
    post_norm: DevBuf,
    attn: Option<Attn>,
    gdn: Option<Gdn>,
    mlp: Mlp,
    /// Index among the softmax-attention layers (KV pool) or among the GDN layers (state).
    slot: usize,
}

pub struct MoeEngine(*mut sys::blz_moe_engine);

unsafe impl Send for MoeEngine {}

impl Drop for MoeEngine {
    fn drop(&mut self) {
        unsafe { sys::blz_moe_engine_destroy(self.0) };
    }
}

#[derive(Clone)]
pub struct Options {
    pub max_tokens: usize,
    pub max_chunk: usize,
    pub max_seqs: usize,
    pub moe_threads: usize,
    pub moe_heat: Option<String>,
    pub moe_options: String,
    /// VRAM kept free after the experts' slots (activations of cuBLAS, the allocator, the display).
    pub reserve_gib: f64,
    /// Decode: stream the next linear layer's weights into L2 on a side stream while the main stream waits on the
    /// all-reduce or runs the small attention / GDN kernels (MiB per window, 0: off).
    pub l2_prefetch_mib: usize,
    /// Keep the full logits of every step (tensor parallelism otherwise exchanges only each shard's best token).
    pub full_logits: bool,
}

/// Byte offsets of the step metadata in one device buffer (filled by one host-to-device copy per step).
#[derive(Clone, Copy)]
pub struct MetaLayout {
    pub max_tokens: usize,
    pub max_seqs: usize,
    pub tokens: usize,
    pub positions: usize,
    pub locs: usize,
    pub rt_index: usize,
    pub seq_start: usize,
    pub seq_slots: usize,
    pub req_idx: usize,
    pub kv_lens: usize,
    pub logit_rows: usize,
    pub bytes: usize,
}

impl MetaLayout {
    pub fn new(max_tokens: usize, max_seqs: usize) -> Self {
        let mut o = 0;
        let mut take = |n: usize| {
            let at = o;
            o += n.next_multiple_of(64);
            at
        };
        let tokens = take(max_tokens * 8);
        let positions = take(max_tokens * 8);
        let locs = take(max_tokens * 8);
        let rt_index = take(max_tokens * 8);
        let seq_start = take((max_seqs + 1) * 4);
        let seq_slots = take(max_seqs * 4);
        let req_idx = take(max_seqs * 4);
        let kv_lens = take(max_seqs * 4);
        let logit_rows = take(max_seqs * 4);
        Self { max_tokens, max_seqs, tokens, positions, locs, rt_index, seq_start, seq_slots, req_idx, kv_lens, logit_rows, bytes: o }
    }
}

/// One sequence of a step: `len` tokens at `offset` in the step's token list (decode sequences first, len 1).
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub offset: usize,
    pub len: usize,
}

/// Activations of one forward of up to max_chunk tokens.
struct Work {
    residual: DevBuf,
    normed: DevBuf,
    x: DevBuf,
    proj: DevBuf,
    conv: DevBuf,
    g: DevBuf,
    beta: DevBuf,
    core: DevBuf,
    core_normed: DevBuf,
    q: DevBuf,
    gate: DevBuf,
    attn: DevBuf,
    ids: DevBuf,
    weights: DevBuf,
    moe_out: DevBuf,
    shared: DevBuf,
    shared_act: DevBuf,
    gu: DevBuf,
    partial_o: DevBuf,
    partial_ml: DevBuf,
    gx: DevBuf,
    gres: DevBuf,
    gnormed: DevBuf,
    pairs: DevBuf,
    logits: DevBuf,
    next: DevBuf,
    meta: DevBuf,
}

pub struct Model {
    pub cfg: ModelConfig,
    pub dims: Dims,
    pub tp: Tp,
    layers: Vec<Layer>,
    final_norm: DevBuf,
    lm_head: Linear,
    embed: Arc<HostMapped>,
    moe: Option<MoeEngine>,
    lin: LinCtx,
    _scratch: DevBuf,
    inv_freq: DevBuf,
    rope_scale: f32,
    // KV pools of the softmax layers (NVFP4): [slots, kv_heads, head_dim / 2] codes, [.., head_dim / 16] scales
    kv: Vec<[DevBuf; 4]>,
    kv_global: DevBuf,
    req_to_token: DevBuf,
    // GDN states: conv [seqs, channels, kernel - 1] bf16, recurrent [seqs, v_heads, dk, dv] f32
    conv_state: Vec<DevBuf>,
    rec_state: Vec<DevBuf>,
    work: Work,
    pub meta: MetaLayout,
    score_logits: Option<DevBuf>,
    pub stream: Stream,
    pf_stream: Stream,
    pf_fork: Event,
    pf_join: Event,
    pf_active: Cell<bool>,
    pub opts: Options,
    num_sms: usize,
}

fn inv_freq(cfg: &ModelConfig) -> (Vec<f32>, f32) {
    let dim = cfg.rope.rotary_dim;
    let base = cfg.rope.theta;
    let extrap: Vec<f64> = (0..dim / 2).map(|i| 1.0 / base.powf((2 * i) as f64 / dim as f64)).collect();
    match cfg.rope.scaling {
        RopeScaling::None => (extrap.iter().map(|&f| f as f32).collect(), 1.0),
        RopeScaling::Yarn { factor, original_max, beta_fast, beta_slow } => {
            // Hugging Face _compute_yarn_parameters (truncated correction range).
            let corr = |rot: f64| (dim as f64 * (original_max as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln());
            let low = corr(beta_fast).floor().max(0.0);
            let high = corr(beta_slow).ceil().min(dim as f64 - 1.0);
            let high = if (high - low).abs() < 1e-9 { high + 0.001 } else { high };
            let f: Vec<f32> = (0..dim / 2)
                .map(|i| {
                    let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                    let extrap_factor = 1.0 - ramp;
                    (extrap[i] / factor * (1.0 - extrap_factor) + extrap[i] * extrap_factor) as f32
                })
                .collect();
            let mscale = if factor <= 1.0 { 1.0 } else { 0.1 * factor.ln() + 1.0 };
            (f, mscale as f32)
        }
    }
}

impl Model {
    /// The token embedding of a checkpoint as a pinned host table (shared by every rank; the GPUs read the rows
    /// they need over PCIe instead of keeping 1-2.5 GB in VRAM).
    pub fn load_embedding(dir: &Path) -> Result<HostMapped> {
        let ckpt = Checkpoint::open(dir)?;
        let name = format!("{}embed_tokens.weight", ckpt.text_prefix());
        let et = ckpt.info(&name)?.clone();
        if et.dtype != Dtype::Bf16 {
            bail!("{name}: expected bf16");
        }
        let mut embed = HostMapped::new(et.bytes())?;
        embed.host_slice_mut().copy_from_slice(ckpt.bytes(&name)?);
        Ok(embed)
    }

    pub fn load(dir: &Path, opts: Options, tp: Tp, embed: Arc<HostMapped>) -> Result<Self> {
        let cfg = ModelConfig::load(dir)?;
        let ckpt = Checkpoint::open(dir)?;
        let root: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let hf_quant = std::fs::read_to_string(dir.join("hf_quant_config.json")).ok().and_then(|s| serde_json::from_str(&s).ok());
        let quant = QuantConfig::from_model_config(&root, hf_quant.as_ref())?;
        let stream = Stream::new()?;
        let s = &stream;
        let p = ckpt.text_prefix();
        let (r, w) = (tp.rank, tp.world);
        let inter = match cfg.mlp {
            MlpConfig::Dense { intermediate } => intermediate,
            MlpConfig::Moe { shared_intermediate, .. } => shared_intermediate,
        };
        let (gk, gv) = cfg.gdn.as_ref().map_or((0, 0), |g| (g.k_heads, g.v_heads));
        for (what, n) in [("attention heads", cfg.attn.heads), ("KV heads", cfg.attn.kv_heads), ("GDN key heads", gk),
                          ("GDN value heads", gv), ("MLP rows", inter), ("vocabulary", cfg.vocab)] {
            if n % w != 0 {
                bail!("{what} ({n}) do not split {w} ways");
            }
        }
        if w > 1 && matches!(cfg.mlp, MlpConfig::Moe { .. }) {
            bail!("tensor parallelism of MoE models is not supported yet");
        }
        let dims = Dims {
            heads: cfg.attn.heads / w,
            kv_heads: cfg.attn.kv_heads / w,
            k_heads: gk / w,
            v_heads: gv / w,
            inter: inter / w,
            vocab: cfg.vocab / w,
        };
        let all = |name: String| (name, Cut::All);
        let rows = |name: String, ranges: Vec<Range<usize>>| (name, if w == 1 { Cut::All } else { Cut::Rows(ranges) });
        let cols = |name: String, range: Range<usize>| (name, if w == 1 { Cut::All } else { Cut::Cols(range) });

        let mut layers = Vec::with_capacity(cfg.layers.len());
        let (mut n_full, mut n_gdn) = (0, 0);
        let hd = cfg.attn.head_dim;
        for (i, kind) in cfg.layers.iter().enumerate() {
            let l = format!("{p}layers.{i}.");
            let (attn, gdn, slot) = match kind {
                LayerKind::Full => {
                    let a = format!("{l}self_attn.");
                    // q_proj rows per head: [query D | gate D]
                    let qr = part(cfg.attn.heads, r, w);
                    let kvr = part(cfg.attn.kv_heads, r, w);
                    let qkv = Linear::load(
                        &ckpt,
                        &quant,
                        &[
                            rows(format!("{a}q_proj"), vec![qr.start * 2 * hd..qr.end * 2 * hd]),
                            rows(format!("{a}k_proj"), vec![kvr.start * hd..kvr.end * hd]),
                            rows(format!("{a}v_proj"), vec![kvr.start * hd..kvr.end * hd]),
                        ],
                        s,
                    )?;
                    let attn = Attn {
                        qkv,
                        o: Linear::load(&ckpt, &quant, &[cols(format!("{a}o_proj"), qr.start * hd..qr.end * hd)], s)?,
                        q_norm: load_vec(&ckpt, &format!("{a}q_norm.weight"))?,
                        k_norm: load_vec(&ckpt, &format!("{a}k_norm.weight"))?,
                    };
                    n_full += 1;
                    (Some(attn), None, n_full - 1)
                }
                LayerKind::Linear => {
                    let gc = cfg.gdn.as_ref().unwrap();
                    let g = format!("{l}linear_attn.");
                    let key_dim = gc.k_heads * gc.k_dim;
                    let (kr, vr) = (part(gc.k_heads, r, w), part(gc.v_heads, r, w));
                    // conv channels / in_proj_qkv rows: [q (key heads) | k (key heads) | v (value heads)]
                    let qkv_rows = vec![
                        kr.start * gc.k_dim..kr.end * gc.k_dim,
                        key_dim + kr.start * gc.k_dim..key_dim + kr.end * gc.k_dim,
                        2 * key_dim + vr.start * gc.v_dim..2 * key_dim + vr.end * gc.v_dim,
                    ];
                    let in_proj = Linear::load(
                        &ckpt,
                        &quant,
                        &[
                            rows(format!("{g}in_proj_qkv"), qkv_rows.clone()),
                            rows(format!("{g}in_proj_z"), vec![vr.start * gc.v_dim..vr.end * gc.v_dim]),
                            rows(format!("{g}in_proj_b"), vec![vr.clone()]),
                            rows(format!("{g}in_proj_a"), vec![vr.clone()]),
                        ],
                        s,
                    )?;
                    let cut = |v: Vec<Range<usize>>| if w == 1 { None } else { Some(v) };
                    let gdn = Gdn {
                        in_proj,
                        out: Linear::load(&ckpt, &quant, &[cols(format!("{g}out_proj"), vr.start * gc.v_dim..vr.end * gc.v_dim)], s)?,
                        conv_w: load_rows(&ckpt, &format!("{g}conv1d.weight"), cut(qkv_rows).as_deref())?,
                        a_log: load_rows(&ckpt, &format!("{g}A_log"), cut(vec![vr.clone()]).as_deref())?,
                        dt_bias: load_rows(&ckpt, &format!("{g}dt_bias"), cut(vec![vr.clone()]).as_deref())?,
                        norm: load_vec(&ckpt, &format!("{g}norm.weight"))?,
                    };
                    n_gdn += 1;
                    (None, Some(gdn), n_gdn - 1)
                }
            };
            let m = format!("{l}mlp.");
            let mlp = match &cfg.mlp {
                MlpConfig::Moe { .. } => {
                    let has_shared_gate = ckpt.get(&format!("{m}shared_expert_gate.weight")).is_some();
                    let mut gate_mods = vec![all(format!("{m}gate"))];
                    if has_shared_gate {
                        gate_mods.push(all(format!("{m}shared_expert_gate")));
                    }
                    Mlp::Moe {
                        gate: Linear::load(&ckpt, &quant, &gate_mods, s)?,
                        shared_gu: Linear::load(
                            &ckpt,
                            &quant,
                            &[all(format!("{m}shared_expert.gate_proj")), all(format!("{m}shared_expert.up_proj"))],
                            s,
                        )?,
                        shared_down: Linear::load(&ckpt, &quant, &[all(format!("{m}shared_expert.down_proj"))], s)?,
                        has_shared_gate,
                    }
                }
                MlpConfig::Dense { intermediate } => {
                    let ir = part(*intermediate, r, w);
                    Mlp::Dense {
                        gu: Linear::load(
                            &ckpt,
                            &quant,
                            &[rows(format!("{m}gate_proj"), vec![ir.clone()]), rows(format!("{m}up_proj"), vec![ir.clone()])],
                            s,
                        )?,
                        down: Linear::load(&ckpt, &quant, &[cols(format!("{m}down_proj"), ir)], s)?,
                    }
                }
            };
            layers.push(Layer {
                kind: *kind,
                input_norm: load_vec(&ckpt, &format!("{l}input_layernorm.weight"))?,
                post_norm: load_vec(&ckpt, &format!("{l}post_attention_layernorm.weight"))?,
                attn,
                gdn,
                mlp,
                slot,
            });
        }
        let final_norm = load_vec(&ckpt, &format!("{p}norm.weight"))?;
        let lm_name = if cfg.tie_embeddings { format!("{p}embed_tokens") } else { "lm_head".into() };
        let lm_head = Linear::load(&ckpt, &quant, &[rows(lm_name, vec![part(cfg.vocab, r, w)])], s)?;

        let (freqs, rope_scale) = inv_freq(&cfg);
        let inv_freq = DevBuf::from_slice(&freqs)?;

        // KV pools (NVFP4) for every softmax layer, GDN states for every linear layer, this rank's heads.
        let mut kv = Vec::new();
        for _ in 0..n_full {
            let codes = opts.max_tokens * dims.kv_heads * hd / 2;
            let scales = opts.max_tokens * dims.kv_heads * hd / 16;
            kv.push([DevBuf::zeroed(codes, s)?, DevBuf::zeroed(codes, s)?, DevBuf::zeroed(scales, s)?, DevBuf::zeroed(scales, s)?]);
        }
        let kv_global = DevBuf::from_slice(&[1.0f32])?;
        let req_to_token = DevBuf::zeroed(opts.max_seqs * cfg.max_positions.min(opts.max_tokens) * 4, s)?;
        let (mut conv_state, mut rec_state) = (Vec::new(), Vec::new());
        if let Some(g) = &cfg.gdn {
            let channels = 2 * dims.k_heads * g.k_dim + dims.v_heads * g.v_dim;
            for _ in 0..n_gdn {
                conv_state.push(DevBuf::zeroed(opts.max_seqs * channels * (g.conv_kernel - 1) * 2, s)?);
                rec_state.push(DevBuf::zeroed(opts.max_seqs * dims.v_heads * g.k_dim * g.v_dim * 4, s)?);
            }
        }

        let meta = MetaLayout::new(opts.max_chunk.max(opts.max_seqs), opts.max_seqs);
        let work = Self::alloc_work(&cfg, &dims, &layers, &opts, &meta, s)?;
        // Scratch NVFP4 weights are widened into for prefill GEMMs.
        let mut scratch_elems = lm_head.scratch_elems();
        for l in &layers {
            let mut lins: Vec<&Linear> = Vec::new();
            if let Some(a) = &l.attn {
                lins.extend([&a.qkv, &a.o]);
            }
            if let Some(g) = &l.gdn {
                lins.extend([&g.in_proj, &g.out]);
            }
            match &l.mlp {
                Mlp::Moe { gate, shared_gu, shared_down, .. } => lins.extend([gate, shared_gu, shared_down]),
                Mlp::Dense { gu, down } => lins.extend([gu, down]),
            }
            scratch_elems = scratch_elems.max(lins.iter().map(|l| l.scratch_elems()).max().unwrap_or(0));
        }
        let scratch = DevBuf::new(scratch_elems.max(1) * 2)?;
        let mut gemm = null_mut();
        if unsafe { sys::blz_gemm_create(&mut gemm) } != 0 {
            bail!("cuBLAS init failed");
        }
        s.sync()?;

        let mut num_sms: c_int = 0;
        let mut dev: c_int = 0;
        unsafe {
            unsafe extern "C" {
                fn cudaGetDevice(device: *mut c_int) -> c_int;
            }
            cudaGetDevice(&mut dev);
            sys::cudaDeviceGetAttribute(&mut num_sms, sys::ATTR_MULTIPROCESSOR_COUNT, dev);
        }

        let mut model = Self {
            cfg,
            dims,
            tp,
            layers,
            final_norm,
            lm_head,
            embed,
            moe: None,
            lin: LinCtx { gemm, scratch: scratch.ptr(), scratch_elems },
            _scratch: scratch,
            inv_freq,
            rope_scale,
            kv,
            kv_global,
            req_to_token,
            conv_state,
            rec_state,
            work,
            meta,
            score_logits: None,
            stream,
            pf_stream: Stream::new()?,
            pf_fork: Event::new()?,
            pf_join: Event::new()?,
            pf_active: Cell::new(false),
            opts,
            num_sms: num_sms.max(1) as usize,
        };
        if let MlpConfig::Moe { top_k, .. } = model.cfg.mlp {
            model.moe = Some(model.start_moe(dir, top_k)?);
        }
        Ok(model)
    }

    fn start_moe(&self, dir: &Path, top_k: usize) -> Result<MoeEngine> {
        let (free, _) = crate::gpu::mem_info()?;
        let gib = (free as f64 / (1u64 << 30) as f64 - self.opts.reserve_gib).max(0.0);
        let model_dir = CString::new(dir.to_string_lossy().as_bytes())?;
        let heat = self.opts.moe_heat.as_ref().map(|h| CString::new(h.as_bytes())).transpose()?;
        let options = CString::new(self.opts.moe_options.as_bytes())?;
        let cfg = sys::blz_moe_engine_config {
            model_dir: model_dir.as_ptr(),
            topk: top_k as c_int,
            gpu0: 0,
            gpu1: -1,
            gpu0_gib: gib,
            gpu1_gib: 0.0,
            ram_gib: -1.0,
            cpu_threads: self.opts.moe_threads as c_int,
            cpu_cores: null(),
            heat_path: heat.as_ref().map_or(null(), |h| h.as_ptr()),
            swap_options: null(),
            options: options.as_ptr(),
        };
        let mut engine = null_mut();
        let mut err = vec![0i8; 1024];
        if unsafe { sys::blz_moe_engine_create(&cfg, &mut engine, err.as_mut_ptr().cast(), err.len()) } != 0 {
            let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr().cast()) }.to_string_lossy().into_owned();
            return Err(anyhow!("MoE engine: {msg}"));
        }
        eprintln!("blaze: MoE engine with {:.2} GiB of expert slots in VRAM, the rest in pinned RAM", gib);
        Ok(MoeEngine(engine))
    }

    fn alloc_work(cfg: &ModelConfig, d: &Dims, layers: &[Layer], opts: &Options, meta: &MetaLayout, s: &Stream) -> Result<Work> {
        let t = opts.max_chunk;
        let h = cfg.hidden;
        let mut proj_cols = 0;
        for l in layers {
            if let Some(a) = &l.attn {
                proj_cols = proj_cols.max(a.qkv.n);
            }
            if let Some(g) = &l.gdn {
                proj_cols = proj_cols.max(g.in_proj.n);
            }
            match &l.mlp {
                Mlp::Moe { gate, shared_gu, .. } => proj_cols = proj_cols.max(gate.n).max(shared_gu.n),
                Mlp::Dense { gu, .. } => proj_cols = proj_cols.max(gu.n),
            }
        }
        let (heads, hd) = (d.heads, cfg.attn.head_dim);
        let (conv_ch, vh, vd) = match &cfg.gdn {
            Some(g) => (2 * d.k_heads * g.k_dim + d.v_heads * g.v_dim, d.v_heads, g.v_dim),
            None => (0, 0, 0),
        };
        let topk = match cfg.mlp {
            MlpConfig::Moe { top_k, .. } => top_k,
            _ => 0,
        };
        let inter = d.inter;
        let parts_prefill = unsafe { sys::blz_nvfp4_mha_parts(heads as c_int, d.kv_heads as c_int, t as c_int, 1) } as usize;
        let parts_decode = unsafe { sys::blz_nvfp4_mha_parts(heads as c_int, d.kv_heads as c_int, 1, 64) } as usize;
        let partial_rows = (t * parts_prefill).max(opts.max_seqs * parts_decode);
        let _ = proj_cols;
        let b = |n: usize| DevBuf::zeroed(n, s);
        Ok(Work {
            residual: b(t * h * 2)?,
            normed: b(t * h * 2)?,
            x: b(t * h * 2)?,
            proj: b(t.max(opts.max_seqs) * proj_cols * 2)?,
            conv: b(t * conv_ch * 2)?,
            g: b(t * vh * 4)?,
            beta: b(t * vh * 4)?,
            core: b(t * vh * vd * 2)?,
            core_normed: b(t * vh * vd * 2)?,
            q: b(t * heads * hd * 2)?,
            gate: b(t * heads * hd * 2)?,
            attn: b(t * heads * hd * 2)?,
            ids: b(t * topk.max(1) * 4)?,
            weights: b(t * topk.max(1) * 4)?,
            moe_out: b(t * h * 2)?,
            shared: b(t * h * 2)?,
            shared_act: b(t * inter * 2)?,
            gu: b(t * 2 * inter * 2)?,
            partial_o: b(partial_rows * heads * hd * 4)?,
            partial_ml: b(partial_rows * heads * 2 * 4)?,
            gx: b(opts.max_seqs * h * 2)?,
            gres: b(opts.max_seqs * h * 2)?,
            gnormed: b(opts.max_seqs * h * 2)?,
            pairs: b((opts.max_seqs * 8 * 8).next_multiple_of(16))?,
            logits: b(opts.max_seqs * cfg.vocab * 2)?,
            next: b(opts.max_seqs * 4)?,
            meta: b(meta.bytes)?,
        })
    }

    /// Positions per request row of the KV table.
    pub fn ctx_stride(&self) -> usize {
        self.cfg.max_positions.min(self.opts.max_tokens)
    }

    pub fn meta_ptr<T>(&self, offset: usize) -> *mut T {
        self.work.meta.at::<u8>(offset).cast()
    }

    pub fn meta_buffer(&self) -> &DevBuf {
        &self.work.meta
    }

    /// Zero the GDN state of row `slot` (a new sequence starts from an empty state).
    pub fn reset_state(&self, slot: usize) -> Result<()> {
        let s = &self.stream;
        if let Some(g) = &self.cfg.gdn {
            let d = &self.dims;
            let channels = 2 * d.k_heads * g.k_dim + d.v_heads * g.v_dim;
            let conv_bytes = channels * (g.conv_kernel - 1) * 2;
            let rec_bytes = d.v_heads * g.k_dim * g.v_dim * 4;
            for (c, r) in self.conv_state.iter().zip(&self.rec_state) {
                check(unsafe { sys::cudaMemsetAsync(c.at::<u8>(slot * conv_bytes).cast(), 0, conv_bytes, s.0) }, "memset")?;
                check(unsafe { sys::cudaMemsetAsync(r.at::<u8>(slot * rec_bytes).cast(), 0, rec_bytes, s.0) }, "memset")?;
            }
        }
        Ok(())
    }

    /// Enqueue the forward of a step whose metadata is already in the device buffer (MetaLayout). `spans`: decode
    /// sequences (len 1) first, then prefill chunks. With `logits` the head runs on the last token of every span and
    /// leaves greedy picks in `next`; capturable into a CUDA graph (no host synchronization).
    pub fn forward(&self, spans: &[Span], logits: bool) -> Result<()> {
        let t: usize = spans.iter().map(|s| s.len).sum();
        let n = spans.len();
        if t == 0 || t > self.meta.max_tokens || n > self.meta.max_seqs {
            bail!("step of {t} tokens in {n} sequences exceeds the buffers");
        }
        let s = &self.stream;
        let cfg = &self.cfg;
        let h = cfg.hidden;
        let w = &self.work;
        let m = &self.meta;
        check(
            unsafe {
                sys::blz_set_req_to_token(self.req_to_token.ptr(), self.meta_ptr(m.rt_index), self.meta_ptr(m.locs), t as c_int, s.0)
            },
            "set_req_to_token",
        )?;
        check(
            unsafe {
                sys::blz_embed_gather(self.meta_ptr(m.tokens), t as i64, self.embed.dev_ptr(), cfg.vocab as i64, (h * 2) as i64,
                                      w.residual.ptr(), s.0)
            },
            "embed",
        )?;
        let eps = cfg.rms_eps;
        for (li, layer) in self.layers.iter().enumerate() {
            if li == 0 {
                check(unsafe { sys::blz_rmsnorm(w.residual.ptr(), null_mut(), layer.input_norm.ptr(), eps, t as c_int, h as c_int, 1, w.normed.ptr(), s.0) }, "rmsnorm")?;
            } else {
                self.reduce_norm(&layer.input_norm, t)?;
            }
            match layer.kind {
                LayerKind::Full => self.attention(layer, t, spans)?,
                LayerKind::Linear => self.gdn(layer, t, n)?,
            }
            self.reduce_norm(&layer.post_norm, t)?;
            self.mlp(li, layer, t)?;
        }
        self.all_reduce(w.x.ptr(), t * h)?;
        self.prefetch_join()?;
        if logits {
            // h = x + residual of the last token of every span, normed, through the head, greedy pick.
            let rows: *const i32 = self.meta_ptr(m.logit_rows);
            check(unsafe { sys::blz_gather_rows(w.x.ptr(), h as i64, rows, n as c_int, h as c_int, w.gx.ptr(), s.0) }, "gather")?;
            check(unsafe { sys::blz_gather_rows(w.residual.ptr(), h as i64, rows, n as c_int, h as c_int, w.gres.ptr(), s.0) }, "gather")?;
            check(unsafe { sys::blz_rmsnorm(w.gx.ptr(), w.gres.ptr(), self.final_norm.ptr(), eps, n as c_int, h as c_int, 1, w.gnormed.ptr(), s.0) }, "final norm")?;
            if self.tp.world > 1 && !self.opts.full_logits {
                return self.greedy_sharded(w.gnormed.ptr(), n);
            }
            self.head(w.gnormed.ptr(), n, w.logits.ptr())?;
            check(unsafe { sys::blz_argmax_bf16(w.logits.ptr(), n as c_int, cfg.vocab as c_int, cfg.vocab as i64, w.next.ptr(), s.0) }, "argmax")?;
        }
        Ok(())
    }

    /// x holds this rank's partial sums of a row-parallel projection: residual += sum over the group, normed =
    /// RMSNorm(residual) * (1 + w). One fused kernel for decode-sized steps.
    fn reduce_norm(&self, norm: &DevBuf, t: usize) -> Result<()> {
        let (w, s, h, eps) = (&self.work, &self.stream, self.cfg.hidden, self.cfg.rms_eps);
        if self.tp.world > 1 && t <= 64 {
            return check(
                unsafe {
                    sys::blz_host_ar_rmsnorm(self.tp.ar, w.x.ptr(), w.residual.ptr(), norm.ptr(), eps, t as c_int, h as c_int, 1,
                                             w.normed.ptr(), s.0)
                },
                "all-reduce + rmsnorm",
            );
        }
        self.all_reduce(w.x.ptr(), t * h)?;
        check(unsafe { sys::blz_rmsnorm(w.x.ptr(), w.residual.ptr(), norm.ptr(), eps, t as c_int, h as c_int, 1, w.normed.ptr(), s.0) }, "rmsnorm")
    }

    /// Start pulling up to l2_prefetch_mib of `lin`'s weights into L2 on the side stream (decode steps only); the
    /// side stream joins the main one at the end of the forward.
    fn prefetch(&self, lin: &Linear, t: usize) -> Result<()> {
        let budget = self.opts.l2_prefetch_mib << 20;
        if budget == 0 || t > 16 {
            return Ok(());
        }
        stream_after(&self.pf_stream, &self.stream, &self.pf_fork)?;
        let mut left = budget;
        for (p, bytes) in lin.regions() {
            if left == 0 {
                break;
            }
            let n = bytes.min(left);
            check(unsafe { sys::blz_l2_prefetch(p.cast(), n, self.pf_stream.0) }, "l2 prefetch")?;
            left -= n;
        }
        self.pf_active.set(true);
        Ok(())
    }

    fn prefetch_join(&self) -> Result<()> {
        if self.pf_active.replace(false) {
            stream_after(&self.stream, &self.pf_stream, &self.pf_join)?;
        }
        Ok(())
    }

    /// The first linear layer of layer `li` (what the next layer reads first).
    fn first_linear(&self, li: usize) -> Option<&Linear> {
        let l = self.layers.get(li)?;
        l.attn.as_ref().map(|a| &a.qkv).or(l.gdn.as_ref().map(|g| &g.in_proj))
    }

    /// In-place sum of `elems` bf16 values over the tensor-parallel group (nothing with one rank).
    fn all_reduce(&self, data: *mut u16, elems: usize) -> Result<()> {
        if self.tp.world == 1 {
            return Ok(());
        }
        let bytes = elems * 2;
        if bytes > self.tp.ar_bytes {
            bail!("all-reduce of {bytes} B exceeds the buffer ({} B)", self.tp.ar_bytes);
        }
        check(unsafe { sys::blz_host_ar_all_reduce(self.tp.ar, data.cast(), bytes as i64, sys::BLZ_AR_BF16, self.stream.0) }, "all-reduce")
    }

    /// Logits of `rows` normed hidden states into out [rows, vocab]: this rank's vocabulary slice, then the sum over
    /// the group (the other slices are zero).
    fn head(&self, normed: *const u16, rows: usize, out: *mut u16) -> Result<()> {
        let (v, s) = (self.cfg.vocab, &self.stream);
        if self.tp.world > 1 {
            check(unsafe { sys::cudaMemsetAsync(out.cast(), 0, rows * v * 2, s.0) }, "memset logits")?;
        }
        let col0 = self.tp.rank * self.dims.vocab;
        self.lm_head.run(&self.lin, normed, self.cfg.hidden, rows, unsafe { out.add(col0) }, v, s)?;
        self.all_reduce(out, rows * v)
    }

    /// Greedy picks of `rows` normed states into `next` without the full logits: each rank takes the argmax of its
    /// vocabulary shard, the (value, index) pairs of all ranks meet in a 16-byte-per-row all-reduce.
    fn greedy_sharded(&self, normed: *const u16, rows: usize) -> Result<()> {
        let (w, s, v) = (&self.work, &self.stream, self.dims.vocab);
        let (rank, world) = (self.tp.rank, self.tp.world);
        self.lm_head.run(&self.lin, normed, self.cfg.hidden, rows, w.logits.ptr(), v, s)?;
        let bytes = (world * rows * 8).next_multiple_of(16);
        check(unsafe { sys::cudaMemsetAsync(w.pairs.ptr(), 0, bytes, s.0) }, "memset pairs")?;
        check(
            unsafe {
                sys::blz_argmax_part(w.logits.ptr(), rows as c_int, v as c_int, v as i64, (rank * v) as c_int,
                                     w.pairs.at::<f32>(rank * rows * 2), s.0)
            },
            "argmax part",
        )?;
        check(unsafe { sys::blz_host_ar_all_reduce(self.tp.ar, w.pairs.ptr(), bytes as i64, sys::BLZ_AR_FP32, s.0) }, "all-reduce")?;
        check(unsafe { sys::blz_argmax_merge(w.pairs.ptr(), world as c_int, rows as c_int, w.next.ptr(), s.0) }, "argmax merge")
    }

    /// Greedy picks of the last forward (one per span), after a stream synchronization.
    pub fn next_tokens(&self, n: usize) -> Result<Vec<i32>> {
        self.stream.sync()?;
        self.work.next.download::<i32>(n)
    }

    /// Logits of every position of one sequence (scoring): the head over all rows of the last forward.
    pub fn all_logits(&mut self, t: usize) -> Result<Vec<Vec<f32>>> {
        let v = self.cfg.vocab;
        if self.score_logits.as_ref().is_none_or(|b| b.bytes() < t * v * 2) {
            self.score_logits = Some(DevBuf::new(t * v * 2)?);
        }
        let (w, s) = (&self.work, &self.stream);
        let out = self.score_logits.as_ref().unwrap();
        check(unsafe { sys::blz_rmsnorm(w.x.ptr(), w.residual.ptr(), self.final_norm.ptr(), self.cfg.rms_eps, t as c_int, self.cfg.hidden as c_int, 1, w.normed.ptr(), s.0) }, "final norm")?;
        // in blocks of rows the all-reduce buffer holds
        let block = if self.tp.world == 1 { t } else { (self.tp.ar_bytes / (v * 2)).max(1) };
        let mut r0 = 0;
        while r0 < t {
            let n = block.min(t - r0);
            self.head(w.normed.at::<u16>(r0 * self.cfg.hidden), n, out.at::<u16>(r0 * v))?;
            r0 += n;
        }
        s.sync()?;
        let raw = out.download::<u16>(t * v)?;
        Ok(raw.chunks(v).map(|r| r.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect()).collect())
    }

    pub fn last_logits(&self, row: usize) -> Result<Vec<f32>> {
        self.stream.sync()?;
        let v = self.cfg.vocab;
        let raw = self.work.logits.download::<u16>((row + 1) * v)?;
        Ok(raw[row * v..].iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect())
    }

    fn attention(&self, layer: &Layer, t: usize, spans: &[Span]) -> Result<()> {
        let a = layer.attn.as_ref().unwrap();
        let (cfg, w, s, m) = (&self.cfg, &self.work, &self.stream, &self.meta);
        let (heads, kvh, hd) = (self.dims.heads, self.dims.kv_heads, cfg.attn.head_dim);
        let ld = a.qkv.n;
        a.qkv.run(&self.lin, w.normed.ptr(), cfg.hidden, t, w.proj.ptr(), ld, s)?;
        self.prefetch(&a.o, t)?;
        let k: *mut u16 = w.proj.at(heads * 2 * hd);
        let v: *const u16 = w.proj.at(heads * 2 * hd + kvh * hd);
        check(
            unsafe {
                sys::blz_attn_prep(w.proj.ptr(), ld as i64, k, ld as i64, self.meta_ptr(m.positions), a.q_norm.ptr(), a.k_norm.ptr(),
                                   cfg.rms_eps, 1, t as c_int, heads as c_int, kvh as c_int, hd as c_int,
                                   cfg.rope.rotary_dim as c_int, self.inv_freq.ptr(), self.rope_scale, w.q.ptr(),
                                   w.gate.ptr(), s.0)
            },
            "attn_prep",
        )?;
        let pool = &self.kv[layer.slot];
        check(
            unsafe {
                sys::blz_nvfp4_kv_store(k.cast(), v.cast(), ld as i64, ld as i64, sys::BLZ_BF16, self.meta_ptr::<i64>(m.locs).cast(),
                                        sys::BLZ_I64, t as i64, kvh as c_int, hd as c_int, pool[0].ptr(), pool[1].ptr(),
                                        pool[2].ptr(), pool[3].ptr(), self.kv_global.ptr(), self.kv_global.ptr(), 1, s.0)
            },
            "kv_store",
        )?;
        let view = sys::blz_nvfp4_pool {
            k_data: pool[0].ptr(),
            v_data: pool[1].ptr(),
            k_scale: pool[2].ptr(),
            v_scale: pool[3].ptr(),
            k_g: self.kv_global.ptr(),
            v_g: self.kv_global.ptr(),
            heads: kvh as c_int,
            dim: hd as c_int,
        };
        // Decode spans (len 1, first) in one call; every prefill span on its own (its own q_len).
        let n_decode = spans.iter().take_while(|sp| sp.len == 1).count();
        let mut calls: Vec<(usize, usize, usize, usize)> = Vec::new(); // (first span, spans, q_len, token offset)
        if n_decode > 0 {
            calls.push((0, n_decode, 1, 0));
        }
        for (i, sp) in spans.iter().enumerate().skip(n_decode) {
            calls.push((i, 1, sp.len, sp.offset));
        }
        for (first, count, q_len, off) in calls {
            let splits = if q_len == 1 { (4 * self.num_sms / (count * kvh)).clamp(1, 64) } else { 1 };
            check(
                unsafe {
                    sys::blz_nvfp4_mha_decode(
                        w.q.at::<u16>(off * heads * hd).cast(),
                        (heads * hd) as i64,
                        hd as i64,
                        heads as c_int,
                        view,
                        self.req_to_token.ptr(),
                        self.ctx_stride() as i64,
                        sys::blz_index_ptr { ptr: self.meta_ptr::<i32>(m.req_idx + first * 4).cast(), kind: sys::BLZ_I32 },
                        sys::blz_index_ptr { ptr: self.meta_ptr::<i32>(m.kv_lens + first * 4).cast(), kind: sys::BLZ_I32 },
                        count as c_int,
                        q_len as c_int,
                        splits as c_int,
                        1.0 / (hd as f32).sqrt(),
                        w.partial_o.ptr(),
                        w.partial_ml.ptr(),
                        sys::BLZ_BF16,
                        w.attn.at::<u16>(off * heads * hd).cast(),
                        (heads * hd) as i64,
                        hd as i64,
                        s.0,
                    )
                },
                "mha",
            )?;
        }
        check(unsafe { sys::blz_mul_sigmoid(w.attn.ptr(), w.gate.ptr(), (t * heads * hd) as i64, s.0) }, "gate")?;
        a.o.run(&self.lin, w.attn.ptr(), heads * hd, t, w.x.ptr(), cfg.hidden, s)?;
        self.prefetch_mlp(layer, t)
    }

    fn prefetch_mlp(&self, layer: &Layer, t: usize) -> Result<()> {
        match &layer.mlp {
            Mlp::Dense { gu, .. } => self.prefetch(gu, t),
            Mlp::Moe { gate, .. } => self.prefetch(gate, t),
        }
    }

    fn gdn(&self, layer: &Layer, t: usize, seqs: usize) -> Result<()> {
        let g = layer.gdn.as_ref().unwrap();
        let gc = self.cfg.gdn.as_ref().unwrap();
        let (cfg, w, s) = (&self.cfg, &self.work, &self.stream);
        let ld = g.in_proj.n;
        g.in_proj.run(&self.lin, w.normed.ptr(), cfg.hidden, t, w.proj.ptr(), ld, s)?;
        self.prefetch(&g.out, t)?;
        let (kh, vh) = (self.dims.k_heads, self.dims.v_heads);
        let key_dim = kh * gc.k_dim;
        let channels = 2 * key_dim + vh * gc.v_dim;
        let z: *const u16 = w.proj.at(channels);
        let b: *const u16 = w.proj.at(channels + vh * gc.v_dim);
        let a: *const u16 = w.proj.at(channels + vh * gc.v_dim + vh);
        let m = &self.meta;
        check(
            unsafe {
                sys::blz_gdn_conv(w.proj.ptr(), ld as i64, self.meta_ptr(m.seq_start), self.meta_ptr(m.seq_slots), seqs as c_int, channels as c_int,
                                  gc.conv_kernel as c_int, g.conv_w.ptr(), self.conv_state[layer.slot].ptr(), w.conv.ptr(), s.0)
            },
            "gdn_conv",
        )?;
        check(
            unsafe {
                sys::blz_gdn_gate(a, ld as i64, b, ld as i64, g.a_log.ptr(), g.dt_bias.ptr(), t as c_int, vh as c_int,
                                  w.g.ptr(), w.beta.ptr(), s.0)
            },
            "gdn_gate",
        )?;
        check(
            unsafe {
                sys::blz_gdn_recurrent(w.conv.ptr(), w.g.ptr(), w.beta.ptr(), self.meta_ptr(m.seq_start), self.meta_ptr(m.seq_slots), seqs as c_int, t as c_int,
                                       kh as c_int, vh as c_int, gc.k_dim as c_int, gc.v_dim as c_int,
                                       self.rec_state[layer.slot].ptr(), w.core.ptr(), s.0)
            },
            "gdn_recurrent",
        )?;
        check(
            unsafe {
                sys::blz_gated_rmsnorm(w.core.ptr(), (vh * gc.v_dim) as i64, z, ld as i64, g.norm.ptr(), cfg.rms_eps,
                                       t as c_int, vh as c_int, gc.v_dim as c_int, w.core_normed.ptr(), s.0)
            },
            "gated_rmsnorm",
        )?;
        g.out.run(&self.lin, w.core_normed.ptr(), vh * gc.v_dim, t, w.x.ptr(), cfg.hidden, s)?;
        self.prefetch_mlp(layer, t)
    }

    fn mlp(&self, li: usize, layer: &Layer, t: usize) -> Result<()> {
        let (cfg, w, s) = (&self.cfg, &self.work, &self.stream);
        let h = cfg.hidden;
        match &layer.mlp {
            Mlp::Moe { gate, shared_gu, shared_down, has_shared_gate } => {
                let MlpConfig::Moe { experts, top_k, norm_topk, shared_intermediate, .. } = cfg.mlp else { unreachable!() };
                gate.run(&self.lin, w.normed.ptr(), h, t, w.proj.ptr(), gate.n, s)?;
                check(
                    unsafe {
                        sys::blz_router_topk(w.proj.ptr(), gate.n as i64, t as c_int, experts as c_int, top_k as c_int,
                                             norm_topk as c_int, w.ids.ptr(), w.weights.ptr(), s.0)
                    },
                    "router",
                )?;
                let moe = self.moe.as_ref().ok_or_else(|| anyhow!("MoE engine not started"))?;
                check(
                    unsafe {
                        sys::blz_moe_engine_forward(moe.0, li as c_int, w.normed.ptr(), t as c_int, w.ids.ptr(), w.weights.ptr(),
                                                    w.moe_out.ptr(), s.0)
                    },
                    "moe",
                )?;
                // shared expert
                let gu_ld = shared_gu.n;
                shared_gu.run(&self.lin, w.normed.ptr(), h, t, w.gu.ptr(), gu_ld, s)?;
                let up: *const u16 = w.gu.at(shared_intermediate);
                check(
                    unsafe {
                        sys::blz_silu_mul(w.gu.ptr(), gu_ld as i64, up, gu_ld as i64, t as c_int, shared_intermediate as c_int,
                                          w.shared_act.ptr(), s.0)
                    },
                    "silu_mul",
                )?;
                shared_down.run(&self.lin, w.shared_act.ptr(), shared_intermediate, t, w.shared.ptr(), h, s)?;
                if *has_shared_gate {
                    let logit: *const u16 = w.proj.at(experts);
                    check(
                        unsafe {
                            sys::blz_add_sigmoid_scaled(w.moe_out.ptr(), w.shared.ptr(), logit, gate.n as i64, t as c_int, h as c_int,
                                                        w.x.ptr(), s.0)
                        },
                        "shared gate",
                    )
                } else {
                    check(unsafe { sys::blz_add(w.moe_out.ptr(), w.shared.ptr(), (t * h) as i64, w.x.ptr(), s.0) }, "add")
                }
            }
            Mlp::Dense { gu, down } => {
                let inter = gu.n / 2;
                gu.run(&self.lin, w.normed.ptr(), h, t, w.proj.ptr(), gu.n, s)?;
                let up: *const u16 = w.proj.at(inter);
                check(
                    unsafe { sys::blz_silu_mul(w.proj.ptr(), gu.n as i64, up, gu.n as i64, t as c_int, inter as c_int, w.shared_act.ptr(), s.0) },
                    "silu_mul",
                )?;
                down.run(&self.lin, w.shared_act.ptr(), inter, t, w.x.ptr(), h, s)?;
                if let Some(next) = self.first_linear(li + 1) {
                    self.prefetch(next, t)?;
                }
                Ok(())
            }
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        self.moe = None;
        unsafe {
            sys::blz_gemm_destroy(self.lin.gemm);
            if !self.tp.ar.is_null() {
                sys::blz_host_ar_close(self.tp.ar);
            }
        }
    }
}

pub fn describe_load(path: &Path) -> Result<String> {
    let cfg = ModelConfig::load(path).with_context(|| format!("loading {}", path.display()))?;
    Ok(format!("{} layers, hidden {}", cfg.layers.len(), cfg.hidden))
}
