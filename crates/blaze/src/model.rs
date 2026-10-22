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

fn load_hc(ckpt: &Checkpoint, prefix: &str) -> Result<HcBlock> {
    let inject = format!("{prefix}block_inject_weight.weight");
    Ok(HcBlock {
        norm: load_vec(ckpt, &format!("{prefix}hc_norm.weight"))?,
        down: load_vec(ckpt, &format!("{prefix}input_mix_weight_down.weight"))?,
        up: load_vec(ckpt, &format!("{prefix}input_mix_weight_up.weight"))?,
        inject: if ckpt.get(&inject).is_some() { Some(load_vec(ckpt, &inject)?) } else { None },
    })
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
    /// QSA indexer: its projection is the last module of `qkv`; Gemma norms of the index query and keys
    index: Option<(DevBuf, DevBuf)>,
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

/// Low-rank gated hyper-connection of one block (Qwen4-Exp GatedResidual): per-stream norm, mix down to one input,
/// gated injection of the block's output back into every stream.
pub struct HcBlock {
    norm: DevBuf,
    down: DevBuf,
    up: DevBuf,
    inject: Option<DevBuf>,
}

/// How a layer reads and updates the residual stream.
pub enum Mixing {
    /// Pre-norm residual: RMSNorm before each block, residual add after.
    Norm { input: DevBuf, post: DevBuf },
    /// Hyper-connection streams.
    Hyper { attn: HcBlock, mlp: HcBlock },
}

pub enum FinalMix {
    Norm(DevBuf),
    Hyper(HcBlock),
}

/// Per-layer n-gram embedding (Qwen4-Exp PLE) of one layer.
pub struct Ple {
    key: Linear,
    value: Linear,
    norm_key: DevBuf,
    norm_query: DevBuf,
    norm_conv: DevBuf,
    conv_w: DevBuf,
    mult: DevBuf,
    sizes: DevBuf,
    offsets: DevBuf,
    /// host copies of the hash parameters (row ids of upcoming prefill chunks, for read-ahead)
    host_hash: (Vec<i64>, Vec<i64>, Vec<i64>),
}

pub struct PleTable(*mut sys::blz_ple);

unsafe impl Send for PleTable {}

impl Drop for PleTable {
    fn drop(&mut self) {
        unsafe { sys::blz_ple_destroy(self.0) };
    }
}

pub struct Layer {
    kind: LayerKind,
    mix: Mixing,
    ple: Option<Ple>,
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
    /// Expert slots on the second GPU (GiB; 0: none), pinned RAM for experts (GiB; < 0: every expert not in VRAM,
    /// else the rarest go to the SSD), CPU cores of the expert workers ("3-4,9-17"; empty: any).
    pub moe_gpu1_gib: f64,
    pub moe_ram_gib: f64,
    pub moe_cores: String,
    /// Decode: stream the next linear layer's weights into L2 on a side stream while the main stream waits on the
    /// all-reduce or runs the small attention / GDN kernels (MiB per window, 0: off).
    pub l2_prefetch_mib: usize,
    /// Keep the full logits of every step (tensor parallelism otherwise exchanges only each shard's best token).
    pub full_logits: bool,
    /// Extra recurrent-state rows (GDN, PLE) for prefix-cache snapshots, after the max_seqs rows of live sequences.
    pub cache_rows: usize,
    /// CPU threads reading PLE n-gram rows from the checkpoint (random reads: more threads, more reads in flight)
    pub ple_threads: usize,
    /// GiB of RAM for the PLE row cache
    pub ple_cache_gib: f64,
    /// Layers whose input hidden state (the residual stream entering the layer) every forward copies out, in order
    /// (a speculative drafter's features).
    pub capture_layers: Vec<usize>,
    /// Most rows of a speculative verify per sequence (GDN inputs are recorded for the replay of the accepted rows;
    /// every live sequence may verify in the same step); 0: none.
    pub max_verify: usize,
    pub kv_dtype: KvDtype,
    /// EAGLE-4 drafter checkpoint (run on every rank; the layers it reads are captured)
    pub draft: Option<std::path::PathBuf>,
    /// CUDA graphs of the drafter's chain
    pub draft_graphs: bool,
    /// YaRN context extension (Qwen's recipe) by this factor over max_position_embeddings, when the config has no
    /// rope scaling of its own (the checkpoint stays as is)
    pub yarn_factor: Option<f64>,
}

/// KV cache format of the softmax layers (vLLM's --kv-cache-dtype).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvDtype {
    /// NVFP4 with the block scale searched per block (max/6, max/4 and E4M3 steps around: four over six)
    #[default]
    Nvfp4FourOverSix,
    /// NVFP4 with the block maximum mapped to 6
    Nvfp4,
    /// FP8 E4M3, per-layer scale 1
    Fp8,
    Bf16,
}

impl Options {
    /// Rows of the largest verify step: max_verify rows for every live sequence.
    pub fn verify_rows(&self) -> usize {
        self.max_verify * self.max_seqs.max(1)
    }
}

impl std::str::FromStr for KvDtype {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "nvfp4_4over6" | "nvfp4-4over6" => Self::Nvfp4FourOverSix,
            "nvfp4" => Self::Nvfp4,
            "fp8" | "fp8_e4m3" => Self::Fp8,
            "bf16" | "auto" | "bfloat16" => Self::Bf16,
            _ => bail!("unknown KV cache dtype {s} (nvfp4_4over6, nvfp4, fp8, bf16)"),
        })
    }
}

impl KvDtype {
    fn format(self) -> c_int {
        match self {
            Self::Nvfp4FourOverSix | Self::Nvfp4 => sys::BLZ_KV_NVFP4,
            Self::Fp8 => sys::BLZ_KV_FP8,
            Self::Bf16 => sys::BLZ_KV_BF16,
        }
    }

    /// Bytes of codes and of block scales per value.
    fn bytes(self) -> (f64, f64) {
        match self {
            Self::Nvfp4FourOverSix | Self::Nvfp4 => (0.5, 1.0 / 16.0),
            Self::Fp8 => (1.0, 0.0),
            Self::Bf16 => (2.0, 0.0),
        }
    }
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
    /// per token: its sequence's KV-table row and length after the step (QSA)
    pub tok_row: usize,
    pub tok_kv_len: usize,
    // sampling: logit edits of the sampled rows (ids, (a, b, c) per entry, offsets [rows + 1]); grammar masks
    // ([max_seqs][mask_words]) and the mask row of each logit row (-1: none)
    pub max_pen: usize,
    pub pen_off: usize,
    pub mask_row: usize,
    /// bytes copied every step; the lists below go up as far as filled
    pub base_bytes: usize,
    pub pen_ids: usize,
    pub pen_coef: usize,
    pub mask_words: usize,
    pub mask: usize,
    pub bytes: usize,
}

impl MetaLayout {
    /// `head_rows`: most logit rows of a step (a verify step: every row of every sequence).
    pub fn new(max_tokens: usize, max_seqs: usize, head_rows: usize, vocab: usize) -> Self {
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
        let logit_rows = take(max_tokens * 4);
        let tok_row = take(max_tokens * 4);
        let tok_kv_len = take(max_tokens * 4);
        // a sampled row's edits: its penalized tokens (the output's distinct tokens, the prompt's with a repetition
        // penalty), bans, biases
        let max_pen = (head_rows * 32768).clamp(1 << 16, 1 << 21);
        let pen_off = take((max_tokens + 1) * 4);
        let mask_row = take(max_tokens * 4);
        let base_bytes = take(0);
        let pen_ids = take(max_pen * 4);
        let pen_coef = take(max_pen * 12);
        let mask_words = vocab.div_ceil(32);
        let mask = take(max_seqs * mask_words * 4);
        Self {
            max_tokens,
            max_seqs,
            tokens,
            positions,
            locs,
            rt_index,
            seq_start,
            seq_slots,
            req_idx,
            kv_lens,
            logit_rows,
            tok_row,
            tok_kv_len,
            max_pen,
            pen_off,
            mask_row,
            base_bytes,
            pen_ids,
            pen_coef,
            mask_words,
            mask,
            bytes: o,
        }
    }
}

/// QSA: query rows scored per pass (the scores scratch holds this many rows of the longest context's blocks), and the
/// row stride of the selection table (block_topk * ratio + ratio - 1 rounded up).
const QSA_SCORE_ROWS: usize = 128;
const QSA_TABLE_STRIDE: usize = 2056;

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
    // hyper-connection streams [t, hc * hidden], their norm, mix workspace; head rows of the streams
    streams: Option<DevBuf>,
    hnorm: Option<DevBuf>,
    hc_t: Option<DevBuf>,
    hc_ws: Option<DevBuf>,
    gstreams: Option<DevBuf>,
    /// gate partial sums of the split combine (decode rows)
    hc_partial: Option<DevBuf>,
    ghnorm: Option<DevBuf>,
    // PLE: n-gram row ids [t, heads], embeddings [t, embed], key / gated / gated-normed [t, hc * hidden], value
    ple_ids: Option<DevBuf>,
    ple_emb: Option<DevBuf>,
    ple_key: Option<DevBuf>,
    ple_value: Option<DevBuf>,
    ple_gated: Option<DevBuf>,
    ple_gatedn: Option<DevBuf>,
    pairs: DevBuf,
    // sampling candidates of this rank's vocabulary shard, and every rank's packed (value, id) pairs
    cand_vals: DevBuf,
    cand_ids: DevBuf,
    cand_lse: DevBuf,
    cand_pairs: DevBuf,
    // QSA: index queries [t, heads, 128], block scores [score rows, max blocks], selected KV slots [t, table stride] with
    // their counts, the identity row list of the selection table
    qsa_q: Option<DevBuf>,
    qsa_scores: Option<DevBuf>,
    qsa_table: Option<DevBuf>,
    qsa_counts: Option<DevBuf>,
    qsa_rows: Option<DevBuf>,
    logits: DevBuf,
    next: DevBuf,
    meta: DevBuf,
}

pub struct Model {
    pub cfg: ModelConfig,
    pub dims: Dims,
    pub tp: Tp,
    layers: Vec<Layer>,
    final_mix: FinalMix,
    lm_head: Linear,
    ple_table: Option<PleTable>,
    /// PLE short-conv state [seqs, hc * hidden, 9] bf16 and n-gram history [seqs, 2] int64.
    ple_state: Option<(DevBuf, DevBuf)>,
    embed: Arc<HostMapped>,
    moe: Option<MoeEngine>,
    lin: LinCtx,
    _scratch: DevBuf,
    inv_freq: DevBuf,
    rope_scale: f32,
    // KV pools of the softmax layers (NVFP4): [slots, kv_heads, head_dim / 2] codes, [.., head_dim / 16] scales
    kv: Vec<[DevBuf; 4]>,
    kv_global: DevBuf,
    /// QSA: compressed index keys per softmax layer [pages, 128] and the raw keys of each sequence's incomplete block
    /// [max_seqs, ratio, 128]
    qsa_comp: Vec<DevBuf>,
    qsa_ring: Vec<DevBuf>,
    // verify records (rows of the step, every verifying sequence's at its offset): PLE conv inputs [verify_rows,
    // channels], raw index keys per QSA layer [verify_rows, 128]; and per verify slot (the sequence's index in the
    // verify step): the ring rows before the verify [max_seqs, ratio, 128], the n-gram history before it
    ple_record: Option<DevBuf>,
    qsa_record: Vec<DevBuf>,
    qsa_ring_save: Vec<DevBuf>,
    hist_save: std::cell::RefCell<Vec<[i64; 2]>>,
    req_to_token: DevBuf,
    // GDN states: conv [seqs, channels, kernel - 1] bf16, recurrent [seqs, v_heads, dk, dv] f32
    conv_state: Vec<DevBuf>,
    rec_state: Vec<DevBuf>,
    // Verify records per GDN layer: pre-conv qkv [verify_rows, channels] bf16, g and beta [verify_rows, v_heads] f32
    records: Vec<[DevBuf; 3]>,
    /// commit metadata per verify slot ([0, accepted, row, pad] i32): the uploads are synchronous copies that do not
    /// wait for the model stream, so the commits of one step must not share it
    commit_meta: DevBuf,
    /// Captured hidden states [max_tokens, capture_layers, hidden] bf16.
    pub aux: Option<DevBuf>,
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
        let mut cfg = ModelConfig::load(dir)?;
        if let Some(f) = opts.yarn_factor.filter(|&f| f > 1.0) {
            if !matches!(cfg.rope.scaling, RopeScaling::None) {
                bail!("--yarn-factor: the config already scales its rope");
            }
            cfg.rope.scaling = RopeScaling::Yarn { factor: f, original_max: cfg.max_positions, beta_fast: 32.0, beta_slow: 1.0 };
            cfg.max_positions = (cfg.max_positions as f64 * f) as usize;
        }
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
                    let mut mods = vec![
                        rows(format!("{a}q_proj"), vec![qr.start * 2 * hd..qr.end * 2 * hd]),
                        rows(format!("{a}k_proj"), vec![kvr.start * hd..kvr.end * hd]),
                        rows(format!("{a}v_proj"), vec![kvr.start * hd..kvr.end * hd]),
                    ];
                    if cfg.qsa.is_some() {
                        if w > 1 {
                            bail!("tensor parallelism of QSA models is not supported yet");
                        }
                        mods.push(all(format!("{a}indexer.index_qk_proj")));
                    }
                    let qkv = Linear::load(&ckpt, &quant, &mods, s)?;
                    let index = match &cfg.qsa {
                        Some(_) => Some((
                            load_vec(&ckpt, &format!("{a}indexer.q_layernorm.weight"))?,
                            load_vec(&ckpt, &format!("{a}indexer.k_layernorm.weight"))?,
                        )),
                        None => None,
                    };
                    let attn = Attn {
                        qkv,
                        o: Linear::load(&ckpt, &quant, &[cols(format!("{a}o_proj"), qr.start * hd..qr.end * hd)], s)?,
                        q_norm: load_vec(&ckpt, &format!("{a}q_norm.weight"))?,
                        k_norm: load_vec(&ckpt, &format!("{a}k_norm.weight"))?,
                        index,
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
            let mix = if cfg.hc.is_some() {
                Mixing::Hyper {
                    attn: load_hc(&ckpt, &format!("{l}attn_hyper_connection."))?,
                    mlp: load_hc(&ckpt, &format!("{l}mlp_hyper_connection."))?,
                }
            } else {
                Mixing::Norm {
                    input: load_vec(&ckpt, &format!("{l}input_layernorm.weight"))?,
                    post: load_vec(&ckpt, &format!("{l}post_attention_layernorm.weight"))?,
                }
            };
            let ple = match &cfg.ple {
                Some(pc) if pc.layers.contains(&i) => {
                    let q = format!("{l}ple.");
                    let e = format!("{q}ple_embedding.");
                    Some(Ple {
                        key: Linear::load(&ckpt, &quant, &[all(format!("{q}key_proj"))], s)?,
                        value: Linear::load(&ckpt, &quant, &[all(format!("{q}value_proj"))], s)?,
                        norm_key: load_vec(&ckpt, &format!("{q}norm_key.weight"))?,
                        norm_query: load_vec(&ckpt, &format!("{q}norm_query.weight"))?,
                        norm_conv: load_vec(&ckpt, &format!("{q}norm_conv.weight"))?,
                        conv_w: load_vec(&ckpt, &format!("{q}conv1d.weight"))?,
                        mult: DevBuf::from_slice(ckpt.bytes(&format!("{e}layer_multipliers"))?)?,
                        sizes: DevBuf::from_slice(ckpt.bytes(&format!("{e}ngram_heads_vocab_sizes"))?)?,
                        offsets: DevBuf::from_slice(ckpt.bytes(&format!("{e}ngram_heads_offsets"))?)?,
                        host_hash: {
                            let i64s = |n: &str| -> Result<Vec<i64>> {
                                Ok(ckpt.bytes(n)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
                            };
                            (
                                i64s(&format!("{e}layer_multipliers"))?,
                                i64s(&format!("{e}ngram_heads_vocab_sizes"))?,
                                i64s(&format!("{e}ngram_heads_offsets"))?,
                            )
                        },
                    })
                }
                _ => None,
            };
            layers.push(Layer {
                kind: *kind,
                mix,
                ple,
                attn,
                gdn,
                mlp,
                slot,
            });
        }
        let final_mix = if cfg.hc.is_some() {
            FinalMix::Hyper(load_hc(&ckpt, &format!("{p}hyper_connection_mixer."))?)
        } else {
            FinalMix::Norm(load_vec(&ckpt, &format!("{p}norm.weight"))?)
        };
        if cfg.hc.is_some() && w > 1 {
            bail!("tensor parallelism of hyper-connection models is not supported yet");
        }
        let lm_name = if cfg.tie_embeddings { format!("{p}embed_tokens") } else { "lm_head".into() };
        let lm_head = Linear::load(&ckpt, &quant, &[rows(lm_name, vec![part(cfg.vocab, r, w)])], s)?;

        let (freqs, rope_scale) = inv_freq(&cfg);
        let inv_freq = DevBuf::from_slice(&freqs)?;

        // KV pools for every softmax layer, GDN states for every linear layer, this rank's heads.
        let mut kv = Vec::new();
        let (code_bytes, scale_bytes) = opts.kv_dtype.bytes();
        for _ in 0..n_full {
            let values = (opts.max_tokens * dims.kv_heads * hd) as f64;
            let (codes, scales) = ((values * code_bytes) as usize, ((values * scale_bytes) as usize).max(16));
            kv.push([DevBuf::zeroed(codes, s)?, DevBuf::zeroed(codes, s)?, DevBuf::zeroed(scales, s)?, DevBuf::zeroed(scales, s)?]);
        }
        let kv_global = DevBuf::from_slice(&[1.0f32])?;
        let (mut qsa_comp, mut qsa_ring) = (Vec::new(), Vec::new());
        let (mut qsa_record, mut qsa_ring_save) = (Vec::new(), Vec::new());
        if let Some(q) = &cfg.qsa {
            for _ in 0..n_full {
                qsa_comp.push(DevBuf::zeroed((opts.max_tokens / q.compress_ratio + 1) * q.head_dim * 2, s)?);
                qsa_ring.push(DevBuf::zeroed(opts.max_seqs * q.compress_ratio * q.head_dim * 2, s)?);
                if opts.max_verify > 0 {
                    qsa_record.push(DevBuf::zeroed(opts.verify_rows() * q.head_dim * 2, s)?);
                    qsa_ring_save.push(DevBuf::zeroed(opts.max_seqs * q.compress_ratio * q.head_dim * 2, s)?);
                }
            }
        }
        let ple_record = match (&cfg.ple, &cfg.hc, opts.max_verify > 0) {
            (Some(_), Some(hc), true) => Some(DevBuf::zeroed(opts.verify_rows() * hc.count * cfg.hidden * 2, s)?),
            _ => None,
        };
        let req_to_token = DevBuf::zeroed(opts.max_seqs * cfg.max_positions.min(opts.max_tokens) * 4, s)?;
        let state_rows = opts.max_seqs + opts.cache_rows;
        let (mut conv_state, mut rec_state) = (Vec::new(), Vec::new());
        if let Some(g) = &cfg.gdn {
            let channels = 2 * dims.k_heads * g.k_dim + dims.v_heads * g.v_dim;
            for _ in 0..n_gdn {
                conv_state.push(DevBuf::zeroed(state_rows * channels * (g.conv_kernel - 1) * 2, s)?);
                rec_state.push(DevBuf::zeroed(state_rows * dims.v_heads * g.k_dim * g.v_dim * 4, s)?);
            }
        }

        let mut records = Vec::new();
        if let (Some(g), true) = (&cfg.gdn, opts.max_verify > 0) {
            let channels = 2 * dims.k_heads * g.k_dim + dims.v_heads * g.v_dim;
            for _ in 0..n_gdn {
                records.push([
                    DevBuf::zeroed(opts.verify_rows() * channels * 2, s)?,
                    DevBuf::zeroed(opts.verify_rows() * dims.v_heads * 4, s)?,
                    DevBuf::zeroed(opts.verify_rows() * dims.v_heads * 4, s)?,
                ]);
            }
        }
        let commit_meta = DevBuf::zeroed(opts.max_seqs.max(1) * 16, s)?;
        // PLE: the n-gram table stays in its safetensors file, rows read on demand by CPU threads.
        let (mut ple_table, mut ple_state) = (None, None);
        if let (Some(pc), Some(hc)) = (&cfg.ple, &cfg.hc) {
            if pc.layers.len() != 1 {
                bail!("PLE on {} layers (one supported)", pc.layers.len());
            }
            let shard0 = format!("{p}layers.{}.ple.ple_embedding.ngram_embedding.shard_0.weight", pc.layers[0]);
            let path = &ckpt.files[ckpt.info(&shard0)?.file];
            let cpath = CString::new(path.to_string_lossy().as_bytes())?;
            let max_ids = opts.max_chunk.max(opts.max_seqs).max(opts.verify_rows()) * 2 * pc.heads_per_ngram;
            let mut table = null_mut();
            let mut err = vec![0i8; 512];
            if unsafe { sys::blz_ple_create(cpath.as_ptr(), max_ids as c_int, opts.ple_threads as c_int, opts.ple_cache_gib, &mut table, err.as_mut_ptr().cast(), err.len()) } != 0 {
                let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr().cast()) }.to_string_lossy().into_owned();
                bail!("PLE table: {msg}");
            }
            let table = PleTable(table);
            let row_dim = unsafe { sys::blz_ple_row_dim(table.0) } as usize;
            if row_dim * 2 * pc.heads_per_ngram != pc.embed_dim {
                bail!("PLE rows of {row_dim} values for a {}-wide embedding", pc.embed_dim);
            }
            let channels = hc.count * cfg.hidden;
            let state_len = (pc.conv_kernel - 1) * pc.ngram_size;
            let conv = DevBuf::zeroed(state_rows * channels * state_len * 2, s)?;
            let eos = cfg.eos.ok_or_else(|| anyhow!("PLE needs eos_token_id"))?;
            let hist = DevBuf::from_slice(&vec![eos; state_rows * 2])?;
            ple_table = Some(table);
            ple_state = Some((conv, hist));
        }
        let meta = MetaLayout::new(
            opts.max_chunk.max(opts.max_seqs).max(opts.verify_rows()),
            opts.max_seqs,
            opts.max_seqs.max(opts.verify_rows()),
            cfg.vocab,
        );
        let aux = if opts.capture_layers.is_empty() {
            None
        } else {
            Some(DevBuf::zeroed(meta.max_tokens * opts.capture_layers.len() * cfg.hidden * 2, s)?)
        };
        let work = Self::alloc_work(&cfg, &dims, w, &layers, &opts, &meta, s)?;
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
            if let Some(pl) = &l.ple {
                lins.extend([&pl.key, &pl.value]);
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
            final_mix,
            lm_head,
            ple_table,
            ple_state,
            embed,
            moe: None,
            lin: LinCtx { gemm, scratch: scratch.ptr(), scratch_elems },
            _scratch: scratch,
            inv_freq,
            rope_scale,
            kv,
            kv_global,
            qsa_comp,
            qsa_ring,
            ple_record,
            qsa_record,
            qsa_ring_save,
            hist_save: std::cell::RefCell::new(vec![[0, 0]; opts.max_seqs]),
            req_to_token,
            conv_state,
            rec_state,
            records,
            commit_meta,
            aux,
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
        let cores = CString::new(self.opts.moe_cores.as_bytes())?;
        let cfg = sys::blz_moe_engine_config {
            model_dir: model_dir.as_ptr(),
            topk: top_k as c_int,
            gpu0: 0,
            gpu1: if self.opts.moe_gpu1_gib > 0.0 { 1 } else { -1 },
            gpu0_gib: gib,
            gpu1_gib: self.opts.moe_gpu1_gib,
            ram_gib: self.opts.moe_ram_gib,
            cpu_threads: self.opts.moe_threads as c_int,
            cpu_cores: if self.opts.moe_cores.is_empty() { null() } else { cores.as_ptr() },
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
        eprintln!(
            "blaze: MoE engine: {:.2} GiB of expert slots on GPU0, {:.2} GiB on GPU1, the rest in pinned RAM{}",
            gib,
            self.opts.moe_gpu1_gib.max(0.0),
            if self.opts.moe_ram_gib >= 0.0 { " and on the SSD" } else { "" }
        );
        Ok(MoeEngine(engine))
    }

    fn alloc_work(cfg: &ModelConfig, d: &Dims, world: usize, layers: &[Layer], opts: &Options, meta: &MetaLayout, s: &Stream) -> Result<Work> {
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
        let mut partial_rows = (t * parts_prefill).max(opts.max_seqs * parts_decode);
        if cfg.qsa.is_some() {
            // sparse attention: every row a one-token sequence, rows x splits <= max(rows, 4 x SMs / kv heads)
            let per_split = unsafe { sys::blz_nvfp4_mha_parts(heads as c_int, d.kv_heads as c_int, 1, 1) } as usize;
            partial_rows = partial_rows.max(t.max(opts.max_seqs).max(opts.verify_rows()).max(1024) * per_split);
        }
        let _ = proj_cols;
        // rows the head may take in one step: one per sequence, or every row of a verify
        let head_rows = opts.max_seqs.max(opts.verify_rows());
        let ctx = cfg.max_positions.min(opts.max_tokens);
        let hcw = cfg.hc.as_ref().map(|hc| hc.count * h);
        let ple_c = cfg.ple.as_ref().and(hcw);
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
            gx: b(head_rows * h * 2)?,
            gres: b(head_rows * h * 2)?,
            gnormed: b(head_rows * h * 2)?,
            streams: hcw.map(|c| b(t * c * 2)).transpose()?,
            hnorm: hcw.map(|c| b(t * c * 2)).transpose()?,
            hc_t: cfg.hc.as_ref().map(|hc| b(t.max(head_rows) * hc.lowrank * 2)).transpose()?,
            hc_ws: cfg
                .hc
                .as_ref()
                .map(|hc| {
                    let rows = t.max(head_rows) as c_int;
                    let bytes = unsafe { sys::blz_hc_mix_workspace_bytes(rows, hc.count as c_int, h as c_int, hc.lowrank as c_int) };
                    b(bytes.max(16))
                })
                .transpose()?,
            gstreams: hcw.map(|c| b(head_rows * c * 2)).transpose()?,
            hc_partial: hcw.map(|c| b(16usize.max(head_rows) * c.div_ceil(1024) * 8 * 4)).transpose()?,
            ghnorm: hcw.map(|c| b(head_rows * c * 2)).transpose()?,
            ple_ids: cfg.ple.as_ref().map(|p| b(t * 2 * p.heads_per_ngram * 8)).transpose()?,
            ple_emb: cfg.ple.as_ref().map(|p| b(t * p.embed_dim * 2)).transpose()?,
            ple_key: ple_c.map(|c| b(t * c * 2)).transpose()?,
            ple_value: cfg.ple.as_ref().map(|_| b(t * h * 2)).transpose()?,
            ple_gated: ple_c.map(|c| b(t * c * 2)).transpose()?,
            ple_gatedn: ple_c.map(|c| b(t * c * 2)).transpose()?,
            pairs: b((head_rows * 8 * 8).next_multiple_of(16))?,
            cand_vals: b(head_rows * sys::BLZ_TOPK_MAX * 4)?,
            cand_ids: b(head_rows * sys::BLZ_TOPK_MAX * 4)?,
            cand_lse: b(head_rows * 4)?,
            cand_pairs: b((world * head_rows * (sys::BLZ_TOPK_MAX + 1) * 8).next_multiple_of(16))?,
            qsa_q: cfg.qsa.as_ref().map(|q| b(t.max(head_rows) * q.heads * q.head_dim * 2)).transpose()?,
            qsa_scores: cfg.qsa.as_ref().map(|q| b(QSA_SCORE_ROWS * (ctx / q.compress_ratio + 1) * 4)).transpose()?,
            qsa_table: cfg.qsa.as_ref().map(|_| b(t.max(head_rows) * QSA_TABLE_STRIDE * 4)).transpose()?,
            qsa_counts: cfg.qsa.as_ref().map(|_| b(t.max(head_rows) * 4)).transpose()?,
            qsa_rows: cfg
                .qsa
                .as_ref()
                .map(|_| DevBuf::from_slice(&(0..t.max(head_rows) as i32).collect::<Vec<i32>>()))
                .transpose()?,
            logits: b(head_rows * cfg.vocab * 2)?,
            next: b(head_rows * 4)?,
            meta: b(meta.bytes)?,
        })
    }

    /// KV slots per page: QSA keeps one compressed key per page of `compress_ratio` slots.
    pub fn page_size(&self) -> usize {
        self.cfg.qsa.as_ref().map_or(1, |q| q.compress_ratio)
    }

    /// Context above which QSA selects (below it, the selection is every visible token: dense attention).
    pub fn qsa_budget(&self) -> usize {
        self.cfg.qsa.as_ref().map_or(usize::MAX, |q| q.budget)
    }

    /// Positions per request row of the KV table.
    pub fn ctx_stride(&self) -> usize {
        self.cfg.max_positions.min(self.opts.max_tokens)
    }

    pub fn meta_ptr<T>(&self, offset: usize) -> *mut T {
        self.work.meta.at::<u8>(offset).cast()
    }

    /// The linear-layer context (cuBLAS, scratch) of this rank, for a drafter running next to the model.
    pub fn lin_ctx(&self) -> &LinCtx {
        &self.lin
    }

    /// The pinned token embedding table [vocab, hidden] bf16.
    pub fn embed_table(&self) -> &HostMapped {
        &self.embed
    }

    pub fn sms(&self) -> usize {
        self.num_sms
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
        if let (Some((conv, hist)), Some(pc), Some(hc)) = (&self.ple_state, &self.cfg.ple, &self.cfg.hc) {
            let bytes = hc.count * self.cfg.hidden * (pc.conv_kernel - 1) * pc.ngram_size * 2;
            check(unsafe { sys::cudaMemsetAsync(conv.at::<u8>(slot * bytes).cast(), 0, bytes, s.0) }, "memset")?;
            let eos = self.cfg.eos.unwrap();
            s.sync()?;
            hist.upload_at(slot * 16, &[eos, eos])?;
        }
        Ok(())
    }

    /// Debug: with BLAZE_DUMP=<dir>, write `rows` x `width` bf16 values at `ptr` to <dir>/<name>.bin (eager forwards
    /// only; a capture skips it).
    fn dump(&self, name: &str, ptr: *const u16, rows: usize, width: usize) -> Result<()> {
        let Some(dir) = std::env::var_os("BLAZE_DUMP") else { return Ok(()) };
        let mut cap: c_int = 0;
        unsafe {
            unsafe extern "C" {
                fn cudaStreamIsCapturing(s: sys::cudaStream_t, status: *mut c_int) -> c_int;
            }
            cudaStreamIsCapturing(self.stream.0, &mut cap);
        }
        if cap != 0 {
            return Ok(());
        }
        self.stream.sync()?;
        let mut host = vec![0u16; rows * width];
        check(unsafe { sys::cudaMemcpy(host.as_mut_ptr().cast(), ptr.cast(), rows * width * 2, sys::MEMCPY_D2H) }, "dump")?;
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::create_dir_all(&dir)?;
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.bin")), bytes)?;
        Ok(())
    }

    /// The streams' input to a block: per-stream norm, then the gated low-rank mix into `normed`.
    fn hc_enter(&self, hc: &HcBlock, rows: usize, streams: *const u16, hnorm: *mut u16, out: *mut u16) -> Result<()> {
        let (w, s, h) = (&self.work, &self.stream, self.cfg.hidden);
        let c = self.cfg.hc.as_ref().unwrap();
        check(
            unsafe { sys::blz_grouped_rmsnorm(streams, hc.norm.ptr(), self.cfg.rms_eps, rows as c_int, c.count as c_int, h as c_int, hnorm, s.0) },
            "hc norm",
        )?;
        check(
            unsafe {
                sys::blz_hc_mix(hnorm, hc.down.ptr(), hc.up.ptr(), w.hc_t.as_ref().unwrap().ptr(), out, rows as c_int, c.count as c_int,
                                h as c_int, c.lowrank as c_int, w.hc_ws.as_ref().unwrap().ptr(), s.0)
            },
            "hc mix",
        )
    }

    /// The block's output x back into the streams (gated injection, gates from the streams' norm).
    fn hc_exit(&self, hc: &HcBlock, rows: usize) -> Result<()> {
        let (w, s, h) = (&self.work, &self.stream, self.cfg.hidden);
        let c = self.cfg.hc.as_ref().unwrap();
        let (streams, hnorm, inject) = (w.streams.as_ref().unwrap(), w.hnorm.as_ref().unwrap(), hc.inject.as_ref().unwrap());
        if rows <= 16 {
            // few rows: gate dot products over many blocks
            return check(
                unsafe {
                    sys::blz_hc_combine_split(streams.ptr(), hnorm.ptr(), inject.ptr(), w.x.ptr(), h as i64, rows as c_int, c.count as c_int,
                                              h as c_int, w.hc_partial.as_ref().unwrap().ptr(), s.0)
                },
                "hc combine",
            );
        }
        check(
            unsafe {
                sys::blz_hc_combine(streams.ptr(), hnorm.ptr(), inject.ptr(), w.x.ptr(), h as i64, rows as c_int, c.count as c_int, h as c_int,
                                    s.0)
            },
            "hc combine",
        )
    }

    /// Start reading the PLE rows of `tokens` (appended after `prev`, the sequence's two previous tokens, or fewer at
    /// its start) into the page cache: the same hash as blz_ngram_hash, on the host.
    pub fn ple_warm(&self, prev: &[i64], tokens: &[i64]) {
        let (Some(table), Some(pc)) = (&self.ple_table, &self.cfg.ple) else { return };
        let ple = self.layers[pc.layers[0]].ple.as_ref().unwrap();
        let (mult, sizes, offsets) = &ple.host_hash;
        let eos = self.cfg.eos.unwrap_or(0);
        let mut win: Vec<i64> = vec![eos, eos];
        for (i, &t) in prev.iter().rev().take(2).enumerate() {
            win[1 - i] = t;
        }
        win.extend_from_slice(tokens);
        let hpn = pc.heads_per_ngram;
        let mut ids = Vec::with_capacity(tokens.len() * 2 * hpn);
        for i in 0..tokens.len() {
            let (w0, w1, w2) = (win[i], win[i + 1], win[i + 2]);
            let reach = if w1 == eos { 0 } else if w0 == eos { 1 } else { 2 };
            let t1 = (if reach >= 1 { w1 } else { eos }) as u64;
            let t2 = (if reach >= 2 { w0 } else { eos }) as u64;
            let m2 = (w2 as u64).wrapping_mul(mult[0] as u64) ^ t1.wrapping_mul(mult[1] as u64);
            let m3 = m2 ^ t2.wrapping_mul(mult[2] as u64);
            for h in 0..2 * hpn {
                let mix = (if h < hpn { m2 } else { m3 }) as i64;
                ids.push(mix.rem_euclid(sizes[h]) + offsets[h]);
            }
        }
        unsafe { sys::blz_ple_warm(table.0, ids.as_ptr(), ids.len() as c_int) };
    }

    /// n-gram row ids of the step's tokens and the table request (the CPU reads them while the first layers run).
    fn ple_request(&self, t: usize, seqs: usize, verify: bool) -> Result<()> {
        let (Some(table), Some((_, hist)), Some(pc)) = (&self.ple_table, &self.ple_state, &self.cfg.ple) else { return Ok(()) };
        let ple = self.layers[pc.layers[0]].ple.as_ref().unwrap();
        let (w, s, m) = (&self.work, &self.stream, &self.meta);
        let ids = w.ple_ids.as_ref().unwrap();
        check(
            unsafe {
                sys::blz_ngram_hash(self.meta_ptr(m.tokens), self.meta_ptr(m.seq_start), self.meta_ptr(m.seq_slots), seqs as c_int,
                                    t as c_int, hist.ptr(), (!verify) as c_int, ple.mult.ptr(), ple.sizes.ptr(), ple.offsets.ptr(),
                                    pc.heads_per_ngram as c_int, self.cfg.eos.unwrap(), ids.ptr(), s.0)
            },
            "n-gram hash",
        )?;
        check(unsafe { sys::blz_ple_request(table.0, ids.ptr(), (t * 2 * pc.heads_per_ngram) as c_int, s.0) }, "PLE request")
    }

    /// PLE of `layer`: the embedding rows, key / value projections, the gate against the streams, the dilated short
    /// conv; the result is added to the streams.
    fn ple_apply(&self, ple: &Ple, t: usize, seqs: usize, verify: bool) -> Result<()> {
        let (w, s, h, m) = (&self.work, &self.stream, self.cfg.hidden, &self.meta);
        let (pc, hc) = (self.cfg.ple.as_ref().unwrap(), self.cfg.hc.as_ref().unwrap());
        let (table, (conv, _)) = (self.ple_table.as_ref().unwrap(), self.ple_state.as_ref().unwrap());
        let ch = hc.count * h;
        let (emb, key, value) = (w.ple_emb.as_ref().unwrap(), w.ple_key.as_ref().unwrap(), w.ple_value.as_ref().unwrap());
        let (gated, gatedn) = (w.ple_gated.as_ref().unwrap(), w.ple_gatedn.as_ref().unwrap());
        let (streams, query_n) = (w.streams.as_ref().unwrap(), w.hnorm.as_ref().unwrap());
        let eps = self.cfg.rms_eps;
        check(unsafe { sys::blz_ple_fetch(table.0, emb.ptr(), (t * 2 * pc.heads_per_ngram) as c_int, s.0) }, "PLE fetch")?;
        ple.key.run(&self.lin, emb.ptr(), pc.embed_dim, t, key.ptr(), ch, s)?;
        ple.value.run(&self.lin, emb.ptr(), pc.embed_dim, t, value.ptr(), h, s)?;
        let g = hc.count as c_int;
        check(unsafe { sys::blz_grouped_rmsnorm(key.ptr(), ple.norm_key.ptr(), eps, t as c_int, g, h as c_int, key.ptr(), s.0) }, "PLE key norm")?;
        check(
            unsafe { sys::blz_grouped_rmsnorm(streams.ptr(), ple.norm_query.ptr(), eps, t as c_int, g, h as c_int, query_n.ptr(), s.0) },
            "PLE query norm",
        )?;
        check(unsafe { sys::blz_ple_gate(key.ptr(), query_n.ptr(), value.ptr(), t as c_int, g, h as c_int, gated.ptr(), s.0) }, "PLE gate")?;
        check(
            unsafe { sys::blz_grouped_rmsnorm(gated.ptr(), ple.norm_conv.ptr(), eps, t as c_int, g, h as c_int, gatedn.ptr(), s.0) },
            "PLE conv norm",
        )?;
        if verify && let Some(rec) = &self.ple_record {
            check(unsafe { sys::cudaMemcpyAsync(rec.ptr(), gatedn.ptr::<u16>().cast(), t * ch * 2, sys::MEMCPY_D2D, s.0) }, "ple record")?;
        }
        check(
            unsafe {
                sys::blz_ple_conv(gatedn.ptr(), gated.ptr(), self.meta_ptr(m.seq_start), self.meta_ptr(m.seq_slots), seqs as c_int,
                                  ch as c_int, pc.conv_kernel as c_int, pc.ngram_size as c_int, ple.conv_w.ptr(), conv.ptr(),
                                  (!verify) as c_int, streams.ptr(), s.0)
            },
            "PLE conv",
        )
    }

    /// Enqueue the forward of a step whose metadata is already in the device buffer (MetaLayout). `spans`: decode
    /// sequences (len 1) first, then prefill chunks. The head runs on the first `n_logits` rows listed in
    /// logit_rows and leaves greedy picks in `next`. `verify`: one span of draft rows whose GDN state is only read
    /// (its inputs are recorded for commit). Capturable into a CUDA graph (no host synchronization).
    pub fn forward(&self, spans: &[Span], n_logits: usize, verify: bool, sample: bool, sparse: bool) -> Result<()> {
        let t: usize = spans.iter().map(|s| s.len).sum();
        let n = spans.len();
        if t == 0 || t > self.meta.max_tokens || n > self.meta.max_seqs || n_logits > self.meta.max_tokens {
            bail!("step of {t} tokens in {n} sequences exceeds the buffers");
        }
        if verify && spans.iter().any(|sp| sp.len > self.opts.max_verify) {
            bail!("verify of more than {} rows in a sequence", self.opts.max_verify);
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
        self.ple_request(t, n, verify)?;
        for (li, layer) in self.layers.iter().enumerate() {
            match &layer.mix {
                Mixing::Norm { input, .. } => {
                    if li == 0 {
                        check(unsafe { sys::blz_rmsnorm(w.residual.ptr(), null_mut(), input.ptr(), eps, t as c_int, h as c_int, 1, w.normed.ptr(), s.0) }, "rmsnorm")?;
                    } else {
                        self.reduce_norm(input, t)?;
                    }
                }
                Mixing::Hyper { attn, .. } => {
                    let streams = w.streams.as_ref().unwrap();
                    if li == 0 {
                        let c = cfg.hc.as_ref().unwrap().count;
                        check(unsafe { sys::blz_hc_widen(w.residual.ptr(), t as c_int, c as c_int, h as c_int, streams.ptr(), s.0) }, "hc widen")?;
                    }
                    if let Some(ple) = &layer.ple {
                        self.ple_apply(ple, t, n, verify)?;
                        self.dump(&format!("l{li}_ple"), streams.ptr(), t, cfg.hc.as_ref().unwrap().count * h)?;
                    }
                    self.hc_enter(attn, t, streams.ptr(), w.hnorm.as_ref().unwrap().ptr(), w.normed.ptr())?;
                    self.dump(&format!("l{li}_attn_in"), w.normed.ptr(), t, h)?;
                }
            }
            if let Some(ci) = self.opts.capture_layers.iter().position(|&c| c == li) {
                let aux = self.aux.as_ref().unwrap();
                let ncap = self.opts.capture_layers.len();
                check(
                    unsafe {
                        sys::cudaMemcpy2DAsync(aux.at::<u16>(ci * h).cast(), ncap * h * 2, w.residual.ptr::<u16>().cast(), h * 2, h * 2,
                                               t, sys::MEMCPY_D2D, s.0)
                    },
                    "capture",
                )?;
            }
            match layer.kind {
                LayerKind::Full => self.attention(layer, t, spans, sparse, verify)?,
                LayerKind::Linear => self.gdn(layer, t, n, verify)?,
            }
            match &layer.mix {
                Mixing::Norm { post, .. } => self.reduce_norm(post, t)?,
                Mixing::Hyper { attn, mlp } => {
                    self.dump(&format!("l{li}_attn_out"), w.x.ptr(), t, h)?;
                    self.hc_exit(attn, t)?;
                    let streams = w.streams.as_ref().unwrap();
                    self.hc_enter(mlp, t, streams.ptr(), w.hnorm.as_ref().unwrap().ptr(), w.normed.ptr())?;
                }
            }
            self.mlp(li, layer, t)?;
            if let Mixing::Hyper { mlp, .. } = &layer.mix {
                self.dump(&format!("l{li}_mlp_out"), w.x.ptr(), t, h)?;
                self.hc_exit(mlp, t)?;
                let streams = w.streams.as_ref().unwrap();
                self.dump(&format!("l{li}_streams"), streams.ptr(), t, cfg.hc.as_ref().unwrap().count * h)?;
            }
        }
        self.all_reduce(w.x.ptr(), t * h)?;
        self.prefetch_join()?;
        if n_logits > 0 {
            // h = x + residual of the listed rows, normed, through the head, greedy pick.
            let n = n_logits;
            let rows: *const i32 = self.meta_ptr(m.logit_rows);
            match &self.final_mix {
                FinalMix::Norm(norm) => {
                    check(unsafe { sys::blz_gather_rows(w.x.ptr(), h as i64, rows, n as c_int, h as c_int, w.gx.ptr(), s.0) }, "gather")?;
                    check(unsafe { sys::blz_gather_rows(w.residual.ptr(), h as i64, rows, n as c_int, h as c_int, w.gres.ptr(), s.0) }, "gather")?;
                    check(unsafe { sys::blz_rmsnorm(w.gx.ptr(), w.gres.ptr(), norm.ptr(), eps, n as c_int, h as c_int, 1, w.gnormed.ptr(), s.0) }, "final norm")?;
                }
                FinalMix::Hyper(mixer) => {
                    let width = cfg.hc.as_ref().unwrap().count * h;
                    let (gs, gh) = (w.gstreams.as_ref().unwrap(), w.ghnorm.as_ref().unwrap());
                    let streams = w.streams.as_ref().unwrap();
                    check(unsafe { sys::blz_gather_rows(streams.ptr(), width as i64, rows, n as c_int, width as c_int, gs.ptr(), s.0) }, "gather")?;
                    self.hc_enter(mixer, n, gs.ptr(), gh.ptr(), w.gnormed.ptr())?;
                }
            }
            if sample {
                return self.sample_candidates(w.gnormed.ptr(), n);
            }
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

    /// Sampling candidates of `rows` normed states: this rank's vocabulary shard through the head, the rows' penalties,
    /// the shard's top BLZ_TOPK_MAX, packed as (value, id) pairs at the rank's slot of cand_pairs and merged over the
    /// group (cand_pairs: [world][rows][K][2] floats).
    fn sample_candidates(&self, normed: *const u16, rows: usize) -> Result<()> {
        let (w, s, v, m) = (&self.work, &self.stream, self.dims.vocab, &self.meta);
        let (rank, world, k) = (self.tp.rank, self.tp.world, sys::BLZ_TOPK_MAX);
        self.lm_head.run(&self.lin, normed, self.cfg.hidden, rows, w.logits.ptr(), v, s)?;
        check(
            unsafe {
                sys::blz_logit_edits(w.logits.ptr(), v as i64, rows as c_int, (rank * v) as c_int, v as c_int, self.meta_ptr(m.pen_ids),
                                     self.meta_ptr(m.pen_coef), self.meta_ptr(m.pen_off), s.0)
            },
            "logit edits",
        )?;
        check(
            unsafe {
                sys::blz_token_mask(w.logits.ptr(), v as i64, rows as c_int, (rank * v) as c_int, v as c_int, m.mask_words as c_int,
                                    self.meta_ptr(m.mask), self.meta_ptr(m.mask_row), s.0)
            },
            "token mask",
        )?;
        check(unsafe { sys::blz_row_lse(w.logits.ptr(), v as i64, rows as c_int, v as c_int, w.cand_lse.ptr(), s.0) }, "lse")?;
        check(
            unsafe {
                sys::blz_topk(w.logits.ptr(), v as i64, rows as c_int, v as c_int, (rank * v) as c_int, k as c_int, w.cand_vals.ptr(),
                              w.cand_ids.ptr(), null_mut(), s.0)
            },
            "top-k",
        )?;
        let bytes = (world * rows * (k + 1) * 8).next_multiple_of(16);
        if world > 1 {
            check(unsafe { sys::cudaMemsetAsync(w.cand_pairs.ptr(), 0, bytes, s.0) }, "memset")?;
        }
        check(
            unsafe {
                sys::blz_topk_pack(w.cand_vals.ptr(), w.cand_ids.ptr(), w.cand_lse.ptr(), rows as c_int, k as c_int,
                                   w.cand_pairs.at::<f32>(rank * rows * (k + 1) * 2), s.0)
            },
            "pack",
        )?;
        if world > 1 {
            check(unsafe { sys::blz_host_ar_all_reduce(self.tp.ar, w.cand_pairs.ptr(), bytes as i64, sys::BLZ_AR_FP32, s.0) }, "all-reduce")?;
        }
        Ok(())
    }

    /// The merged candidate pairs of the last sampled forward: [world][rows][K + 1][2] floats (the last pair of a row:
    /// the shard's log-sum-exp).
    pub fn candidates(&self, rows: usize) -> Result<Vec<f32>> {
        self.stream.sync()?;
        self.work.cand_pairs.download::<f32>(self.tp.world * rows * (sys::BLZ_TOPK_MAX + 1) * 2)
    }

    /// Copy the recurrent state (GDN conv and recurrent, PLE conv and n-gram history) of row `src` to row `dst`.
    pub fn copy_state(&self, src: usize, dst: usize) -> Result<()> {
        let s = &self.stream;
        let copy = |buf: &DevBuf, row_bytes: usize| -> Result<()> {
            check(
                unsafe {
                    sys::cudaMemcpyAsync(buf.at::<u8>(dst * row_bytes).cast(), buf.at::<u8>(src * row_bytes).cast(), row_bytes,
                                         sys::MEMCPY_D2D, s.0)
                },
                "state copy",
            )
        };
        if let Some(g) = &self.cfg.gdn {
            let d = &self.dims;
            let channels = 2 * d.k_heads * g.k_dim + d.v_heads * g.v_dim;
            for (c, r) in self.conv_state.iter().zip(&self.rec_state) {
                copy(c, channels * (g.conv_kernel - 1) * 2)?;
                copy(r, d.v_heads * g.k_dim * g.v_dim * 4)?;
            }
        }
        if let (Some((conv, hist)), Some(pc), Some(hc)) = (&self.ple_state, &self.cfg.ple, &self.cfg.hc) {
            copy(conv, hc.count * self.cfg.hidden * (pc.conv_kernel - 1) * pc.ngram_size * 2)?;
            copy(hist, 16)?;
        }
        Ok(())
    }

    /// Row `row` of the KV table: position p's KV slot = slots[p] (a sequence starting from a cached prefix).
    pub fn set_kv_row(&self, row: usize, slots: &[i32]) -> Result<()> {
        if slots.is_empty() {
            return Ok(());
        }
        self.stream.sync()?;
        self.req_to_token.upload_at(row * self.ctx_stride() * 4, slots)
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
        match &self.final_mix {
            FinalMix::Norm(norm) => check(
                unsafe { sys::blz_rmsnorm(w.x.ptr(), w.residual.ptr(), norm.ptr(), self.cfg.rms_eps, t as c_int, self.cfg.hidden as c_int, 1, w.normed.ptr(), s.0) },
                "final norm",
            )?,
            FinalMix::Hyper(mixer) => {
                let streams = w.streams.as_ref().unwrap();
                self.hc_enter(mixer, t, streams.ptr(), w.hnorm.as_ref().unwrap().ptr(), w.normed.ptr())?
            }
        }
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

    /// QSA of a softmax layer: index queries and compressed keys of the step (always, so later steps can select), then
    /// with `sparse` the KV slots every row attends to, in the selection table (rows 0..t, counts per row).
    fn qsa_index(&self, layer: &Layer, t: usize, spans: &[Span], sparse: bool, verify: bool) -> Result<()> {
        let (a, q) = (layer.attn.as_ref().unwrap(), self.cfg.qsa.as_ref().unwrap());
        let (qn, kn) = a.index.as_ref().unwrap();
        let (cfg, w, s, m) = (&self.cfg, &self.work, &self.stream, &self.meta);
        let (heads, kvh, hd) = (self.dims.heads, self.dims.kv_heads, cfg.attn.head_dim);
        let ld = a.qkv.n;
        let col = heads * 2 * hd + 2 * kvh * hd;
        let qbuf = w.qsa_q.as_ref().unwrap();
        check(
            unsafe {
                sys::blz_qsa_index_q(w.proj.ptr(), ld as i64, col as c_int, self.meta_ptr(m.positions), qn.ptr(), cfg.rms_eps, t as c_int,
                                     q.heads as c_int, q.head_dim as c_int, cfg.rope.rotary_dim as c_int, self.inv_freq.ptr(),
                                     self.rope_scale, qbuf.ptr(), s.0)
            },
            "qsa index q",
        )?;
        let (comp, ring) = (&self.qsa_comp[layer.slot], &self.qsa_ring[layer.slot]);
        if verify && let Some(rec) = self.qsa_record.get(layer.slot) {
            // raw index keys of the verify rows, for the commit to rebuild the ring
            let kcol = col + q.heads * q.head_dim;
            check(
                unsafe {
                    sys::cudaMemcpy2DAsync(rec.ptr(), q.head_dim * 2, w.proj.at::<u16>(kcol).cast(), ld * 2, q.head_dim * 2, t,
                                           sys::MEMCPY_D2D, s.0)
                },
                "qsa record",
            )?;
        }
        check(
            unsafe {
                sys::blz_qsa_compress(w.proj.ptr(), ld as i64, (col + q.heads * q.head_dim) as c_int, self.meta_ptr(m.positions),
                                      self.meta_ptr(m.tok_row), self.meta_ptr(m.tok_kv_len), t as c_int, q.head_dim as c_int,
                                      q.compress_ratio as c_int, kn.ptr(), cfg.rms_eps, cfg.rope.rotary_dim as c_int,
                                      self.inv_freq.ptr(), self.rope_scale, self.req_to_token.ptr(), self.ctx_stride() as i64,
                                      ring.ptr(), comp.ptr(), s.0)
            },
            "qsa compress",
        )?;
        if !sparse {
            return Ok(());
        }
        let max_blocks = self.ctx_stride() / q.compress_ratio + 1;
        let (scores, table, counts) = (w.qsa_scores.as_ref().unwrap(), w.qsa_table.as_ref().unwrap(), w.qsa_counts.as_ref().unwrap());
        let block_topk = q.budget / q.compress_ratio;
        // decode rows one sequence each, prefill spans in passes of QSA_SCORE_ROWS rows of one sequence
        let mut passes: Vec<(usize, usize, bool)> = Vec::new();
        for sp in spans {
            if sp.len == 1 {
                match passes.last_mut() {
                    Some((r0, n, false)) if *r0 + *n == sp.offset && *n < QSA_SCORE_ROWS => *n += 1,
                    _ => passes.push((sp.offset, 1, false)),
                }
            } else {
                let mut r = sp.offset;
                while r < sp.offset + sp.len {
                    let n = QSA_SCORE_ROWS.min(sp.offset + sp.len - r);
                    passes.push((r, n, true));
                    r += n;
                }
            }
        }
        for (r0, n, same) in passes {
            check(
                unsafe {
                    sys::blz_qsa_scores(qbuf.ptr(), q.heads as c_int, q.head_dim as c_int, self.meta_ptr(m.positions), self.meta_ptr(m.tok_row),
                                        r0 as c_int, n as c_int, same as c_int, q.compress_ratio as c_int, self.req_to_token.ptr(),
                                        self.ctx_stride() as i64, comp.ptr(), max_blocks as c_int, scores.ptr(), max_blocks as i64, s.0)
                },
                "qsa scores",
            )?;
            check(
                unsafe {
                    sys::blz_qsa_select(scores.ptr(), max_blocks as i64, self.meta_ptr(m.positions), self.meta_ptr(m.tok_row), r0 as c_int,
                                        n as c_int, q.compress_ratio as c_int, block_topk as c_int, self.req_to_token.ptr(),
                                        self.ctx_stride() as i64, table.ptr(), QSA_TABLE_STRIDE as i64, counts.ptr(), s.0)
                },
                "qsa select",
            )?;
        }
        Ok(())
    }

    fn attention(&self, layer: &Layer, t: usize, spans: &[Span], sparse: bool, verify: bool) -> Result<()> {
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
        let kvd = self.opts.kv_dtype;
        check(
            unsafe {
                match kvd {
                    KvDtype::Nvfp4FourOverSix | KvDtype::Nvfp4 => sys::blz_nvfp4_kv_store(
                        k.cast(), v.cast(), ld as i64, ld as i64, sys::BLZ_BF16, self.meta_ptr::<i64>(m.locs).cast(),
                        sys::BLZ_I64, t as i64, kvh as c_int, hd as c_int, pool[0].ptr(), pool[1].ptr(), pool[2].ptr(),
                        pool[3].ptr(), self.kv_global.ptr(), self.kv_global.ptr(),
                        (kvd == KvDtype::Nvfp4FourOverSix) as c_int, s.0),
                    _ => sys::blz_kv_store_dense(k.cast(), v.cast(), ld as i64, ld as i64, self.meta_ptr::<i64>(m.locs).cast(),
                                                 sys::BLZ_I64, t as i64, kvh as c_int, hd as c_int, kvd.format(), pool[0].ptr(),
                                                 pool[1].ptr(), self.kv_global.ptr(), self.kv_global.ptr(), s.0),
                }
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
            format: kvd.format(),
        };
        let sparse = sparse && self.cfg.qsa.is_some();
        if self.cfg.qsa.is_some() {
            self.qsa_index(layer, t, spans, sparse, verify)?;
        }
        if sparse {
            // every row attends to its selected slots: the selection table is a KV table of one-row "sequences"
            let (table, counts, rows) = (w.qsa_table.as_ref().unwrap(), w.qsa_counts.as_ref().unwrap(), w.qsa_rows.as_ref().unwrap());
            let splits = (4 * self.num_sms / (t * kvh)).clamp(1, 16);
            check(
                unsafe {
                    sys::blz_nvfp4_mha_decode(
                        w.q.ptr::<u16>().cast(),
                        (heads * hd) as i64,
                        hd as i64,
                        heads as c_int,
                        view,
                        table.ptr(),
                        QSA_TABLE_STRIDE as i64,
                        sys::blz_index_ptr { ptr: rows.ptr::<i32>().cast(), kind: sys::BLZ_I32 },
                        sys::blz_index_ptr { ptr: counts.ptr::<i32>().cast(), kind: sys::BLZ_I32 },
                        t as c_int,
                        1,
                        splits as c_int,
                        1.0 / (hd as f32).sqrt(),
                        w.partial_o.ptr(),
                        w.partial_ml.ptr(),
                        sys::BLZ_BF16,
                        w.attn.ptr::<u16>().cast(),
                        (heads * hd) as i64,
                        hd as i64,
                        s.0,
                    )
                },
                "qsa mha",
            )?;
            check(unsafe { sys::blz_mul_sigmoid(w.attn.ptr(), w.gate.ptr(), (t * heads * hd) as i64, s.0) }, "gate")?;
            a.o.run(&self.lin, w.attn.ptr(), heads * hd, t, w.x.ptr(), cfg.hidden, s)?;
            return self.prefetch_mlp(layer, t);
        }
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

    /// Before a verify of row `row` (verify slot `k`: the sequence's index among the step's verifying sequences): keep
    /// what the verify overwrites and the commit restores (QSA rings, the n-gram history).
    pub fn pre_verify(&self, row: usize, k: usize) -> Result<()> {
        let s = &self.stream;
        if let Some(q) = &self.cfg.qsa {
            let bytes = q.compress_ratio * q.head_dim * 2;
            for (ring, save) in self.qsa_ring.iter().zip(&self.qsa_ring_save) {
                check(
                    unsafe { sys::cudaMemcpyAsync(save.at::<u8>(k * bytes).cast(), ring.at::<u8>(row * bytes).cast(), bytes, sys::MEMCPY_D2D, s.0) },
                    "ring save",
                )?;
            }
        }
        if let Some((_, hist)) = &self.ple_state {
            s.sync()?;
            let mut h = [0i64; 2];
            check(unsafe { sys::cudaMemcpy(h.as_mut_ptr().cast(), hist.at::<u8>(row * 16).cast(), 16, sys::MEMCPY_D2H) }, "hist")?;
            self.hist_save.borrow_mut()[k] = h;
        }
        Ok(())
    }

    /// Replay the first `accepted` rows of the last verify (row `row`'s sequence, verify slot `k`, its rows at `offset`
    /// in the step, which started at position `start` with `tokens`) into the recurrent states: GDN conv and recurrent
    /// state, the PLE conv state and n-gram history, the QSA ring of the trailing block, as if the rows had been
    /// decoded one by one.
    pub fn commit(&self, row: usize, accepted: usize, start: usize, tokens: &[i64], k: usize, offset: usize) -> Result<()> {
        if accepted == 0 {
            return Ok(());
        }
        let s = &self.stream;
        if let Some(q) = &self.cfg.qsa {
            // ring = the saved one, plus the accepted rows of the trailing block
            let (dim, ratio) = (q.head_dim, q.compress_ratio);
            let bytes = ratio * dim * 2;
            let end = start + accepted;
            for ((ring, save), rec) in self.qsa_ring.iter().zip(&self.qsa_ring_save).zip(&self.qsa_record) {
                let r: *mut u8 = ring.at::<u8>(row * bytes);
                check(unsafe { sys::cudaMemcpyAsync(r.cast(), save.at::<u8>(k * bytes).cast(), bytes, sys::MEMCPY_D2D, s.0) }, "ring restore")?;
                for p in (end / ratio * ratio).max(start)..end {
                    check(
                        unsafe {
                            sys::cudaMemcpyAsync(r.add((p % ratio) * dim * 2).cast(), rec.at::<u16>((offset + p - start) * dim).cast(),
                                                 dim * 2, sys::MEMCPY_D2D, s.0)
                        },
                        "ring fix",
                    )?;
                }
            }
        }
        if let (Some((conv, hist)), Some(rec), Some(pc), Some(hc)) = (&self.ple_state, &self.ple_record, &self.cfg.ple, &self.cfg.hc) {
            let ple = self.layers[pc.layers[0]].ple.as_ref().unwrap();
            let ch = hc.count * self.cfg.hidden;
            self.commit_meta.upload_at(k * 16, &[0i32, accepted as i32, row as i32])?;
            let dummy = self.work.ple_key.as_ref().unwrap();
            check(
                unsafe {
                    sys::blz_ple_conv(rec.at::<u16>(offset * ch), rec.at::<u16>(offset * ch), self.commit_meta.at(k * 4), self.commit_meta.at(k * 4 + 2), 1, ch as c_int,
                                      pc.conv_kernel as c_int, pc.ngram_size as c_int, ple.conv_w.ptr(), conv.ptr(), 1, dummy.ptr(), s.0)
                },
                "commit ple conv",
            )?;
            let mut all: Vec<i64> = self.hist_save.borrow()[k].to_vec();
            all.extend_from_slice(&tokens[..accepted]);
            s.sync()?;
            hist.upload_at(row * 16, &all[all.len() - 2..])?;
        }
        let Some(gc) = self.cfg.gdn.as_ref() else { return Ok(()) };
        let (w, s) = (&self.work, &self.stream);
        let (kh, vh) = (self.dims.k_heads, self.dims.v_heads);
        let channels = 2 * kh * gc.k_dim + vh * gc.v_dim;
        self.commit_meta.upload_at(k * 16, &[0i32, accepted as i32, row as i32])?;
        let seq_start: *const i32 = self.commit_meta.at(k * 4);
        let slots: *const i32 = self.commit_meta.at(k * 4 + 2);
        for layer in &self.layers {
            let Some(g) = &layer.gdn else { continue };
            let [qkv, gg, beta] = &self.records[layer.slot];
            check(
                unsafe {
                    sys::blz_gdn_conv_ex(qkv.at::<u16>(offset * channels), channels as i64, seq_start, slots, 1, channels as c_int, gc.conv_kernel as c_int,
                                         g.conv_w.ptr(), self.conv_state[layer.slot].ptr(), w.conv.ptr(), 1, s.0)
                },
                "commit conv",
            )?;
            check(
                unsafe {
                    sys::blz_gdn_recurrent_ex(w.conv.ptr(), gg.at::<f32>(offset * vh), beta.at::<f32>(offset * vh), seq_start, slots, 1,
                                              accepted as c_int, kh as c_int,
                                              vh as c_int, gc.k_dim as c_int, gc.v_dim as c_int, self.rec_state[layer.slot].ptr(),
                                              w.core.ptr(), 1, s.0)
                },
                "commit recurrent",
            )?;
        }
        Ok(())
    }

    fn gdn(&self, layer: &Layer, t: usize, seqs: usize, verify: bool) -> Result<()> {
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
        let write = (!verify) as c_int;
        if verify {
            let rec = &self.records[layer.slot][0];
            check(
                unsafe {
                    sys::cudaMemcpy2DAsync(rec.ptr(), channels * 2, w.proj.ptr::<u16>().cast(), ld * 2, channels * 2, t, sys::MEMCPY_D2D, s.0)
                },
                "record qkv",
            )?;
        }
        check(
            unsafe {
                sys::blz_gdn_conv_ex(w.proj.ptr(), ld as i64, self.meta_ptr(m.seq_start), self.meta_ptr(m.seq_slots), seqs as c_int,
                                     channels as c_int, gc.conv_kernel as c_int, g.conv_w.ptr(), self.conv_state[layer.slot].ptr(),
                                     w.conv.ptr(), write, s.0)
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
        if verify {
            let [_, rg, rb] = &self.records[layer.slot];
            check(unsafe { sys::cudaMemcpyAsync(rg.ptr(), w.g.ptr::<f32>().cast(), t * vh * 4, sys::MEMCPY_D2D, s.0) }, "record g")?;
            check(unsafe { sys::cudaMemcpyAsync(rb.ptr(), w.beta.ptr::<f32>().cast(), t * vh * 4, sys::MEMCPY_D2D, s.0) }, "record beta")?;
        }
        check(
            unsafe {
                sys::blz_gdn_recurrent_ex(w.conv.ptr(), w.g.ptr(), w.beta.ptr(), self.meta_ptr(m.seq_start), self.meta_ptr(m.seq_slots),
                                          seqs as c_int, t as c_int, kh as c_int, vh as c_int, gc.k_dim as c_int, gc.v_dim as c_int,
                                          self.rec_state[layer.slot].ptr(), w.core.ptr(), write, s.0)
            },
            "gdn_recurrent",
        )?;
        check(
            unsafe {
                sys::blz_gated_rmsnorm(w.core.ptr(), (vh * gc.v_dim) as i64, z, ld as i64, g.norm.ptr(), cfg.rms_eps,
                                       t as c_int, vh as c_int, gc.v_dim as c_int, gc.sigmoid_gate as c_int, w.core_normed.ptr(), s.0)
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
