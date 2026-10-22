//! Qwen3.5 / 3.6 / 3.8 model graphs with Gated DeltaNet, softmax attention and dense or MoE MLPs.
//! Projections with a shared input are stacked at load time.

use crate::config::{LayerKind, MlpConfig, ModelConfig, RopeScaling};
use crate::gpu::{DevBuf, HostMapped, Stream, check};
use crate::quant::{Format, QuantConfig};
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Context, Result, anyhow, bail};
use blaze_sys as sys;
use std::ffi::{CString, c_int};
use std::path::Path;
use std::ptr::{null, null_mut};

/// A linear layer as one bf16 matrix [n, k] (rows of every stacked module).
pub struct Linear {
    pub w: DevBuf,
    pub n: usize,
    pub k: usize,
}

impl Linear {
    pub fn load(ckpt: &Checkpoint, quant: &QuantConfig, modules: &[String], s: &Stream) -> Result<Self> {
        let mut parts = Vec::with_capacity(modules.len());
        for m in modules {
            let (scheme, names) = quant.resolve(ckpt, m)?;
            let info = ckpt.info(&names.weight)?.clone();
            let n = info.shape[0];
            let k = match scheme.format {
                Format::Nvfp4 { .. } => info.shape[1] * 2,
                _ => info.shape[1],
            };
            parts.push((m.clone(), scheme, names, n, k));
        }
        let k = parts[0].4;
        if parts.iter().any(|p| p.4 != k) {
            bail!("stacked modules {modules:?} have different input sizes");
        }
        let n: usize = parts.iter().map(|p| p.3).sum();
        let w = DevBuf::new(n * k * 2)?;
        let mut row = 0;
        for (m, scheme, names, rows, _) in &parts {
            let dst: *mut u16 = w.at(row * k);
            match &scheme.format {
                Format::Dense(Dtype::Bf16) => w.upload_at(row * k * 2, ckpt.bytes(&names.weight)?)?,
                Format::Dense(d @ (Dtype::F16 | Dtype::F32)) => {
                    let raw = ckpt.bytes(&names.weight)?;
                    let v: Vec<u16> = if *d == Dtype::F16 {
                        raw.chunks_exact(2)
                            .map(|c| half::bf16::from_f32(half::f16::from_le_bytes([c[0], c[1]]).to_f32()).to_bits())
                            .collect()
                    } else {
                        raw.chunks_exact(4).map(|c| half::bf16::from_f32(f32::from_le_bytes(c.try_into().unwrap())).to_bits()).collect()
                    };
                    w.upload_at(row * k * 2, &v)?;
                }
                Format::Fp8Block { block_n, block_k, scale_dtype, inverse } => {
                    if *inverse {
                        bail!("{m}: weight_scale_inv checkpoints are not supported yet");
                    }
                    let scale_name = names.scale.as_ref().unwrap();
                    let (wq, sc) = (DevBuf::from_slice(ckpt.bytes(&names.weight)?)?, DevBuf::from_slice(ckpt.bytes(scale_name)?)?);
                    let is_f32 = match scale_dtype {
                        Dtype::F32 => 1,
                        Dtype::Bf16 => 0,
                        other => bail!("{m}: fp8 block scales in {other:?}"),
                    };
                    check(
                        unsafe {
                            sys::blz_fp8_block_dequant(wq.ptr(), sc.ptr(), is_f32, *rows as c_int, k as c_int, *block_n as c_int,
                                                       *block_k as c_int, dst, s.0)
                        },
                        "blz_fp8_block_dequant",
                    )?;
                    s.sync()?;
                }
                Format::Fp8Channel { per_tensor } => {
                    let scale_name = names.scale.as_ref().unwrap();
                    let si = ckpt.info(scale_name)?;
                    let (wq, sc) = (DevBuf::from_slice(ckpt.bytes(&names.weight)?)?, DevBuf::from_slice(ckpt.bytes(scale_name)?)?);
                    let (bn, is_f32) = (if *per_tensor { *rows } else { 1 }, (si.dtype == Dtype::F32) as c_int);
                    check(
                        unsafe { sys::blz_fp8_block_dequant(wq.ptr(), sc.ptr(), is_f32, *rows as c_int, k as c_int, bn as c_int, k as c_int, dst, s.0) },
                        "blz_fp8_block_dequant",
                    )?;
                    s.sync()?;
                }
                other => bail!("{m}: {} linear layers need the Marlin kernels (not ported yet)", other.describe()),
            }
            row += rows;
        }
        Ok(Self { w, n, k })
    }

    pub fn run(&self, gemm: *mut sys::blz_gemm, x: *const u16, ldx: usize, m: usize, out: *mut u16, ldo: usize, s: &Stream) -> Result<()> {
        check(
            unsafe { sys::blz_gemm_bf16(gemm, x, ldx as i64, self.w.ptr(), out, ldo as i64, m as c_int, self.n as c_int, self.k as c_int, s.0) },
            "blz_gemm_bf16",
        )
    }
}

fn load_vec(ckpt: &Checkpoint, name: &str) -> Result<DevBuf> {
    let t = ckpt.info(name)?;
    if t.dtype != Dtype::Bf16 {
        bail!("{name}: expected bf16, got {:?}", t.dtype);
    }
    DevBuf::from_slice(ckpt.bytes(name)?)
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

pub struct Options {
    pub max_tokens: usize,
    pub max_chunk: usize,
    pub max_seqs: usize,
    pub moe_threads: usize,
    pub moe_heat: Option<String>,
    pub moe_options: String,
    /// VRAM kept free after the experts' slots (activations of cuBLAS, the allocator, the display).
    pub reserve_gib: f64,
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
    logits: DevBuf,
    next: DevBuf,
    tokens: DevBuf,
    positions: DevBuf,
    locs: DevBuf,
    seq_start: DevBuf,
    seq_slots: DevBuf,
    req_idx: DevBuf,
    kv_lens: DevBuf,
    proj_cols: usize,
}

pub struct Model {
    pub cfg: ModelConfig,
    layers: Vec<Layer>,
    final_norm: DevBuf,
    lm_head: Linear,
    embed: HostMapped,
    moe: Option<MoeEngine>,
    gemm: *mut sys::blz_gemm,
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
    pub stream: Stream,
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
    pub fn load(dir: &Path, opts: Options) -> Result<Self> {
        let cfg = ModelConfig::load(dir)?;
        let ckpt = Checkpoint::open(dir)?;
        let root: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let hf_quant = std::fs::read_to_string(dir.join("hf_quant_config.json")).ok().and_then(|s| serde_json::from_str(&s).ok());
        let quant = QuantConfig::from_model_config(&root, hf_quant.as_ref())?;
        let stream = Stream::new()?;
        let s = &stream;
        let p = ckpt.text_prefix();
        let mut gemm = null_mut();
        if unsafe { sys::blz_gemm_create(&mut gemm) } != 0 {
            bail!("cuBLAS init failed");
        }

        let mut layers = Vec::with_capacity(cfg.layers.len());
        let (mut n_full, mut n_gdn) = (0, 0);
        for (i, kind) in cfg.layers.iter().enumerate() {
            let l = format!("{p}layers.{i}.");
            let (attn, gdn, slot) = match kind {
                LayerKind::Full => {
                    let a = format!("{l}self_attn.");
                    let qkv = Linear::load(&ckpt, &quant, &[format!("{a}q_proj"), format!("{a}k_proj"), format!("{a}v_proj")], s)?;
                    let attn = Attn {
                        qkv,
                        o: Linear::load(&ckpt, &quant, &[format!("{a}o_proj")], s)?,
                        q_norm: load_vec(&ckpt, &format!("{a}q_norm.weight"))?,
                        k_norm: load_vec(&ckpt, &format!("{a}k_norm.weight"))?,
                    };
                    n_full += 1;
                    (Some(attn), None, n_full - 1)
                }
                LayerKind::Linear => {
                    let g = format!("{l}linear_attn.");
                    let in_proj = Linear::load(
                        &ckpt,
                        &quant,
                        &[format!("{g}in_proj_qkv"), format!("{g}in_proj_z"), format!("{g}in_proj_b"), format!("{g}in_proj_a")],
                        s,
                    )?;
                    let gdn = Gdn {
                        in_proj,
                        out: Linear::load(&ckpt, &quant, &[format!("{g}out_proj")], s)?,
                        conv_w: load_vec(&ckpt, &format!("{g}conv1d.weight"))?,
                        a_log: load_vec(&ckpt, &format!("{g}A_log"))?,
                        dt_bias: load_vec(&ckpt, &format!("{g}dt_bias"))?,
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
                    let mut gate_mods = vec![format!("{m}gate")];
                    if has_shared_gate {
                        gate_mods.push(format!("{m}shared_expert_gate"));
                    }
                    Mlp::Moe {
                        gate: Linear::load(&ckpt, &quant, &gate_mods, s)?,
                        shared_gu: Linear::load(&ckpt, &quant, &[format!("{m}shared_expert.gate_proj"), format!("{m}shared_expert.up_proj")], s)?,
                        shared_down: Linear::load(&ckpt, &quant, &[format!("{m}shared_expert.down_proj")], s)?,
                        has_shared_gate,
                    }
                }
                MlpConfig::Dense { .. } => Mlp::Dense {
                    gu: Linear::load(&ckpt, &quant, &[format!("{m}gate_proj"), format!("{m}up_proj")], s)?,
                    down: Linear::load(&ckpt, &quant, &[format!("{m}down_proj")], s)?,
                },
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
        let lm_head = Linear::load(&ckpt, &quant, &[lm_name], s)?;

        // Token embedding: pinned host table read by the GPU (it costs VRAM worth of KV cache otherwise).
        let embed_name = format!("{p}embed_tokens.weight");
        let et = ckpt.info(&embed_name)?.clone();
        if et.dtype != Dtype::Bf16 {
            bail!("{embed_name}: expected bf16");
        }
        let mut embed = HostMapped::new(et.bytes())?;
        embed.host_slice_mut().copy_from_slice(ckpt.bytes(&embed_name)?);

        let (freqs, rope_scale) = inv_freq(&cfg);
        let inv_freq = DevBuf::from_slice(&freqs)?;

        // KV pools (NVFP4) for every softmax layer, GDN states for every linear layer.
        let (kvh, hd) = (cfg.attn.kv_heads, cfg.attn.head_dim);
        let mut kv = Vec::new();
        for _ in 0..n_full {
            let codes = opts.max_tokens * kvh * hd / 2;
            let scales = opts.max_tokens * kvh * hd / 16;
            kv.push([DevBuf::zeroed(codes, s)?, DevBuf::zeroed(codes, s)?, DevBuf::zeroed(scales, s)?, DevBuf::zeroed(scales, s)?]);
        }
        let kv_global = DevBuf::from_slice(&[1.0f32])?;
        let req_to_token = DevBuf::zeroed(opts.max_seqs * cfg.max_positions.min(opts.max_tokens) * 4, s)?;
        let (mut conv_state, mut rec_state) = (Vec::new(), Vec::new());
        if let Some(g) = &cfg.gdn {
            let channels = 2 * g.k_heads * g.k_dim + g.v_heads * g.v_dim;
            for _ in 0..n_gdn {
                conv_state.push(DevBuf::zeroed(opts.max_seqs * channels * (g.conv_kernel - 1) * 2, s)?);
                rec_state.push(DevBuf::zeroed(opts.max_seqs * g.v_heads * g.k_dim * g.v_dim * 4, s)?);
            }
        }

        let work = Self::alloc_work(&cfg, &layers, &opts, s)?;
        s.sync()?;

        let mut num_sms: c_int = 0;
        unsafe {
            // cudaDevAttrMultiProcessorCount = 16
            unsafe extern "C" {
                fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
            }
            cudaDeviceGetAttribute(&mut num_sms, 16, 0);
        }

        let mut model = Self {
            cfg,
            layers,
            final_norm,
            lm_head,
            embed,
            moe: None,
            gemm,
            inv_freq,
            rope_scale,
            kv,
            kv_global,
            req_to_token,
            conv_state,
            rec_state,
            work,
            stream,
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

    fn alloc_work(cfg: &ModelConfig, layers: &[Layer], opts: &Options, s: &Stream) -> Result<Work> {
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
        let (heads, hd) = (cfg.attn.heads, cfg.attn.head_dim);
        let (conv_ch, vh, vd) = match &cfg.gdn {
            Some(g) => (2 * g.k_heads * g.k_dim + g.v_heads * g.v_dim, g.v_heads, g.v_dim),
            None => (0, 0, 0),
        };
        let topk = match cfg.mlp {
            MlpConfig::Moe { top_k, .. } => top_k,
            _ => 0,
        };
        let inter = match cfg.mlp {
            MlpConfig::Moe { shared_intermediate, .. } => shared_intermediate,
            MlpConfig::Dense { intermediate } => intermediate,
        };
        let parts_prefill = unsafe { sys::blz_nvfp4_mha_parts(heads as c_int, cfg.attn.kv_heads as c_int, t as c_int, 1) } as usize;
        let parts_decode = unsafe { sys::blz_nvfp4_mha_parts(heads as c_int, cfg.attn.kv_heads as c_int, 1, 64) } as usize;
        let partial_rows = (t * parts_prefill).max(opts.max_seqs * parts_decode);
        let b = |n: usize| DevBuf::zeroed(n, s);
        Ok(Work {
            residual: b(t * h * 2)?,
            normed: b(t * h * 2)?,
            x: b(t * h * 2)?,
            proj: b(t * proj_cols * 2)?,
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
            logits: b(opts.max_seqs * cfg.vocab * 2)?,
            next: b(opts.max_seqs * 4)?,
            tokens: b(t * 8)?,
            positions: b(t * 8)?,
            locs: b(t * 8)?,
            seq_start: b((opts.max_seqs + 1) * 4)?,
            seq_slots: b(opts.max_seqs * 4)?,
            req_idx: b(opts.max_seqs * 4)?,
            kv_lens: b(opts.max_seqs * 4)?,
            proj_cols,
        })
    }

    fn ctx_stride(&self) -> usize {
        self.cfg.max_positions.min(self.opts.max_tokens)
    }

    /// Forward of `tokens` (one sequence, positions start..start + len) with KV request row `req` and GDN state slot
    /// `seq_slot`; returns the greedy next token. A new sequence (start == 0) resets its GDN state.
    pub fn forward_seq(&mut self, tokens: &[i64], start: usize, req: usize, seq_slot: usize) -> Result<i32> {
        let t = tokens.len();
        if t == 0 || t > self.opts.max_chunk {
            bail!("forward of {t} tokens (max chunk {})", self.opts.max_chunk);
        }
        if start + t > self.ctx_stride() {
            bail!("context of {} tokens exceeds the KV pool ({})", start + t, self.ctx_stride());
        }
        let s = &self.stream;
        let cfg = &self.cfg;
        let h = cfg.hidden;
        let w = &self.work;
        if start == 0 {
            if let Some(g) = &cfg.gdn {
                let channels = 2 * g.k_heads * g.k_dim + g.v_heads * g.v_dim;
                let conv_bytes = channels * (g.conv_kernel - 1) * 2;
                let rec_bytes = g.v_heads * g.k_dim * g.v_dim * 4;
                for (c, r) in self.conv_state.iter().zip(&self.rec_state) {
                    check(unsafe { sys::cudaMemsetAsync(c.at::<u8>(seq_slot * conv_bytes).cast(), 0, conv_bytes, s.0) }, "memset")?;
                    check(unsafe { sys::cudaMemsetAsync(r.at::<u8>(seq_slot * rec_bytes).cast(), 0, rec_bytes, s.0) }, "memset")?;
                }
            }
        }
        // Metadata: one sequence, KV slot of position p = req * stride + p (identity pages for now).
        let positions: Vec<i64> = (start..start + t).map(|p| p as i64).collect();
        let base = (req * self.ctx_stride()) as i64;
        let locs: Vec<i64> = positions.iter().map(|p| base + p).collect();
        let row: Vec<i32> = (0..(start + t) as i32).map(|p| base as i32 + p).collect();
        w.tokens.upload_async(tokens, s)?;
        w.positions.upload_async(&positions, s)?;
        w.locs.upload_async(&locs, s)?;
        w.seq_start.upload_async(&[0i32, t as i32], s)?;
        w.seq_slots.upload_async(&[seq_slot as i32], s)?;
        w.req_idx.upload_async(&[req as i32], s)?;
        w.kv_lens.upload_async(&[(start + t) as i32], s)?;
        check(
            unsafe {
                sys::cudaMemcpyAsync(
                    self.req_to_token.at::<i32>(req * self.ctx_stride()).cast(),
                    row.as_ptr().cast(),
                    row.len() * 4,
                    sys::MEMCPY_H2D,
                    s.0,
                )
            },
            "req_to_token",
        )?;
        s.sync()?; // the host vectors above are dropped at return

        // Embedding into the residual stream.
        check(
            unsafe {
                sys::blz_embed_gather(w.tokens.ptr(), t as i64, self.embed.dev_ptr(), cfg.vocab as i64, (h * 2) as i64,
                                      w.residual.ptr(), s.0)
            },
            "embed",
        )?;
        let eps = cfg.rms_eps;
        for (li, layer) in self.layers.iter().enumerate() {
            // input norm: the first layer normalizes the embedding, later ones add the previous MLP output first
            let resid = if li == 0 { null_mut() } else { w.residual.ptr() };
            let x_in: *const u16 = if li == 0 { w.residual.ptr() } else { w.x.ptr() };
            check(unsafe { sys::blz_rmsnorm(x_in, resid, layer.input_norm.ptr(), eps, t as c_int, h as c_int, 1, w.normed.ptr(), s.0) }, "rmsnorm")?;
            match layer.kind {
                LayerKind::Full => self.attention(layer, t, start)?,
                LayerKind::Linear => self.gdn(layer, t, seq_slot)?,
            }
            check(unsafe { sys::blz_rmsnorm(w.x.ptr(), w.residual.ptr(), layer.post_norm.ptr(), eps, t as c_int, h as c_int, 1, w.normed.ptr(), s.0) }, "rmsnorm")?;
            self.mlp(li, layer, t)?;
        }
        // Final norm of the last token only, then the head and the greedy pick.
        let last_x: *const u16 = w.x.at(( t - 1) * h);
        let last_res: *mut u16 = w.residual.at((t - 1) * h);
        check(unsafe { sys::blz_rmsnorm(last_x, last_res, self.final_norm.ptr(), eps, 1, h as c_int, 1, w.normed.ptr(), s.0) }, "final norm")?;
        self.lm_head.run(self.gemm, w.normed.ptr(), h, 1, w.logits.ptr(), cfg.vocab, s)?;
        check(unsafe { sys::blz_argmax_bf16(w.logits.ptr(), 1, cfg.vocab as c_int, cfg.vocab as i64, w.next.ptr(), s.0) }, "argmax")?;
        s.sync()?;
        Ok(w.next.download::<i32>(1)?[0])
    }

    pub fn last_logits(&self) -> Result<Vec<f32>> {
        let raw = self.work.logits.download::<u16>(self.cfg.vocab)?;
        Ok(raw.into_iter().map(|b| half::bf16::from_bits(b).to_f32()).collect())
    }

    fn attention(&self, layer: &Layer, t: usize, start: usize) -> Result<()> {
        let a = layer.attn.as_ref().unwrap();
        let (cfg, w, s) = (&self.cfg, &self.work, &self.stream);
        let (heads, kvh, hd) = (cfg.attn.heads, cfg.attn.kv_heads, cfg.attn.head_dim);
        let ld = a.qkv.n;
        a.qkv.run(self.gemm, w.normed.ptr(), cfg.hidden, t, w.proj.ptr(), ld, s)?;
        let k: *mut u16 = w.proj.at(heads * 2 * hd);
        let v: *const u16 = w.proj.at(heads * 2 * hd + kvh * hd);
        check(
            unsafe {
                sys::blz_attn_prep(w.proj.ptr(), ld as i64, k, ld as i64, w.positions.ptr(), a.q_norm.ptr(), a.k_norm.ptr(),
                                   cfg.rms_eps, 1, t as c_int, heads as c_int, kvh as c_int, hd as c_int,
                                   cfg.rope.rotary_dim as c_int, self.inv_freq.ptr(), self.rope_scale, w.q.ptr(),
                                   w.gate.ptr(), s.0)
            },
            "attn_prep",
        )?;
        let pool = &self.kv[layer.slot];
        check(
            unsafe {
                sys::blz_nvfp4_kv_store(k.cast(), v.cast(), ld as i64, ld as i64, sys::BLZ_BF16, w.locs.ptr::<i64>().cast(),
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
        let splits = if t == 1 { (4 * self.num_sms / kvh).clamp(1, 64) } else { 1 };
        let _ = start;
        check(
            unsafe {
                sys::blz_nvfp4_mha_decode(
                    w.q.ptr::<u16>().cast(),
                    (heads * hd) as i64,
                    hd as i64,
                    heads as c_int,
                    view,
                    self.req_to_token.ptr(),
                    self.ctx_stride() as i64,
                    sys::blz_index_ptr { ptr: w.req_idx.ptr::<i32>().cast(), kind: sys::BLZ_I32 },
                    sys::blz_index_ptr { ptr: w.kv_lens.ptr::<i32>().cast(), kind: sys::BLZ_I32 },
                    1,
                    t as c_int,
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
            "mha",
        )?;
        check(unsafe { sys::blz_mul_sigmoid(w.attn.ptr(), w.gate.ptr(), (t * heads * hd) as i64, s.0) }, "gate")?;
        a.o.run(self.gemm, w.attn.ptr(), heads * hd, t, w.x.ptr(), cfg.hidden, s)
    }

    fn gdn(&self, layer: &Layer, t: usize, seq_slot: usize) -> Result<()> {
        let g = layer.gdn.as_ref().unwrap();
        let gc = self.cfg.gdn.as_ref().unwrap();
        let (cfg, w, s) = (&self.cfg, &self.work, &self.stream);
        let ld = g.in_proj.n;
        g.in_proj.run(self.gemm, w.normed.ptr(), cfg.hidden, t, w.proj.ptr(), ld, s)?;
        let key_dim = gc.k_heads * gc.k_dim;
        let channels = 2 * key_dim + gc.v_heads * gc.v_dim;
        let z: *const u16 = w.proj.at(channels);
        let b: *const u16 = w.proj.at(channels + gc.v_heads * gc.v_dim);
        let a: *const u16 = w.proj.at(channels + gc.v_heads * gc.v_dim + gc.v_heads);
        let _ = seq_slot;
        check(
            unsafe {
                sys::blz_gdn_conv(w.proj.ptr(), ld as i64, w.seq_start.ptr(), w.seq_slots.ptr(), 1, channels as c_int,
                                  gc.conv_kernel as c_int, g.conv_w.ptr(), self.conv_state[layer.slot].ptr(), w.conv.ptr(), s.0)
            },
            "gdn_conv",
        )?;
        check(
            unsafe {
                sys::blz_gdn_gate(a, ld as i64, b, ld as i64, g.a_log.ptr(), g.dt_bias.ptr(), t as c_int, gc.v_heads as c_int,
                                  w.g.ptr(), w.beta.ptr(), s.0)
            },
            "gdn_gate",
        )?;
        check(
            unsafe {
                sys::blz_gdn_recurrent(w.conv.ptr(), w.g.ptr(), w.beta.ptr(), w.seq_start.ptr(), w.seq_slots.ptr(), 1, t as c_int,
                                       gc.k_heads as c_int, gc.v_heads as c_int, gc.k_dim as c_int, gc.v_dim as c_int,
                                       self.rec_state[layer.slot].ptr(), w.core.ptr(), s.0)
            },
            "gdn_recurrent",
        )?;
        check(
            unsafe {
                sys::blz_gated_rmsnorm(w.core.ptr(), (gc.v_heads * gc.v_dim) as i64, z, ld as i64, g.norm.ptr(), cfg.rms_eps,
                                       t as c_int, gc.v_heads as c_int, gc.v_dim as c_int, w.core_normed.ptr(), s.0)
            },
            "gated_rmsnorm",
        )?;
        g.out.run(self.gemm, w.core_normed.ptr(), gc.v_heads * gc.v_dim, t, w.x.ptr(), cfg.hidden, s)
    }

    fn mlp(&self, li: usize, layer: &Layer, t: usize) -> Result<()> {
        let (cfg, w, s) = (&self.cfg, &self.work, &self.stream);
        let h = cfg.hidden;
        match &layer.mlp {
            Mlp::Moe { gate, shared_gu, shared_down, has_shared_gate } => {
                let MlpConfig::Moe { experts, top_k, norm_topk, shared_intermediate, .. } = cfg.mlp else { unreachable!() };
                gate.run(self.gemm, w.normed.ptr(), h, t, w.proj.ptr(), gate.n, s)?;
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
                shared_gu.run(self.gemm, w.normed.ptr(), h, t, w.gu.ptr(), gu_ld, s)?;
                let up: *const u16 = w.gu.at(shared_intermediate);
                check(
                    unsafe {
                        sys::blz_silu_mul(w.gu.ptr(), gu_ld as i64, up, gu_ld as i64, t as c_int, shared_intermediate as c_int,
                                          w.shared_act.ptr(), s.0)
                    },
                    "silu_mul",
                )?;
                shared_down.run(self.gemm, w.shared_act.ptr(), shared_intermediate, t, w.shared.ptr(), h, s)?;
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
                gu.run(self.gemm, w.normed.ptr(), h, t, w.proj.ptr(), gu.n, s)?;
                let up: *const u16 = w.proj.at(inter);
                check(
                    unsafe { sys::blz_silu_mul(w.proj.ptr(), gu.n as i64, up, gu.n as i64, t as c_int, inter as c_int, w.shared_act.ptr(), s.0) },
                    "silu_mul",
                )?;
                down.run(self.gemm, w.shared_act.ptr(), inter, t, w.x.ptr(), h, s)
            }
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        self.moe = None;
        unsafe { sys::blz_gemm_destroy(self.gemm) };
    }
}

pub fn describe_load(path: &Path) -> Result<String> {
    let cfg = ModelConfig::load(path).with_context(|| format!("loading {}", path.display()))?;
    Ok(format!("{} layers, hidden {}", cfg.layers.len(), cfg.hidden))
}
