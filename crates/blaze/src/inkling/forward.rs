use super::{KvDtype, cpu};
use super::cuda::{AttentionScratch, Compute, Dense, Kv, expert_size, from_bf16, to_bf16};
use super::generate::Input;
use super::vision::Vision;
use super::weights::{Experts, read_float};
use crate::config::{InklingLoop, MlpConfig, ModelConfig};
use crate::gpu;
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Context, Result, ensure};
use std::path::Path;

const PREFIX: &str = "model.llm";
const SHORT_KV_TOKENS: usize = 8192;

struct Shape {
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    relative: usize,
    extent: usize,
    conv: usize,
    block: usize,
    vocab: usize,
    unpadded_vocab: usize,
    eps: f32,
    qk_clip: f32,
    rel_clip: f32,
    log_floor: f32,
    log_alpha: f32,
    top_k: usize,
    experts: usize,
    intermediate: usize,
    route_scale: f32,
    gate_cap: f32,
    up_cap: f32,
    logit_cap: f32,
}

impl Shape {
    fn load(cfg: &ModelConfig, loops: &InklingLoop) -> Result<Self> {
        let get = |k: &str| {
            cfg.raw[k]
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v > 0)
                .with_context(|| format!("Inkling: expected positive integer {k}"))
        };
        let float = |k: &str| {
            cfg.raw[k]
                .as_f64()
                .map(|v| v as f32)
                .filter(|v| v.is_finite() && *v > 0.0)
                .with_context(|| format!("Inkling: expected positive number {k}"))
        };
        let MlpConfig::Moe {
            experts,
            top_k,
            intermediate,
            ..
        } = cfg.mlp
        else {
            anyhow::bail!("Inkling requires routed experts");
        };
        let (heads, kv_heads, dim) = (cfg.attn.heads, cfg.attn.kv_heads, cfg.attn.head_dim);
        ensure!(
            heads > 0 && kv_heads > 0 && heads % kv_heads == 0 && dim == 128,
            "Inkling requires grouped 128-dimensional heads"
        );
        for (key, expected) in [
            ("swa_num_attention_heads", heads),
            ("swa_num_key_value_heads", kv_heads),
            ("swa_head_dim", dim),
        ] {
            ensure!(
                cfg.raw[key].as_u64().is_none_or(|n| n == expected as u64),
                "Inkling: mismatched {key}"
            );
        }
        let block = get("n_layers_per_attn_res_block")?;
        ensure!(
            loops.core.start > 0
                && loops.core.start % block == 0
                && loops.core.end % block == 0
                && loops.core.len() >= block
                && !loops.variable,
            "Inkling: unsupported loop/block layout"
        );
        let unpadded_vocab = get("unpadded_vocab_size")?;
        ensure!(
            unpadded_vocab <= cfg.vocab,
            "Inkling: unpadded vocabulary exceeds padded vocabulary"
        );
        let extent = get("rel_extent")?;
        ensure!(
            extent == loops.window,
            "Inkling: different global and local relative extents are unsupported"
        );
        ensure!(cfg.rms_eps.is_finite() && cfg.rms_eps > 0.0, "Inkling: invalid RMS epsilon");
        Ok(Self {
            hidden: cfg.hidden,
            heads,
            kv_heads,
            dim,
            relative: get("d_rel")?,
            extent,
            conv: get("sconv_kernel_size")?,
            block,
            vocab: cfg.vocab,
            unpadded_vocab,
            eps: cfg.rms_eps,
            qk_clip: float("qk_l2_clip_threshold")?,
            rel_clip: float("rel_l2_clip_threshold")?,
            log_floor: float("log_scaling_n_floor")?,
            log_alpha: float("log_scaling_alpha")?,
            top_k,
            experts,
            intermediate,
            route_scale: float("route_scale")?,
            gate_cap: float("mlp_gate_softcap")?,
            up_cap: float("mlp_up_softcap")?,
            logit_cap: float("final_logit_softcapping")?,
        })
    }

    fn q_width(&self) -> usize {
        self.heads * self.dim
    }
    fn kv_width(&self) -> usize {
        self.kv_heads * self.dim
    }
    fn rel_width(&self) -> usize {
        self.heads * self.relative
    }
}

struct Conv {
    // Taps are stored in lag order so each past row is contiguous.
    taps: Vec<Vec<f32>>,
    dynamic: Option<Dense>,
    scale: f32,
}

impl Conv {
    fn load(ckpt: &Checkpoint, prefix: &str, width: usize, s: &Shape, dynamic: bool) -> Result<Self> {
        let k = s.conv;
        let (taps, proj, scale) = if dynamic {
            let weights = read_float(ckpt, &format!("{prefix}.static_w"), &[k, width])?;
            let proj = Dense::load(ckpt, &format!("{prefix}.weight_proj"), s.kv_heads * k, s.hidden)?;
            (
                weights.chunks_exact(width).map(<[f32]>::to_vec).collect(),
                Some(proj),
                ((2 * k + 1) as f32).sqrt().recip(),
            )
        } else {
            let w = read_float(ckpt, &format!("{prefix}.weight"), &[width, 1, k])?;
            let taps = (0..k)
                .map(|lag| w.chunks_exact(k).map(|row| row[k - lag - 1]).collect())
                .collect();
            (taps, None, ((k + 1) as f32).sqrt().recip())
        };
        Ok(Self {
            taps,
            dynamic: proj,
            scale,
        })
    }

    fn run(
        &self,
        gpu: &mut Compute,
        state: &mut ConvState,
        x: &[f32],
        input: &[f32],
        pos: usize,
        dim: usize,
    ) -> Result<Vec<f32>> {
        let k = self.taps.len();
        let dyn_weights = self.dynamic.as_ref().map(|m| gpu.linear(m, input)).transpose()?;
        let mut out = x.to_vec();
        for lag in 0..k.min(pos + 1) {
            let src = if lag == 0 { x } else { &state.rows[(pos - lag) % (k - 1)] };
            let taps = &self.taps[lag];
            if let Some(d) = &dyn_weights {
                for (head, ((out, src), taps)) in out
                    .chunks_exact_mut(dim)
                    .zip(src.chunks_exact(dim))
                    .zip(taps.chunks_exact(dim))
                    .enumerate()
                {
                    let extra = d[head * k + lag];
                    for ((o, &v), &w) in out.iter_mut().zip(src).zip(taps) {
                        *o += v * (w + extra);
                    }
                }
            } else {
                for ((o, &v), &w) in out.iter_mut().zip(src).zip(taps) {
                    *o += v * w;
                }
            }
        }
        if k > 1 {
            state.rows[pos % (k - 1)].copy_from_slice(x);
        }
        out.iter_mut().for_each(|v| *v = round(*v * self.scale));
        Ok(out)
    }
}

struct ConvState {
    rows: Vec<Vec<f32>>,
}

impl ConvState {
    fn new(taps: usize, width: usize) -> Self {
        Self {
            rows: vec![vec![0.0; width]; taps - 1],
        }
    }
}

struct Controller {
    down: Dense,
    up: GateWeights,
}

enum GateWeights {
    Host { forget: Vec<f32>, gain: Vec<f32> },
    Device(Dense),
}

impl Controller {
    fn load(ckpt: &Checkpoint, name: &str, hidden: usize, rank: usize) -> Result<Self> {
        let forget = format!("{name}.forget_weight");
        let gain = format!("{name}.gain_weight");
        let up = if rank == 64 && std::env::var_os("BLAZE_INKLING_GPU_CONTROLLER").is_some_and(|v| v == "1")
            && ckpt.info(&forget)?.dtype == Dtype::Bf16 && ckpt.info(&gain)?.dtype == Dtype::Bf16
        {
            GateWeights::Device(Dense::stack(ckpt, &[(&forget, hidden), (&gain, hidden)], rank)?)
        } else {
            GateWeights::Host {
                forget: read_float(ckpt, &forget, &[hidden, rank])?,
                gain: read_float(ckpt, &gain, &[hidden, rank])?,
            }
        };
        Ok(Self {
            down: Dense::load(ckpt, &format!("{name}.down_weight"), rank, hidden * 2)?,
            up,
        })
    }

    fn input(x: &[f32], message: &[f32], eps: f32) -> Vec<f32> {
        normalized(x, None, eps)
            .into_iter()
            .chain(normalized(message, None, eps))
            .collect()
    }

    fn apply_host(x: &mut [f32], message: &[f32], h: &[f32], forget: &[f32], gain: &[f32]) {
        let forget: Vec<_> = forget.chunks_exact(h.len()).map(|row| dot(row, h)).collect();
        let gain: Vec<_> = gain.chunks_exact(h.len()).map(|row| dot(row, h)).collect();
        cpu::controller_mix(x, message, &forget, &gain);
    }

    fn merge(&self, gpu: &mut Compute, x: &mut [f32], message: &[f32], eps: f32) -> Result<()> {
        let input = Self::input(x, message, eps);
        match &self.up {
            GateWeights::Device(up) => {
                let gates = gpu.controller_gates([&self.down], [up], [&input])?;
                let (forget, gain) = gates.split_at(x.len());
                cpu::controller_mix(x, message, forget, gain);
            }
            GateWeights::Host { forget, gain } => {
                let h = gpu.linear(&self.down, &input)?;
                Self::apply_host(x, message, &h, forget, gain);
            }
        }
        Ok(())
    }
}

struct LoopInputs {
    queries: [Vec<f32>; 4],
    controllers: [Controller; 4],
}

impl LoopInputs {
    fn mix_attention(&self, gpu: &mut Compute, qkv: &mut [Vec<f32>], memory: &[Vec<f32>], eps: f32) -> Result<()> {
        let messages: [Vec<f32>; 3] = std::array::from_fn(|i| mix_normalized(memory, &self.queries[i]));
        let controllers = self.controllers.first_chunk::<3>().unwrap();
        let weights = controllers.each_ref().map(|c| &c.up);
        if let [GateWeights::Device(q), GateWeights::Device(k), GateWeights::Device(v)] = weights {
            let inputs: [Vec<f32>; 3] = std::array::from_fn(|i| Controller::input(&qkv[i], &messages[i], eps));
            let gates = gpu.controller_gates(controllers.each_ref().map(|c| &c.down), [q, k, v], inputs.each_ref().map(Vec::as_slice))?;
            let width = qkv[0].len();
            for ((x, message), gates) in qkv.iter_mut().zip(messages).zip(gates.chunks_exact(2 * width)) {
                let (forget, gain) = gates.split_at(width);
                cpu::controller_mix(x, &message, forget, gain);
            }
            return Ok(());
        }
        if weights.iter().any(|w| matches!(w, GateWeights::Device(_))) {
            for ((controller, x), message) in controllers.iter().zip(qkv).zip(messages) {
                controller.merge(gpu, x, &message, eps)?;
            }
            return Ok(());
        }
        let inputs: [Vec<f32>; 3] = std::array::from_fn(|i| Controller::input(&qkv[i], &messages[i], eps));
        let inputs = inputs.each_ref().map(Vec::as_slice).map(|row| [row]);
        let gates = gpu.project_three(
            std::array::from_fn(|i| &self.controllers[i].down),
            inputs.each_ref().map(|row| row.as_slice()),
        )?;
        for (i, gate) in gates.into_iter().enumerate() {
            let h: Vec<_> = gate.into_iter().map(from_bf16).collect();
            let GateWeights::Host { forget, gain } = weights[i] else { unreachable!() };
            Controller::apply_host(&mut qkv[i], &messages[i], &h, forget, gain);
        }
        Ok(())
    }
}

enum Router {
    Host(Vec<f32>),
    Device(Dense),
}

struct Layer {
    local: bool,
    query: Dense,
    key: Dense,
    value: Dense,
    output: Dense,
    bias: Dense,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    norm_v: Vec<f32>,
    norm_mlp: Vec<f32>,
    q_clip: Vec<f32>,
    rel_clip: Vec<f32>,
    res_query: Vec<f32>,
    res_mlp: Vec<f32>,
    loop_inputs: Option<LoopInputs>,
    k_conv: Conv,
    v_conv: Conv,
    attn_conv: Conv,
    mlp_conv: Conv,
    router: Router,
    route_bias: Vec<f32>,
    experts: Experts,
}

impl Layer {
    fn load(
        ckpt: &Checkpoint,
        cfg: &ModelConfig,
        loops: &InklingLoop,
        s: &Shape,
        id: usize,
        messages: (&[f32], &[f32]),
    ) -> Result<Self> {
        let prefix = format!("{PREFIX}.layers.{id}");
        let t = |name: &str| format!("{prefix}.{name}");
        let local = loops.is_local(id);
        let q = s.q_width();
        let kv = s.kv_width();
        let read = |name: &str, shape: &[usize]| read_float(ckpt, &t(name), shape);
        let dense = |name: &str, rows: usize, cols: usize| Dense::load(ckpt, &t(name), rows, cols);
        let attn_res = format!("{PREFIX}.attn_res");
        let res_query = if id == 0 {
            Vec::new()
        } else {
            let shape = if local { vec![s.hidden] } else { vec![3, s.hidden] };
            read_float(ckpt, &format!("{attn_res}.attn_queries.{}", id - 1), &shape)?
        };
        let loop_inputs = if loops.core.contains(&id) {
            let at = (id - loops.core.start) * 2 * s.hidden;
            let row = |data: &[f32], row: usize| data[at + row * s.hidden..at + (row + 1) * s.hidden].to_vec();
            let controller = |name| Controller::load(ckpt, &t(name), s.hidden, loops.gate_rank);
            Some(LoopInputs {
                queries: [row(messages.0, 0), row(messages.1, 0), row(messages.1, 1), row(messages.0, 1)],
                controllers: [
                    controller("controllers.attn")?,
                    controller("controllers.k")?,
                    controller("controllers.v")?,
                    controller("controllers.mlp")?,
                ],
            })
        } else {
            None
        };
        let rel = read("attn.rel_logits_proj.proj", &[s.relative, s.extent])?;
        let mut rel_transposed = vec![0u16; rel.len()];
        for (i, row) in rel.chunks_exact(s.extent).enumerate() {
            for (j, &v) in row.iter().enumerate() {
                rel_transposed[j * s.relative + i] = to_bf16(v);
            }
        }

        Ok(Self {
            local,
            query: Dense::stack(
                ckpt,
                &[
                    (&t("attn.wq_du.weight"), q),
                    (&t("attn.gate_attn.weight"), q),
                    (&t("attn.wr_du.weight"), s.rel_width()),
                    (&t("attn.attn_sink_proj.weight"), s.heads),
                ],
                s.hidden,
            )?,
            key: dense("attn.wk_dv.weight", kv, s.hidden)?,
            value: dense("attn.wv_dv.weight", kv, s.hidden)?,
            output: dense("attn.wo_ud.weight", s.hidden, q)?,
            bias: Dense::from_bf16(&rel_transposed, s.extent, s.relative)?,
            norm_q: read(if local { "attn_norm.weight" } else { "q_input_norm.weight" }, &[s.hidden])?,
            norm_k: if local {
                Vec::new()
            } else {
                read("k_input_norm.weight", &[s.hidden])?
            },
            norm_v: if local {
                Vec::new()
            } else {
                read("v_input_norm.weight", &[s.hidden])?
            },
            norm_mlp: read("mlp_norm.weight", &[s.hidden])?,
            q_clip: read("attn.q_l2_clip.scale", &[s.heads, s.dim])?,
            rel_clip: read("attn.rel_l2_clip.scale", &[s.heads, s.relative])?,
            res_query,
            res_mlp: read_float(ckpt, &format!("{attn_res}.mlp_queries.{id}"), &[s.hidden])?,
            loop_inputs,
            k_conv: Conv::load(ckpt, &t("attn.k_sconv"), kv, s, !local)?,
            v_conv: Conv::load(ckpt, &t("attn.v_sconv"), kv, s, !local)?,
            attn_conv: Conv::load(ckpt, &t("attn_sconv"), s.hidden, s, false)?,
            mlp_conv: Conv::load(ckpt, &t("mlp_sconv"), s.hidden, s, false)?,
            router: if s.hidden.is_multiple_of(32) && ckpt.info(&t("mlp.gate.weight"))?.dtype == Dtype::Bf16
                && std::env::var_os("BLAZE_INKLING_GPU_ROUTER").is_some_and(|v| v == "1")
            {
                Router::Device(dense("mlp.gate.weight", s.experts, s.hidden)?)
            } else {
                Router::Host(read("mlp.gate.weight", &[s.experts, s.hidden])?)
            },
            route_bias: read("mlp.gate.bias", &[s.experts])?,
            experts: Experts::load(ckpt, cfg, id)?,
        })
    }

    fn routes(&self, logits: impl Iterator<Item = f32>, s: &Shape) -> Vec<(usize, f32)> {
        let mut routes: Vec<_> = logits
            .zip(&self.route_bias)
            .enumerate()
            .map(|(id, (logit, &bias))| {
                let score = (1.0 + (-logit).exp()).recip();
                (id, score, score + bias)
            })
            .collect();
        let by_score = |a: &(usize, f32, f32), b: &(usize, f32, f32)| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0));
        if s.top_k < routes.len() {
            routes.select_nth_unstable_by(s.top_k, by_score);
        }
        routes.truncate(s.top_k);
        routes.sort_unstable_by_key(|x| x.0);
        let scale = s.route_scale / (routes.iter().map(|x| x.1).sum::<f32>() + 1e-20);
        routes.into_iter().map(|(id, score, _)| (id, score * scale)).collect()
    }
}

struct VisitState {
    kv: Kv,
    key: ConvState,
    value: ConvState,
    attention: ConvState,
    mlp: ConvState,
}

#[derive(Clone, Copy)]
pub(super) enum Readout {
    None,
    Greedy,
    Scores,
}

pub(super) enum Prediction {
    Token(u32),
    Scores(Vec<f32>),
}

pub struct Sequence {
    position: usize,
    capacity: usize,
    visits: Vec<VisitState>,
}

impl Sequence {
    pub fn len(&self) -> usize {
        self.position
    }
}

pub struct Options {
    pub device: i32,
    pub gpus: usize,
    pub context: usize,
    pub expert_cache_mib: usize,
    pub expert_threads: Option<usize>,
    pub host_kv: bool,
    pub kv_dtype: KvDtype,
    pub max_image_tokens: usize,
    pub prefill_chunk: usize,
}

struct Device {
    id: i32,
    compute: Compute,
    scratch: AttentionScratch,
}

impl Drop for Device {
    fn drop(&mut self) {
        let _ = gpu::set_device(self.id);
    }
}

pub struct Model {
    ckpt: Checkpoint,
    cfg: Shape,
    loops: InklingLoop,
    layers: Vec<Layer>,
    head: Dense,
    norm_embed: Vec<f32>,
    norm_final: Vec<f32>,
    query_final: Vec<f32>,
    devices: Vec<Device>,
    placement: Vec<usize>,
    context: usize,
    global_kv_host: bool,
    kv_dtype: KvDtype,
    vision: Option<Vision>,
    prefill_chunk: usize,
}

impl Model {
    pub fn load(path: &Path, opts: Options) -> Result<Self> {
        let Options {
            device,
            gpus,
            context,
            expert_cache_mib,
            expert_threads,
            host_kv,
            kv_dtype,
            max_image_tokens,
            prefill_chunk,
        } = opts;
        ensure!(
            (1..=2).contains(&gpus) && device >= 0 && device <= i32::MAX - 1,
            "Inkling requires one or two contiguous CUDA devices"
        );
        ensure!(
            (1..=1024).contains(&prefill_chunk),
            "Inkling prefill chunk must be in 1..=1024"
        );
        let root = ModelConfig::read_json(path)?;
        let cfg = ModelConfig::from_json(&root)?;
        let loops = cfg.inkling_loop.as_ref().context("expected an Inkling 2 loop checkpoint")?;
        ensure!(
            context > 1 && context <= cfg.max_positions && context < i32::MAX as usize,
            "context must be within 2..={}",
            cfg.max_positions
        );
        let ckpt = Checkpoint::open(path)?;
        let s = Shape::load(&cfg, loops)?;
        let embed = ckpt.info("model.llm.embed.weight")?;
        ensure!(
            embed.dtype == Dtype::Bf16 && embed.shape == [s.vocab, s.hidden],
            "Inkling: invalid embedding table"
        );
        let msg_shape = [loops.core.len() * 2, s.hidden];
        let q_msg = read_float(&ckpt, "model.llm.middle_cycle_cross_pass_queries", &msg_shape)?;
        let kv_msg = read_float(&ckpt, "model.llm.middle_cycle_cross_pass_kv_queries", &msg_shape)?;
        // Keep the entire recurrent core on one GPU. With two devices, the prelude and coda share the other GPU
        // and forward has only two device transitions, independently of the number of loop passes.
        let placement: Vec<_> = (0..cfg.layers.len())
            .map(|id| usize::from(gpus == 2 && !loops.core.contains(&id)))
            .collect();
        let layers = (0..cfg.layers.len())
            .map(|id| {
                gpu::set_device(device + placement[id] as i32)?;
                Layer::load(&ckpt, &cfg, loops, &s, id, (&q_msg, &kv_msg))
            })
            .collect::<Result<Vec<_>>>()?;
        let last_device = *placement.last().unwrap();
        gpu::set_device(device + last_device as i32)?;
        let head = Dense::load(&ckpt, "model.llm.unembed.weight", s.vocab, s.hidden)?;
        let vision = if max_image_tokens == 0 {
            None
        } else {
            Some(Vision::load(&ckpt, &root, s.hidden, max_image_tokens)?)
        };
        let cpu = expert_threads
            .map(|n| -> Result<_> {
                let n = if n == 0 {
                    std::thread::available_parallelism().map_or(1, usize::from)
                } else {
                    n
                };
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .thread_name(|i| format!("inkling-expert-{i}"))
                    .build()?;
                crate::eprintln!("Inkling CPU expert offload uses {n} workers");
                Ok(std::sync::Arc::new(pool))
            })
            .transpose()?;
        let mut devices = Vec::with_capacity(gpus);
        for rank in 0..gpus {
            let id = device + rank as i32;
            gpu::set_device(id)?;
            let scratch = AttentionScratch::new(s.heads, s.kv_heads, s.extent, s.relative)?;
            let mut sizes = [0; 2];
            let mut visits = [0; 2];
            let mut kv_bytes = 0usize;
            let mut all_experts = 0usize;
            for (layer, &assigned) in layers.iter().zip(&placement) {
                if assigned == rank {
                    all_experts += expert_size(&layer.experts) * s.experts;
                }
            }
            for visit in &loops.visits {
                if placement[visit.layer] != rank {
                    continue;
                }
                let layer = &layers[visit.layer];
                let packing = layer.experts.gate_up.packing as usize;
                let slots = if layer.local {
                    loops.window.min(context)
                } else if host_kv {
                    context.min(SHORT_KV_TOKENS)
                } else {
                    context
                };
                kv_bytes += slots * Kv::row_bytes(s.kv_heads, kv_dtype) * 2;
                sizes[packing] = sizes[packing].max(expert_size(&layer.experts));
                visits[packing] += 1;
            }
            let vision_reserve = if rank == last_device && vision.is_some() {
                max_image_tokens * 4 * 32768
            } else {
                0
            };
            let (free, _) = gpu::mem_info()?;
            let reserve = (512 << 20) + vision_reserve;
            let max_experts = free.saturating_sub(kv_bytes).saturating_sub(reserve).min(all_experts);
            let expert_bytes = if expert_cache_mib == 0 {
                max_experts
            } else {
                expert_cache_mib
                    .checked_mul(1 << 20)
                    .context("expert cache budget overflows")?
                    .min(max_experts)
            };
            ensure!(
                expert_bytes > 0,
                "Inkling: no VRAM for experts on CUDA device {id} after reserving {} MiB for KV",
                kv_bytes >> 20
            );
            let compute = Compute::new(
                (s.hidden * 2).max(s.q_width()),
                s.vocab.max(layers[0].query.rows),
                s.hidden,
                s.intermediate,
                expert_bytes,
                sizes,
                visits,
                s.top_k,
                prefill_chunk,
                cpu.clone(),
            )?;
            devices.push(Device { id, compute, scratch });
        }
        let norm_embed = read_float(&ckpt, "model.llm.embed_norm.weight", &[s.hidden])?;
        let norm_final = read_float(&ckpt, "model.llm.norm.weight", &[s.hidden])?;
        let query_final = read_float(&ckpt, "model.llm.attn_res.output_query", &[s.hidden])?;
        Ok(Self {
            ckpt,
            cfg: s,
            loops: loops.clone(),
            layers,
            head,
            norm_embed,
            norm_final,
            query_final,
            devices,
            placement,
            context,
            global_kv_host: host_kv,
            kv_dtype,
            vision,
            prefill_chunk,
        })
    }

    pub fn context(&self) -> usize {
        self.context
    }

    pub fn cpu_expert_routes(&self) -> u64 {
        self.devices.iter().map(|d| d.compute.cpu_routes()).sum()
    }

    pub fn prefill_chunk(&self) -> usize {
        self.prefill_chunk
    }

    pub fn sequence(&self, capacity: usize) -> Result<Sequence> {
        ensure!(
            capacity > 0 && capacity <= self.context,
            "sequence needs {capacity} tokens, context limit is {}",
            self.context
        );
        let s = &self.cfg;
        let visits = self
            .loops
            .visits
            .iter()
            .map(|visit| {
                gpu::set_device(self.devices[self.placement[visit.layer]].id)?;
                let local = self.loops.is_local(visit.layer);
                Ok(VisitState {
                    kv: Kv::new(
                        if local { capacity.min(self.loops.window) } else { capacity },
                        s.kv_heads,
                        !local && self.global_kv_host && capacity > SHORT_KV_TOKENS,
                        self.kv_dtype,
                    )?,
                    key: ConvState::new(s.conv, s.kv_width()),
                    value: ConvState::new(s.conv, s.kv_width()),
                    attention: ConvState::new(s.conv, s.hidden),
                    mlp: ConvState::new(s.conv, s.hidden),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Sequence {
            position: 0,
            capacity,
            visits,
        })
    }

    fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        let s = &self.cfg;
        let token = token as usize;
        ensure!(
            token < s.unpadded_vocab,
            "Inkling: token {token} outside vocabulary {}",
            s.unpadded_vocab
        );
        let row = self
            .ckpt
            .bytes("model.llm.embed.weight")?
            .chunks_exact(s.hidden * 2)
            .nth(token)
            .unwrap();
        let x: Vec<_> = row
            .chunks_exact(2)
            .map(|p| from_bf16(u16::from_le_bytes([p[0], p[1]])))
            .collect();
        Ok(normalized(&x, Some(&self.norm_embed), s.eps).into_iter().map(round).collect())
    }

    pub fn encode_image(&self, bytes: &[u8]) -> Result<Vec<Vec<u16>>> {
        let vision = self.vision.as_ref().context("Inkling vision is disabled")?;
        let device = &self.devices[*self.placement.last().unwrap()];
        gpu::set_device(device.id)?;
        vision.encode(&device.compute, bytes)
    }

    pub(super) fn forward(&mut self, seq: &mut Sequence, token: u32, readout: Readout) -> Result<Option<Prediction>> {
        let x = self.embedding(token)?;
        self.forward_rows(seq, vec![ForwardRow::new(x)], readout)
    }

    pub(super) fn prefill(&mut self, seq: &mut Sequence, input: &[Input], readout: Readout) -> Result<Option<Prediction>> {
        let rows = input
            .iter()
            .map(|item| {
                let x = match item {
                    Input::Token(id) => self.embedding(*id)?,
                    Input::Image(embedding) => {
                        ensure!(embedding.len() == self.cfg.hidden, "Inkling image embedding width mismatch");
                        embedding.iter().copied().map(from_bf16).collect()
                    }
                };
                Ok(ForwardRow::new(x))
            })
            .collect::<Result<_>>()?;
        self.forward_rows(seq, rows, readout)
    }

    fn forward_rows(&mut self, seq: &mut Sequence, mut rows: Vec<ForwardRow>, readout: Readout) -> Result<Option<Prediction>> {
        ensure!(
            !rows.is_empty() && rows.len() <= seq.capacity.saturating_sub(seq.position),
            "Inkling chunk exceeds the sequence context limit of {}",
            seq.capacity
        );
        let s = &self.cfg;
        let mut active = None;
        // Visit each physical weight bank once per chunk. Per-visit attention and convolution still consume
        // tokens in order, so batching the traversal has the same state transitions as token-wise decoding.
        for (visit, state) in self.loops.visits.iter().zip(&mut seq.visits) {
            let id = visit.layer;
            let rank = self.placement[id];
            let device = &mut self.devices[rank];
            if active != Some(rank) {
                gpu::set_device(device.id)?;
                active = Some(rank);
            }
            let compute = &mut device.compute;
            let layer = &self.layers[id];
            let mut residuals = Vec::with_capacity(rows.len());
            let mut activations = Vec::with_capacity(rows.len());
            let mut q_inputs = Vec::with_capacity(rows.len());
            let mut k_inputs = Vec::with_capacity(rows.len());
            let mut v_inputs = Vec::with_capacity(rows.len());
            for row in &mut rows {
                let ForwardRow {
                    x,
                    blocks,
                    prelude,
                    messages,
                } = row;
                if id == self.loops.core.start {
                    if visit.pass == 0 {
                        *prelude = Some((x.clone(), blocks.len()));
                    } else {
                        let (start, depth) = prelude.as_ref().unwrap();
                        *messages = blocks[*depth + 1..].to_vec();
                        messages.push(normalized(x, None, s.eps));
                        blocks.truncate(*depth);
                        x.clone_from(start);
                    }
                }
                if id == self.loops.core.end {
                    messages.clear();
                }
                let controller = (visit.pass > 0).then(|| layer.loop_inputs.as_ref().unwrap());
                let mut qkv = if id == 0 {
                    vec![x.clone()]
                } else {
                    mix(blocks, x, &layer.res_query, s.eps)
                };
                if qkv.len() == 1 && controller.is_some() {
                    qkv.resize(3, qkv[0].clone());
                }
                if let Some(c) = controller {
                    c.mix_attention(compute, &mut qkv, messages, s.eps)?;
                }
                let norm = |x: &[f32], weight: &[f32]| normalized(x, Some(weight), s.eps);
                let q_in = norm(&qkv[0], &layer.norm_q);
                let k_in = qkv
                    .get(1)
                    .map(|v| norm(v, if layer.local { &layer.norm_q } else { &layer.norm_k }));
                let v_in = qkv
                    .get(2)
                    .map(|v| norm(v, if layer.local { &layer.norm_q } else { &layer.norm_v }));
                q_inputs.push(q_in);
                k_inputs.push(k_in);
                v_inputs.push(v_in);
            }
            let q_refs: Vec<_> = q_inputs.iter().map(Vec::as_slice).collect();
            let k_refs: Vec<_> = k_inputs
                .iter()
                .zip(&q_refs)
                .map(|(k, &q)| k.as_deref().unwrap_or(q))
                .collect();
            let v_refs: Vec<_> = v_inputs
                .iter()
                .zip(&q_refs)
                .map(|(v, &q)| v.as_deref().unwrap_or(q))
                .collect();
            let [q_batch, k_batch, v_batch] =
                compute.project_three([&layer.query, &layer.key, &layer.value], [&q_refs, &k_refs, &v_refs])?;
            for (offset, row) in rows.iter_mut().enumerate() {
                let position = seq.position + offset;
                let ForwardRow { x, blocks, messages, .. } = row;
                let q_raw = &q_batch[offset * layer.query.rows..(offset + 1) * layer.query.rows];
                let (q_raw, rest) = q_raw.split_at(s.q_width());
                let (gate, rest) = rest.split_at(s.q_width());
                let (rel_raw, sink) = rest.split_at(s.rel_width());
                let log_scale = if layer.local {
                    1.0
                } else {
                    1.0 + s.log_alpha * (((position + 1) as f32 / s.log_floor).max(1.0)).ln()
                };
                let q = clipped(q_raw, &layer.q_clip, s.dim, s.qk_clip, log_scale);
                let rel = clipped(rel_raw, &layer.rel_clip, s.relative, s.rel_clip, log_scale);
                let kv_range = offset * s.kv_width()..(offset + 1) * s.kv_width();
                let key: Vec<_> = k_batch[kv_range.clone()].iter().copied().map(from_bf16).collect();
                let value: Vec<_> = v_batch[kv_range].iter().copied().map(from_bf16).collect();
                let key = layer
                    .k_conv
                    .run(compute, &mut state.key, &key, k_refs[offset], position, s.dim)?;
                let value = layer
                    .v_conv
                    .run(compute, &mut state.value, &value, v_refs[offset], position, s.dim)?;
                let key: Vec<_> = key
                    .chunks_exact(s.dim)
                    .flat_map(|row| {
                        let scale = clip_scale(row, s.qk_clip);
                        row.iter().map(move |&v| to_bf16(v * scale))
                    })
                    .collect();
                let value: Vec<_> = value
                    .chunks_exact(s.dim)
                    .flat_map(|row| {
                        let scale = inv_rms(row, s.eps);
                        row.iter().map(move |&v| to_bf16(v * scale))
                    })
                    .collect();
                let attn = device.scratch.run(
                    compute,
                    &state.kv,
                    &q,
                    gate,
                    sink,
                    &rel,
                    &key,
                    &value,
                    &layer.bias,
                    &layer.output,
                    position,
                )?;
                let mut residual = layer
                    .attn_conv
                    .run(compute, &mut state.attention, &attn, &[], position, s.dim)?;
                if id % s.block == 0 {
                    blocks.push(normalized(x, None, s.eps));
                } else {
                    residual.iter_mut().zip(x.iter()).for_each(|(v, &x)| *v += x);
                }
                let mut mlp_in = mix(blocks, &residual, &layer.res_mlp, s.eps).remove(0);
                if visit.pass > 0 {
                    let c = layer.loop_inputs.as_ref().unwrap();
                    let msg = mix_normalized(messages, &c.queries[3]);
                    c.controllers[3].merge(compute, &mut mlp_in, &msg, s.eps)?;
                }
                let mlp_in = normalized(&mlp_in, Some(&layer.norm_mlp), s.eps)
                    .into_iter()
                    .map(round)
                    .collect::<Vec<_>>();
                activations.push(mlp_in);
                residuals.push(residual);
            }
            let routes: Vec<_> = match &layer.router {
                Router::Device(w) => compute.router_logits(w, &activations)?.chunks_exact(s.experts)
                    .map(|row| layer.routes(row.iter().copied(), s)).collect(),
                Router::Host(w) => activations.iter().map(|x| {
                    layer.routes(w.chunks_exact(s.hidden).map(|row| dot(row, x)), s)
                }).collect(),
            };
            let mut moe = compute.moe_batch(
                &self.ckpt,
                &layer.experts,
                id,
                &activations,
                &routes,
                (s.gate_cap, s.up_cap),
                s.experts,
            )?;
            for (offset, ((row, residual), moe)) in rows.iter_mut().zip(residuals).zip(moe.chunks_exact_mut(s.hidden)).enumerate()
            {
                moe.iter_mut().for_each(|v| *v = round(*v));
                let mlp = layer
                    .mlp_conv
                    .run(compute, &mut state.mlp, moe, &[], seq.position + offset, s.dim)?;
                row.x = residual.into_iter().zip(mlp).map(|(a, b)| round(a + b)).collect();
            }
        }
        let result = if matches!(readout, Readout::None) {
            None
        } else {
            let row = rows.last().unwrap();
            let x = mix(&row.blocks, &row.x, &self.query_final, s.eps).remove(0);
            let x = normalized(&x, Some(&self.norm_final), s.eps);
            let gpu = &mut self.devices[*self.placement.last().unwrap()].compute;
            // The model's tanh softcap is monotonic, so greedy decoding can select on the BF16 head output.
            if matches!(readout, Readout::Greedy) {
                Some(Prediction::Token(gpu.argmax(&self.head, &x, s.unpadded_vocab)?))
            } else {
                let mut logits = gpu.linear(&self.head, &x)?;
                logits.truncate(s.unpadded_vocab);
                cpu::softcap(&mut logits, s.logit_cap);
                Some(Prediction::Scores(logits))
            }
        };
        seq.position += rows.len();
        Ok(result)
    }
}

struct ForwardRow {
    x: Vec<f32>,
    blocks: Vec<Vec<f32>>,
    prelude: Option<(Vec<f32>, usize)>,
    messages: Vec<Vec<f32>>,
}

impl ForwardRow {
    fn new(x: Vec<f32>) -> Self {
        Self {
            x,
            blocks: Vec::new(),
            prelude: None,
            messages: Vec::new(),
        }
    }
}

fn round(v: f32) -> f32 {
    from_bf16(to_bf16(v))
}

fn inv_rms(x: &[f32], eps: f32) -> f32 {
    (dot(x, x) / x.len() as f32 + eps).sqrt().recip()
}

fn normalized(x: &[f32], scale: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let rms = inv_rms(x, eps);
    match scale {
        Some(s) => x.iter().zip(s).map(|(&x, &s)| x * rms * s).collect(),
        None => x.iter().map(|&x| x * rms).collect(),
    }
}

fn mix(blocks: &[Vec<f32>], partial: &[f32], queries: &[f32], eps: f32) -> Vec<Vec<f32>> {
    let current = normalized(partial, None, eps);
    queries
        .chunks_exact(partial.len())
        .map(|q| mix_rows(blocks.iter().map(Vec::as_slice).chain(std::iter::once(current.as_slice())), q))
        .collect()
}

fn mix_normalized(blocks: &[Vec<f32>], query: &[f32]) -> Vec<f32> {
    mix_rows(blocks.iter().map(Vec::as_slice), query)
}

fn mix_rows<'a>(rows: impl Iterator<Item = &'a [f32]> + Clone, query: &[f32]) -> Vec<f32> {
    let scale = (query.len() as f32).sqrt().recip();
    let mut logits: Vec<_> = rows.clone().map(|r| dot(r, query) * scale).collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    logits.iter_mut().for_each(|l| *l = (*l - max).exp());
    let inv = logits.iter().sum::<f32>().recip();
    let mut output = vec![0.0; query.len()];
    for (row, weight) in rows.zip(logits) {
        let weight = weight * inv;
        for (out, &v) in output.iter_mut().zip(row) {
            *out += v * weight;
        }
    }
    output
}

fn clip_scale(x: &[f32], max: f32) -> f32 {
    (max / (dot(x, x).sqrt() + 1e-6)).min(1.0)
}

fn clipped(x: &[u16], learned: &[f32], dim: usize, max: f32, log_scale: f32) -> Vec<u16> {
    let scaled: Vec<_> = x.iter().zip(learned).map(|(&x, &w)| from_bf16(x) * w).collect();
    scaled
        .chunks_exact(dim)
        .flat_map(|row| {
            let scale = clip_scale(row, max);
            row.iter().map(move |&v| to_bf16(round(v * scale) * log_scale))
        })
        .collect()
}

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // The target features were detected above.
        return unsafe { dot_avx(a, b) };
    }
    a.iter().zip(b).map(|(&a, &b)| a * b).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = [_mm256_setzero_ps(); 4];
    let n = a.len() / 32 * 32;
    for (a, b) in a[..n].chunks_exact(32).zip(b[..n].chunks_exact(32)) {
        for (i, acc) in acc.iter_mut().enumerate() {
            let (a, b) = unsafe { (_mm256_loadu_ps(a.as_ptr().add(i * 8)), _mm256_loadu_ps(b.as_ptr().add(i * 8))) };
            *acc = _mm256_fmadd_ps(a, b, *acc);
        }
    }
    let mut lanes = [0.0; 8];
    let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
    lanes.into_iter().sum::<f32>() + a[n..].iter().zip(&b[n..]).map(|(&a, &b)| a * b).sum::<f32>()
}
