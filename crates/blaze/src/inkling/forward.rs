use super::KvDtype;
use super::dspark::{Cache as DraftCache, Draft};
use super::cuda::{AttentionScratch, Compute, Dense, Kv, from_bf16, prefill_attention_rows, to_bf16};
use super::generate::Input;
use super::vision::Vision;
use super::weights::{Experts, read_float};
use crate::config::{InklingLoop, MlpConfig, ModelConfig};
use crate::expert_rg16;
use crate::expert_runtime::MoeEngine;
use crate::gpu;
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Context, Result, ensure};
use std::path::Path;

#[path = "resident.rs"]
mod resident;

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

    #[cfg(test)]
    fn run(
        &self,
        state: &mut ConvState,
        x: &[f32],
        dyn_weights: Option<&[u16]>,
        pos: usize,
        dim: usize,
    ) -> Vec<f32> {
        let k = self.taps.len();
        let out = self.apply(x, dyn_weights, k.min(pos + 1), dim, |lag| {
            &state.rows[(pos - lag) % state.rows.len()]
        });
        if k > 1 {
            let slot = pos % state.rows.len();
            state.rows[slot].copy_from_slice(x);
        }
        out
    }

    #[cfg(test)]
    fn apply<'a>(
        &self, x: &[f32], dyn_weights: Option<&[u16]>, taps: usize, dim: usize,
        previous: impl Fn(usize) -> &'a [f32],
    ) -> Vec<f32> {
        let k = self.taps.len();
        let mut out = x.to_vec();
        for lag in 0..taps {
            let src = if lag == 0 { x } else { previous(lag) };
            let taps = &self.taps[lag];
            if let Some(d) = dyn_weights {
                for (head, ((out, src), taps)) in out
                    .chunks_exact_mut(dim)
                    .zip(src.chunks_exact(dim))
                    .zip(taps.chunks_exact(dim))
                    .enumerate()
                {
                    let extra = from_bf16(d[head * k + lag]);
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
        out.iter_mut().for_each(|v| *v = round(*v * self.scale));
        out
    }
}

#[cfg(test)]
struct ConvState {
    rows: Vec<Vec<f32>>,
}

#[cfg(test)]
impl ConvState {
    fn new(taps: usize, width: usize, slack: usize) -> Self {
        Self {
            rows: vec![vec![0.0; width]; if taps > 1 { taps - 1 + slack } else { 0 }],
        }
    }

    fn bf16_history(&self, pos: usize, taps: usize) -> Vec<u16> {
        (pos.saturating_sub(taps - 1)..pos)
            .flat_map(|p| self.rows[p % self.rows.len()].iter().copied().map(to_bf16))
            .collect()
    }

    #[allow(dead_code)]
    fn remember_bf16(&mut self, start: usize, width: usize, rows: &[u16]) {
        let slots = self.rows.len();
        let count = rows.len() / width;
        for r in count.saturating_sub(slots)..count {
            let slot = (start + r) % slots;
            for (dst, &value) in self.rows[slot].iter_mut().zip(&rows[r * width..(r + 1) * width]) {
                *dst = from_bf16(value);
            }
        }
    }

    #[allow(dead_code)]
    fn remember_f32(&mut self, start: usize, width: usize, rows: &[f32]) {
        let slots = self.rows.len();
        let count = rows.len() / width;
        for r in count.saturating_sub(slots)..count {
            self.rows[(start + r) % slots].copy_from_slice(&rows[r * width..(r + 1) * width]);
        }
    }
}

struct Controller {
    down: Dense,
    up: Dense,
}

impl Controller {
    fn load(ckpt: &Checkpoint, name: &str, hidden: usize, rank: usize) -> Result<Self> {
        let forget = format!("{name}.forget_weight");
        let gain = format!("{name}.gain_weight");
        ensure!(rank == 64, "Inkling requires rank-64 loop controllers");
        let up = Dense::stack(ckpt, &[(&forget, hidden), (&gain, hidden)], rank)?;
        Ok(Self {
            down: Dense::load(ckpt, &format!("{name}.down_weight"), rank, hidden * 2)?,
            up,
        })
    }

}

struct LoopInputs {
    queries: [Vec<f32>; 4],
    controllers: [Controller; 4],
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
    router: Dense,
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
            let queries = [row(messages.0, 0), row(messages.1, 0), row(messages.1, 1), row(messages.0, 1)];
            Some(LoopInputs {
                queries,
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
        let norm_mlp = read("mlp_norm.weight", &[s.hidden])?;
        let mut rel_transposed = vec![0u16; rel.len()];
        for (i, row) in rel.chunks_exact(s.extent).enumerate() {
            for (j, &v) in row.iter().enumerate() {
                rel_transposed[j * s.relative + i] = to_bf16(v);
            }
        }
        let k_conv = Conv::load(ckpt, &t("attn.k_sconv"), kv, s, !local)?;
        let v_conv = Conv::load(ckpt, &t("attn.v_sconv"), kv, s, !local)?;
        let q_clip = read("attn.q_l2_clip.scale", &[s.heads, s.dim])?;
        let rel_clip = read("attn.rel_l2_clip.scale", &[s.heads, s.relative])?;
        let norm_q = read(if local { "attn_norm.weight" } else { "q_input_norm.weight" }, &[s.hidden])?;
        let norm_k = if local { Vec::new() } else { read("k_input_norm.weight", &[s.hidden])? };
        let norm_v = if local { Vec::new() } else { read("v_input_norm.weight", &[s.hidden])? };
        let attn_conv = Conv::load(ckpt, &t("attn_sconv"), s.hidden, s, false)?;
        let mlp_conv = Conv::load(ckpt, &t("mlp_sconv"), s.hidden, s, false)?;
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
            norm_q,
            norm_k,
            norm_v,
            norm_mlp,
            q_clip,
            rel_clip,
            res_query,
            res_mlp: read_float(ckpt, &format!("{attn_res}.mlp_queries.{id}"), &[s.hidden])?,
            loop_inputs,
            k_conv,
            v_conv,
            attn_conv,
            mlp_conv,
            router: dense("mlp.gate.weight", s.experts, s.hidden)?,
            route_bias: read("mlp.gate.bias", &[s.experts])?,
            experts: Experts::load(ckpt, cfg, id)?,
        })
    }

}

struct VisitState {
    kv: Kv,
    history: [gpu::DevBuf; 4],
    verification: Option<[gpu::DevBuf; 4]>,
}

pub(super) struct SampleRow<'a> {
    pub params: &'a crate::sampler::SamplingParams,
    pub rng: &'a mut crate::sampler::Rng,
}

pub(super) enum Readout<'a> {
    None,
    Greedy,
    Verify { draft: &'a [u32], sample: Option<SampleRow<'a>> },
    Sample(SampleRow<'a>),
}

impl<'a> Readout<'a> {
    pub fn next(params: &'a crate::sampler::SamplingParams, rng: &'a mut crate::sampler::Rng) -> Self {
        if params.greedy() { Self::Greedy } else { Self::Sample(SampleRow { params, rng }) }
    }
}

pub(super) enum Prediction {
    Token(u32),
    Verified(usize, u32),
}

pub struct Sequence {
    draft: Option<DraftCache>,
    position: usize,
    capacity: usize,
    visits: Vec<VisitState>,
    history_copies: Vec<(gpu::DevBuf, i32)>,
}

impl Sequence {
    pub fn len(&self) -> usize {
        self.position
    }
}

pub struct Options {
    pub draft: Option<std::path::PathBuf>,
    pub dspark_tokens: usize,
    pub dspark_head_vocab: usize,
    pub device: i32,
    pub gpus: usize,
    pub context: usize,
    pub expert_cache_mib: usize,
    pub expert_threads: Option<usize>,
    pub host_kv: bool,
    pub kv_dtype: KvDtype,
    pub max_image_tokens: usize,
    pub prefill_chunk: usize,
    pub prefill_attention: super::PrefillAttention,
}

struct Device {
    id: i32,
    compute: Compute,
    scratch: AttentionScratch,
    resident_moe: bool,
}

impl Drop for Device {
    fn drop(&mut self) {
        let _ = gpu::set_device(self.id);
    }
}

pub struct Model {
    executor: resident::Executor,
    draft: Option<Draft>,
    dflash: Option<super::dflash::Draft>,
    ckpt: Checkpoint,
    cfg: Shape,
    loops: InklingLoop,
    residual_walk: Vec<crate::layer_walk::ResidualVisit>,
    layers: Vec<Layer>,
    head: Dense,
    norm_embed: Vec<f32>,
    devices: Vec<Device>,
    placement: Vec<usize>,
    expert_layers: Vec<usize>,
    context: usize,
    global_kv_host: bool,
    kv_dtype: KvDtype,
    vision: Option<Vision>,
    prefill_chunk: usize,
}

impl Model {
    pub fn load(path: &Path, opts: Options) -> Result<Self> {
        let Options {
            draft: draft_path,
            dspark_tokens,
            dspark_head_vocab,
            device,
            gpus,
            context,
            expert_cache_mib,
            expert_threads,
            host_kv,
            kv_dtype,
            max_image_tokens,
            prefill_chunk,
            prefill_attention,
        } = opts;
        ensure!(
            (1..=2).contains(&gpus) && device >= 0 && device <= i32::MAX - 1,
            "Inkling requires one or two contiguous CUDA devices"
        );
        ensure!(
            (1..=2048).contains(&prefill_chunk),
            "Inkling prefill chunk must be in 1..=2048"
        );
        let root = ModelConfig::read_json(path)?;
        ensure!(dspark_tokens <= 7 && (dspark_tokens == 0 || dspark_tokens < prefill_chunk),
            "DSpark supports up to seven drafts and needs a prefill chunk larger than the draft count");
        ensure!(draft_path.is_none() || dspark_tokens == 0 && prefill_chunk >= 8,
            "Inkling DFlash needs a chunk of at least eight tokens and its own draft checkpoint");
        let verify_limit = if draft_path.is_some() { 4 } else { dspark_tokens };
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
        ensure!(resident::Executor::supports(&s, cfg.layers.len()), "unsupported Inkling execution geometry");
        // Keep the entire recurrent core on one GPU. With two devices, the prelude and coda share the other GPU
        // and forward has only two device transitions, independently of the number of loop passes.
        let placement: Vec<_> = (0..cfg.layers.len())
            .map(|id| usize::from(gpus == 2 && !loops.core.contains(&id)))
            .collect();
        let load_rank = |rank: usize| -> Result<Vec<(usize, Layer)>> {
            gpu::set_device(device + rank as i32)?;
            placement.iter().enumerate().filter(|(_, assigned)| **assigned == rank)
                .map(|(id, _)| {
                    let layer = Layer::load(&ckpt, &cfg, loops, &s, id, (&q_msg, &kv_msg))?;
                    Ok((id, layer))
                })
                .collect()
        };
        let mut loaded = if gpus == 2 {
            std::thread::scope(|scope| -> Result<_> {
                let other = scope.spawn(|| load_rank(1));
                let first = load_rank(0);
                let second = other.join().map_err(|_| anyhow::anyhow!("Inkling second GPU loader stopped"))?;
                let mut layers = first?;
                layers.extend(second?);
                Ok(layers)
            })?
        } else {
            load_rank(0)?
        };
        loaded.sort_unstable_by_key(|(id, _)| *id);
        let layers: Vec<_> = loaded.into_iter().map(|(_, layer)| layer).collect();
        let mut layer_map = vec![0; layers.len()];
        let mut expert_layout: Vec<_> = (0..layers.len()).collect();
        expert_layout.sort_by_key(|&layer| placement[layer]);
        for (index, &layer) in expert_layout.iter().enumerate() {
            layer_map[layer] = index;
        }
        ensure!(s.hidden.is_multiple_of(16) && s.intermediate.is_multiple_of(16)
            && (1..=16).contains(&s.top_k) && s.top_k <= s.experts,
            "Inkling does not support this HybridMoe geometry");
        let threads = std::thread::available_parallelism().map_or(1, usize::from)
            .min(if gpus == 1 { 4 } else { 12 });
        let store = expert_rg16::cached_store(&ckpt, &expert_layout, s.experts, s.hidden, s.intermediate, threads,
            |layer, expert, out| layers[layer].experts.to_rg16(&ckpt, expert, out))?;
        let last_device = *placement.last().unwrap();
        gpu::set_device(device + last_device as i32)?;
        let head = Dense::load(&ckpt, "model.llm.unembed.weight", s.vocab, s.hidden)?;
        let draft = (dspark_tokens > 0).then(|| {
            ensure!(s.hidden == 2304 && loops.visits.len() == 48 && context < i32::MAX as usize - 8, "unsupported DSpark target");
            Draft::load(&ckpt, &root["mtp_config"], dspark_tokens, s.vocab, dspark_head_vocab)
        }).transpose()?;
        let vision = if max_image_tokens == 0 {
            None
        } else {
            Some(Vision::load(&ckpt, &root, s.hidden, max_image_tokens)?)
        };
        let cpu_threads = match expert_threads {
            None => 1,
            Some(0) => std::thread::available_parallelism().map_or(1, usize::from),
            Some(n) => n,
        };
        let mut devices = Vec::with_capacity(gpus);
        let norm_embed = read_float(&ckpt, "model.llm.embed_norm.weight", &[s.hidden])?;
        let norm_final = read_float(&ckpt, "model.llm.norm.weight", &[s.hidden])?;
        let query_final = read_float(&ckpt, "model.llm.attn_res.output_query", &[s.hidden])?;
        let residual_walk = crate::layer_walk::residual_visits(&loops.visits, loops.core.start, s.block)?;
        let capture = if draft_path.is_some() {
            resident::CapturePlacement::Drafter
        } else if draft.is_some() {
            resident::CapturePlacement::Host
        } else {
            resident::CapturePlacement::None
        };
        let executor = resident::Executor::new(device, gpus, &placement, &layers, &s, prefill_chunk,
            capture, &loops.visits, &norm_embed, &norm_final, &query_final)?;
        let mut expert_layer_start = 0;
        for rank in 0..gpus {
            let id = device + rank as i32;
            gpu::set_device(id)?;
            let mut scratch = AttentionScratch::new(s.heads, s.kv_heads, s.extent, s.relative, prefill_chunk, s.conv)?;
            scratch.set_prefill_attention(prefill_attention);
            let mut kv_bytes = 0usize;
            let expert_layer_count = placement.iter().filter(|&&assigned| assigned == rank).count();
            let all_experts = expert_rg16::record_bytes(s.hidden, s.intermediate) * s.experts * expert_layer_count;
            for visit in &loops.visits {
                if placement[visit.layer] != rank {
                    continue;
                }
                let layer = &layers[visit.layer];
                let slots = if layer.local {
                    (loops.window + prefill_attention_rows(prefill_chunk).saturating_sub(1).max(verify_limit)).min(context)
                } else if host_kv {
                    context.min(SHORT_KV_TOKENS)
                } else {
                    context
                };
                kv_bytes += slots * Kv::row_bytes(s.kv_heads, kv_dtype) * 2;
            }
            let vision_reserve = if rank == last_device && vision.is_some() {
                max_image_tokens * 4 * 32768
            } else {
                0
            };
            let (free, _) = gpu::mem_info()?;
            let draft_reserve = if draft.is_some() && rank == last_device { DraftCache::bytes(context, kv_dtype) } else { 0 };
            let dflash_reserve = if rank == last_device && let Some(path) = &draft_path {
                crate::drafter::vram_estimate(path, 1, None)?
            } else { 0 };
            let reserve = (512 << 20) + vision_reserve + draft_reserve + dflash_reserve;
            let max_experts = free.saturating_sub(kv_bytes).saturating_sub(reserve).min(all_experts);
            let expert_bytes = if expert_cache_mib == 0 {
                max_experts
            } else {
                expert_cache_mib
                    .checked_mul(1 << 20)
                    .context("expert cache budget overflows")?
                    .min(max_experts)
            };
            let mut compute = Compute::new(
                (s.hidden * 2).max(s.q_width()),
                s.vocab.saturating_mul(verify_limit + 1).max((layers[0].query.rows + 2 * (s.kv_width() + s.kv_heads * s.conv)) * prefill_chunk
                    + 2 * s.kv_width() * (s.conv - 1)),
                s.hidden,
                prefill_chunk,
            )?;
            compute.tensor_prefill = prefill_attention == super::PrefillAttention::Tensor;
            let end = expert_layer_start + expert_layer_count;
            let threads = cpu_threads.div_ceil(gpus);
            let all_resident = expert_bytes >= all_experts;
            let stream_gpu_only = expert_threads.is_none() && !all_resident;
            let engine = MoeEngine::packed(&store, expert_layer_start..end, id, expert_bytes, s.top_k,
                threads, prefill_chunk, stream_gpu_only, all_resident, true, (s.gate_cap, s.up_cap), compute.tensor_prefill)?;
            compute.attach_expert_engine(engine);
            expert_layer_start = end;
            devices.push(Device { id, compute, scratch, resident_moe: all_resident });
        }
        let dflash = draft_path.as_ref().map(|path| {
            let device = &devices[last_device];
            gpu::set_device(device.id)?;
            super::dflash::Draft::load(path, &device.compute, device.id, ckpt.bytes("model.llm.embed.weight")?,
                &norm_embed, s.vocab, s.unpadded_vocab, loops.visits.len(), context, prefill_chunk)
        }).transpose()?;
        Ok(Self {
            executor,
            draft,
            dflash,
            ckpt,
            cfg: s,
            loops: loops.clone(),
            residual_walk,
            layers,
            head,
            norm_embed,
            devices,
            placement,
            expert_layers: layer_map,
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

    pub(super) fn prefill_wide_min(&self) -> usize {
        if self.devices[0].compute.tensor_prefill { self.prefill_chunk.clamp(32, 256) } else { 1 }
    }

    pub fn sequence(&self, capacity: usize) -> Result<Sequence> {
        ensure!(
            capacity > 0 && capacity <= self.context,
            "sequence needs {capacity} tokens, context limit is {}",
            self.context
        );
        let s = &self.cfg;
        let slack = self.draft_limit();
        let kv_slack = (prefill_attention_rows(self.prefill_chunk) - 1).max(slack);
        let visits = self
            .loops
            .visits
            .iter()
            .map(|visit| {
                gpu::set_device(self.devices[self.placement[visit.layer]].id)?;
                let local = self.loops.is_local(visit.layer);
                Ok(VisitState {
                    kv: Kv::new(
                        if local { capacity.min(self.loops.window + kv_slack) } else { capacity },
                        if local { capacity.min(self.loops.window) } else { capacity },
                        s.kv_heads,
                        !local && self.global_kv_host && capacity > SHORT_KV_TOKENS,
                        self.kv_dtype,
                    )?,
                    history: resident::histories(s, s.conv - 1)?,
                    verification: (slack > 0).then(|| resident::histories(s, s.conv + slack)).transpose()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut history_copies = Vec::new();
        if slack > 0 {
            let widths = [s.kv_width(), s.kv_width(), s.hidden, s.hidden];
            for (rank, device) in self.devices.iter().enumerate() {
                gpu::set_device(device.id)?;
                let mut copies = Vec::new();
                for (visit, state) in self.loops.visits.iter().zip(&visits) {
                    if self.placement[visit.layer] != rank {
                        continue;
                    }
                    let pending = state.verification.as_ref().unwrap();
                    copies.extend(widths.iter().enumerate().map(|(i, &width)| blaze_sys::blz_row_copy {
                        dst: state.history[i].ptr(),
                        src: pending[i].ptr(),
                        width: width as i64,
                    }));
                }
                history_copies.push((gpu::DevBuf::from_slice(&copies)?, copies.len() as i32));
            }
        }
        Ok(Sequence {
            draft: self.draft.as_ref().map(|_| DraftCache::new(capacity, self.kv_dtype)).transpose()?,
            position: 0,
            capacity,
            visits,
            history_copies,
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

    pub(super) fn draft_limit(&self) -> usize {
        if self.dflash.is_some() { 4 } else { self.draft.as_ref().map_or(0, |draft| draft.limit) }
    }

    pub(super) fn draft_name(&self) -> &'static str {
        if self.dflash.is_some() { "DFlash2" } else { "DSpark" }
    }

    pub(super) fn draft_tokens(&mut self, seq: &Sequence, anchor: u32, limit: usize) -> Result<Vec<u32>> {
        if let Some(draft) = &mut self.dflash {
            let device = &self.devices[*self.placement.last().unwrap()];
            gpu::set_device(device.id)?;
            return draft.propose(&device.compute, &self.head, seq.position, anchor, limit);
        }
        let x = self.embedding(anchor)?;
        let device = &mut self.devices[*self.placement.last().unwrap()];
        gpu::set_device(device.id)?;
        self.draft.as_ref().unwrap().propose(&mut device.compute, seq.draft.as_ref().unwrap(), &self.ckpt,
            &self.head, anchor, x, seq.position, self.cfg.unpadded_vocab, limit)
    }

    pub(super) fn verify<'a>(
        &mut self,
        seq: &mut Sequence,
        anchor: u32,
        draft: &'a [u32],
        sample: Option<SampleRow<'a>>,
    ) -> Result<(usize, u32)> {
        if !draft.is_empty() {
            let input: Vec<_> = std::iter::once(anchor).chain(draft.iter().copied()).map(Input::Token).collect();
            return match self.forward_gpu(seq, &input, Readout::Verify { draft, sample }, 0)? {
                Some(Prediction::Verified(kept, next)) => Ok((kept, next)),
                _ => anyhow::bail!("missing Inkling target verdict"),
            };
        }
        let readout = sample.map_or(Readout::Greedy, Readout::Sample);
        match self.forward(seq, anchor, readout)? {
            Some(Prediction::Token(id)) => Ok((0, id)),
            _ => anyhow::bail!("missing Inkling target prediction"),
        }
    }

    pub fn encode_image(&self, bytes: &[u8]) -> Result<Vec<Vec<u16>>> {
        self.encode_decoded_image(&super::vision::decode_image(bytes)?)
    }

    pub(super) fn encode_decoded_image(&self, image: &crate::vision::RgbImage) -> Result<Vec<Vec<u16>>> {
        let vision = self.vision.as_ref().context("Inkling vision is disabled")?;
        let device = &self.devices[*self.placement.last().unwrap()];
        gpu::set_device(device.id)?;
        vision.encode(&device.compute, image)
    }

    pub(super) fn forward(&mut self, seq: &mut Sequence, token: u32, readout: Readout) -> Result<Option<Prediction>> {
        self.forward_gpu(seq, &[Input::Token(token)], readout, 0)
    }

    pub(super) fn prefill(
        &mut self, seq: &mut Sequence, input: &[Input], prompt_len: usize, readout: Readout,
    ) -> Result<Option<Prediction>> {
        let skip_features = self.dflash.as_ref().map_or(0, |draft| {
            prompt_len.saturating_sub(draft.window()).saturating_sub(seq.position).min(input.len())
        });
        self.forward_gpu(seq, input, readout, skip_features)
    }
}

fn round(v: f32) -> f32 {
    from_bf16(to_bf16(v))
}

fn inv_rms(x: &[f32], eps: f32) -> f32 {
    (dot(x, x) / x.len() as f32 + eps).sqrt().recip()
}

pub(super) fn normalized(x: &[f32], scale: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let rms = inv_rms(x, eps);
    match scale {
        Some(s) => x.iter().zip(s).map(|(&x, &s)| x * rms * s).collect(),
        None => x.iter().map(|&x| x * rms).collect(),
    }
}

#[cfg(test)]
fn clip_scale(x: &[f32], max: f32) -> f32 {
    (max / (dot(x, x).sqrt() + 1e-6)).min(1.0)
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

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::gpu::{DevBuf, Stream, check};

    #[test]
    #[ignore = "requires CUDA"]
    fn prefill_residual_convolution_matches_token_order() {
        gpu::set_device(0).unwrap();
        let (width, taps, tokens) = (256, 4, 19);
        let weights = (0..taps).map(|lag| {
            (0..width).map(|c| ((c * 29 + lag * 19) % 197) as f32 / 311.0 - 0.42).collect()
        }).collect();
        let conv = Conv { taps: weights, dynamic: None, scale: ((taps + 1) as f32).sqrt().recip() };
        let device_weights = DevBuf::from_slice(&conv.taps.iter().flatten().copied().collect::<Vec<_>>()).unwrap();
        let raw: Vec<_> = (0..tokens * width).map(|i| to_bf16((i * 13 % 227) as f32 / 57.0 - 2.0)).collect();
        let previous: Vec<_> = (0..tokens * width).map(|i| to_bf16((i * 51 % 389) as f32 / 62.0 - 3.0)).collect();
        let stream = Stream::new().unwrap();
        let mut serial = ConvState::new(taps, width, 0);
        let mut mlp = ConvState::new(taps, width, 0);
        for (start, count, add) in [(0, 2, false), (2, 9, true), (11, 8, true)] {
            let history = (taps - 1).min(start);
            let input = DevBuf::from_slice(&raw[(start - history) * width..(start + count) * width]).unwrap();
            let skip = DevBuf::from_slice(&previous[start * width..(start + count) * width]).unwrap();
            let output = DevBuf::new(count * width * 4).unwrap();
            check(unsafe {
                blaze_sys::blz_inkling_residual_conv(input.ptr(), if add { skip.ptr() } else { std::ptr::null() },
                    device_weights.ptr(), output.ptr(), width as i32, taps as i32, history as i32,
                    count as i32, conv.scale, stream.0)
            }, "prefill residual parity").unwrap();
            stream.sync().unwrap();
            let expected: Vec<_> = (start..start + count).flat_map(|position| {
                let x: Vec<_> = raw[position * width..][..width].iter().copied().map(from_bf16).collect();
                let mut out = conv.run(&mut serial, &x, None, position, 128);
                if add {
                    out.iter_mut().zip(&previous[position * width..][..width])
                        .for_each(|(dst, value)| *dst += from_bf16(*value));
                }
                out
            }).collect();
            assert_eq!(output.download::<f32>(count * width).unwrap(), expected);
            let routed: Vec<_> = raw[start * width..(start + count) * width].iter().copied()
                .map(|v| from_bf16(v) + 0.000_33).collect();
            let residual: Vec<_> = previous[start * width..(start + count) * width].iter().copied()
                .map(|v| from_bf16(v) - 0.029).collect();
            let history_rows = mlp.bf16_history(start, taps);
            let rounded = routed.iter().copied().map(to_bf16).collect::<Vec<_>>();
            let input = DevBuf::from_slice(&rounded).unwrap();
            let skip = DevBuf::from_slice(&residual).unwrap();
            let history = DevBuf::from_slice(&history_rows).unwrap();
            let output = DevBuf::new(count * width * 2).unwrap();
            let tail = DevBuf::new(count * width * 2).unwrap();
            check(unsafe {
                blaze_sys::blz_inkling_mlp_residual_conv(input.ptr(), skip.ptr(), history.ptr(), device_weights.ptr(),
                    output.ptr(), tail.ptr(), width as i32, taps as i32, (history_rows.len() / width) as i32,
                    count as i32, count as i32, conv.scale, stream.0)
            }, "prefill MLP residual parity").unwrap();
            stream.sync().unwrap();
            let expected: Vec<_> = rounded.chunks_exact(width).zip(residual.chunks_exact(width))
                .enumerate().flat_map(|(i, (row, skip))| {
                    let input = row.iter().copied().map(from_bf16).collect::<Vec<_>>();
                    conv.run(&mut mlp, &input, None, start + i, 128).into_iter().zip(skip)
                        .map(|(x, y)| to_bf16(x + y)).collect::<Vec<_>>()
                }).collect();
            assert_eq!(tail.download::<u16>(count * width).unwrap(), rounded);
            assert_eq!(output.download::<u16>(count * width).unwrap(), expected);
        }
    }

    #[test]
    #[ignore = "requires CUDA and AVX2/FMA"]
    fn prefill_kv_convolution_matches_cpu_with_history_and_dynamic_taps() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        gpu::set_device(0).unwrap();
        let stream = Stream::new().unwrap();
        let (heads, dim, taps, count) = (2, 128, 4, 16);
        let width = heads * dim;
        let scale = ((2 * taps + 1) as f32).sqrt().recip();
        let clip = 5.0;
        let eps = 1e-6;
        let signal = |i: usize| ((i * 103 % 199) as f32 - 99.0) / 79.0;
        let static_taps: Vec<_> = (0..taps * width).map(|i| signal(i + 17) / 2.0).collect();
        let weights = DevBuf::from_slice(&static_taps.repeat(2)).unwrap();
        let conv = Conv {
            taps: static_taps.chunks_exact(width).map(<[f32]>::to_vec).collect(),
            dynamic: None,
            scale,
        };
        let raw: Vec<_> = (0..(count + taps - 1) * width).map(|i| to_bf16(signal(i))).collect();
        let dynamic: Vec<_> = (0..(count + taps - 1) * heads * taps)
            .map(|i| to_bf16(signal(i + 13) / 5.0)).collect();
        let k = DevBuf::new(count * width * 2).unwrap();
        let v = DevBuf::new(count * width * 2).unwrap();
        for (history, use_dynamic) in [(0, false), (1, true), (3, true), (3, false)] {
            let raw_gpu = DevBuf::from_slice(&raw).unwrap();
            let dynamic_gpu = DevBuf::from_slice(&dynamic[history * heads * taps..]).unwrap();
            let dyn_ptr = if use_dynamic { dynamic_gpu.ptr() } else { std::ptr::null() };
            check(unsafe {
                blaze_sys::blz_inkling_prefill_conv(raw_gpu.ptr(), raw_gpu.ptr(), dyn_ptr, dyn_ptr, weights.ptr(),
                    k.ptr(), v.ptr(), heads as i32, taps as i32, history as i32, count as i32, scale, clip, eps, stream.0)
            }, "prefill convolution parity").unwrap();
            stream.sync().unwrap();
            let mut state = ConvState::new(taps, width, 0);
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for (pos, row) in raw.chunks_exact(width).take(history + count).enumerate() {
                let x: Vec<_> = row.iter().copied().map(from_bf16).collect();
                let d = use_dynamic.then(|| &dynamic[pos * heads * taps..(pos + 1) * heads * taps]);
                let y = conv.run(&mut state, &x, d, pos, dim);
                if pos >= history {
                    for head in y.chunks_exact(dim) {
                        let ks = clip_scale(head, clip);
                        let vs = inv_rms(head, eps);
                        keys.extend(head.iter().map(|x| to_bf16(x * ks)));
                        values.extend(head.iter().map(|x| to_bf16(x * vs)));
                    }
                }
            }
            assert_eq!(k.download::<u16>(count * width).unwrap(), keys);
            assert_eq!(v.download::<u16>(count * width).unwrap(), values);
        }
    }
}
