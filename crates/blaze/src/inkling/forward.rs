use super::{KvDtype, cpu};
use super::dspark::{Cache as DraftCache, Draft, Features};
use super::cuda::{AttentionClip, AttentionConv, AttentionResidual, AttentionRow, AttentionScratch, Compute, Dense, Kv, MlpResidual, MoeOutput, ProjectionInputs, RecurrentMessages, expert_size, from_bf16, prefill_attention_rows, prompt_stage_bytes, to_bf16};
use super::generate::Input;
use super::vision::Vision;
use super::weights::{Experts, read_float};
use crate::config::{InklingLoop, MlpConfig, ModelConfig};
use crate::expert_rg16;
use crate::expert_runtime::MoeEngine;
use crate::gpu;
use crate::layer_walk::LoopResiduals;
use crate::weights::{Checkpoint, Dtype};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::path::Path;
use std::time::{Duration, Instant};

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

struct ConvState {
    rows: Vec<Vec<f32>>,
}

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
    device_messages: bool,
    controllers: [Controller; 4],
}

impl LoopInputs {
    fn mix_attention_rows(&self, gpu: &mut Compute, qkv: &mut [Vec<Vec<f32>>],
                          rows: &[ForwardRow], norms: Option<&gpu::DevBuf>, bank_ready: &mut bool,
                          workers: Option<&rayon::ThreadPool>, eps: f32) -> Result<bool> {
        let controllers = self.controllers.first_chunk::<3>().unwrap();
        let width = qkv[0][0].len();
        if qkv.len() > 1
            && let [GateWeights::Device(q), GateWeights::Device(k), GateWeights::Device(v)] =
                controllers.each_ref().map(|c| &c.up)
        {
            if let Some(norms) = norms && gpu.recurrent_batch_limit(width) > 1 {
                if self.device_messages && (*bank_ready || rows.iter().all(|r| r.recurrent.messages().len() == 2))
                {
                    let bank: Vec<_> = if *bank_ready { Vec::new() } else {
                        rows.iter().map(|row| row.recurrent.messages()).collect()
                    };
                    let scale = (width as f32).sqrt().recip();
                    let prepare = |row: &ForwardRow| std::array::from_fn(|group| {
                        let logits = [0, 1].map(|i| dot(&row.recurrent.messages()[i], &self.queries[group]) * scale);
                        let peak = logits[0].max(logits[1]);
                        let exp = logits.map(|v| (v - peak).exp());
                        let sum = (exp[0] + exp[1]).recip();
                        exp.map(|v| v * sum)
                    });
                    let weights: Vec<_> = if let Some(pool) = workers {
                        pool.install(|| rows.par_iter().map(prepare).collect())
                    } else {
                        rows.iter().map(prepare).collect()
                    };
                    let messages = RecurrentMessages::Bank {
                        upload: (!*bank_ready).then_some(&bank),
                        weights: &weights,
                    };
                    if gpu.queue_recurrent_attention(controllers.each_ref().map(|c| &c.down), [q, k, v],
                        qkv, messages, norms, eps)?
                    {
                        *bank_ready = true;
                        return Ok(true);
                    }
                }
                let prepare = |row: &ForwardRow| {
                    std::array::from_fn(|i| mix_normalized(row.recurrent.messages(), &self.queries[i]))
                };
                let messages: Vec<[Vec<f32>; 3]> = if let Some(pool) = workers {
                    pool.install(|| rows.par_iter().map(prepare).collect())
                } else {
                    rows.iter().map(prepare).collect()
                };
                if gpu.queue_recurrent_attention(controllers.each_ref().map(|c| &c.down), [q, k, v],
                    qkv, RecurrentMessages::Mixed(&messages), norms, eps)?
                {
                    return Ok(true);
                }
            }
            let limit = gpu.controller_batch_limit::<3>(width).max(1);
            for (queries, rows) in qkv.chunks_mut(limit).zip(rows.chunks(limit)) {
                let messages: Vec<[Vec<f32>; 3]> = rows.iter().map(|row| {
                    std::array::from_fn(|i| mix_normalized(row.recurrent.messages(), &self.queries[i]))
                }).collect();
                let inputs: Vec<[Vec<f32>; 3]> = queries.iter().zip(&messages).map(|(qkv, msg)| {
                    std::array::from_fn(|i| Controller::input(&qkv[i], &msg[i], eps))
                }).collect();
                let input_refs: Vec<_> = inputs.iter().map(|r| r.each_ref().map(Vec::as_slice)).collect();
                let gates = gpu.controller_gates_batch(controllers.each_ref().map(|c| &c.down), [q, k, v], &input_refs)?;
                for ((x, msg), gates) in queries.iter_mut().flatten().zip(messages.iter().flatten()).zip(gates.chunks_exact(2 * width)) {
                    let (forget, gain) = gates.split_at(width);
                    cpu::controller_mix(x, msg, forget, gain);
                }
            }
        } else {
            for (qkv, row) in qkv.iter_mut().zip(rows) {
                self.mix_attention(gpu, qkv, row.recurrent.messages(), eps)?;
            }
        }
        Ok(false)
    }

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
        let gates = gpu.project_group(
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
    device_qkv_norms: Option<gpu::DevBuf>,
    norm_mlp: Vec<f32>,
    gated_norm: Option<gpu::DevBuf>,
    q_clip: Vec<f32>,
    rel_clip: Vec<f32>,
    device_clip: Option<gpu::DevBuf>,
    res_query: Vec<f32>,
    res_mlp: Vec<f32>,
    loop_inputs: Option<LoopInputs>,
    k_conv: Conv,
    v_conv: Conv,
    kv_taps: Option<gpu::DevBuf>,
    attn_conv: Conv,
    attn_conv_device: Option<gpu::DevBuf>,
    mlp_conv: Conv,
    mlp_conv_device: Option<gpu::DevBuf>,
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
            let queries = [row(messages.0, 0), row(messages.1, 0), row(messages.1, 1), row(messages.0, 1)];
            Some(LoopInputs {
                queries,
                device_messages: std::env::var_os("BLAZE_INKLING_DEVICE_LOOP_MESSAGES").is_some_and(|value| value == "1"),
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
        #[cfg(target_arch = "x86_64")]
        let avx_controller = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let avx_controller = false;
        let gated_norm = if avx_controller && s.hidden.is_multiple_of(32) && s.hidden <= 4096
            && loop_inputs.as_ref().is_some_and(|c| matches!(c.controllers[3].up, GateWeights::Device(_)))
            && std::env::var_os("BLAZE_INKLING_GPU_ROUTER").is_some_and(|v| v == "1")
        {
            Some(gpu::DevBuf::from_slice(&norm_mlp)?)
        } else { None };
        let mut rel_transposed = vec![0u16; rel.len()];
        for (i, row) in rel.chunks_exact(s.extent).enumerate() {
            for (j, &v) in row.iter().enumerate() {
                rel_transposed[j * s.relative + i] = to_bf16(v);
            }
        }
        let k_conv = Conv::load(ckpt, &t("attn.k_sconv"), kv, s, !local)?;
        let v_conv = Conv::load(ckpt, &t("attn.v_sconv"), kv, s, !local)?;
        let kv_taps = avx_controller.then(|| {
            let weights: Vec<_> = k_conv.taps.iter().chain(&v_conv.taps).flatten().copied().collect();
            gpu::DevBuf::from_slice(&weights)
        }).transpose()?;
        let q_clip = read("attn.q_l2_clip.scale", &[s.heads, s.dim])?;
        let rel_clip = read("attn.rel_l2_clip.scale", &[s.heads, s.relative])?;
        let device_clip = (avx_controller && s.relative < 32 && s.hidden <= s.q_width() && s.heads.is_multiple_of(8)
            && std::env::var_os("BLAZE_INKLING_DEVICE_ATTENTION").is_none_or(|value| value != "0"))
            .then(|| {
                let weights: Vec<_> = q_clip.iter().chain(&rel_clip).copied().collect();
                gpu::DevBuf::from_slice(&weights)
            }).transpose()?;
        let norm_q = read(if local { "attn_norm.weight" } else { "q_input_norm.weight" }, &[s.hidden])?;
        let norm_k = if local { Vec::new() } else { read("k_input_norm.weight", &[s.hidden])? };
        let norm_v = if local { Vec::new() } else { read("v_input_norm.weight", &[s.hidden])? };
        let device_qkv_norms = (avx_controller && s.hidden <= 4096 && s.hidden.is_multiple_of(32)
            && (std::env::var_os("BLAZE_INKLING_RECURRENT_QKV").is_some_and(|value| value == "1")
                || std::env::var_os("BLAZE_INKLING_BATCH_QKV").is_some_and(|value| value == "1"))
            && loop_inputs.as_ref().is_some_and(|c| c.controllers[..3].iter().all(|c| matches!(c.up, GateWeights::Device(_)))))
            .then(|| {
                let norms: Vec<_> = if local { norm_q.repeat(3) }
                    else { norm_q.iter().chain(&norm_k).chain(&norm_v).copied().collect() };
                gpu::DevBuf::from_slice(&norms)
            }).transpose()?;
        let attn_conv = Conv::load(ckpt, &t("attn_sconv"), s.hidden, s, false)?;
        let attn_conv_device = device_clip.as_ref().filter(|_| {
            std::env::var_os("BLAZE_INKLING_DEVICE_ATTN_RESIDUAL").is_some_and(|value| value == "1")
        }).map(|_| {
            gpu::DevBuf::from_slice(&attn_conv.taps.iter().flatten().copied().collect::<Vec<_>>())
        }).transpose()?;
        let mlp_conv = Conv::load(ckpt, &t("mlp_sconv"), s.hidden, s, false)?;
        let mlp_conv_device = std::env::var_os("BLAZE_INKLING_DEVICE_MLP_RESIDUAL")
            .is_some_and(|value| value == "1")
            .then(|| gpu::DevBuf::from_slice(&mlp_conv.taps.iter().flatten().copied().collect::<Vec<_>>()))
            .transpose()?;

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
            device_qkv_norms,
            norm_mlp,
            gated_norm,
            q_clip,
            rel_clip,
            device_clip,
            res_query,
            res_mlp: read_float(ckpt, &format!("{attn_res}.mlp_queries.{id}"), &[s.hidden])?,
            loop_inputs,
            k_conv,
            v_conv,
            kv_taps,
            attn_conv,
            attn_conv_device,
            mlp_conv,
            mlp_conv_device,
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
pub(super) enum Readout<'a> {
    None,
    Greedy,
    Verify(&'a [u32]),
    Scores,
}

pub(super) enum Prediction {
    Token(u32),
    Verified(usize, u32),
    Scores(Vec<f32>),
}

pub struct Sequence {
    draft: Option<DraftCache>,
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
    draft: Option<Draft>,
    serial_verify: bool,
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
    expert_layers: Option<Vec<usize>>,
    context: usize,
    global_kv_host: bool,
    kv_dtype: KvDtype,
    vision: Option<Vision>,
    prefill_chunk: usize,
    profile_prefill: bool,
    recurrent_qkv: bool,
    batch_qkv: bool,
    parallel_prefill: bool,
}

impl Model {
    pub fn load(path: &Path, opts: Options) -> Result<Self> {
        let Options {
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
        ensure!(dspark_tokens <= 7 && (dspark_tokens == 0 || dspark_tokens < prefill_chunk),
            "DSpark supports up to seven drafts and needs a prefill chunk larger than the draft count");
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
        let packed = std::env::var_os("BLAZE_INKLING_HYBRID_MOE").is_some_and(|value| value == "1");
        let mut layer_map = vec![0; layers.len()];
        let mut expert_layout: Vec<_> = (0..layers.len()).collect();
        expert_layout.sort_by_key(|&layer| placement[layer]);
        for (index, &layer) in expert_layout.iter().enumerate() {
            layer_map[layer] = index;
        }
        let store = packed.then(|| {
            ensure!(s.hidden.is_multiple_of(16) && s.intermediate.is_multiple_of(16)
                && s.top_k <= 16, "Inkling does not support this HybridMoe geometry");
            let threads = std::thread::available_parallelism().map_or(1, usize::from)
                .min(if gpus == 1 { 4 } else { 12 });
            expert_rg16::cached_store(&ckpt, &expert_layout, s.experts, s.hidden, s.intermediate, threads,
                |layer, expert, out| layers[layer].experts.to_rg16(&ckpt, expert, out))
        }).transpose()?;
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
        let mut expert_layer_start = 0;
        for rank in 0..gpus {
            let id = device + rank as i32;
            gpu::set_device(id)?;
            let scratch = AttentionScratch::new(s.heads, s.kv_heads, s.extent, s.relative, prefill_chunk, s.conv)?;
            let mut sizes = [0; 2];
            let mut visits = [0; 2];
            let mut kv_bytes = 0usize;
            let mut all_experts = 0usize;
            let mut expert_layer_count = 0;
            for (layer, &assigned) in layers.iter().zip(&placement) {
                if assigned == rank {
                    expert_layer_count += 1;
                    if !packed {
                        all_experts += expert_size(&layer.experts) * s.experts;
                    }
                }
            }
            if packed {
                all_experts = expert_rg16::record_bytes(s.hidden, s.intermediate) * s.experts * expert_layer_count;
            }
            for visit in &loops.visits {
                if placement[visit.layer] != rank {
                    continue;
                }
                let layer = &layers[visit.layer];
                let packing = layer.experts.gate_up.packing as usize;
                let slots = if layer.local {
                    (loops.window + prefill_attention_rows(prefill_chunk).saturating_sub(1).max(dspark_tokens)).min(context)
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
            let draft_reserve = if draft.is_some() && rank == last_device { DraftCache::bytes(context, kv_dtype) } else { 0 };
            let reserve = (512 << 20) + vision_reserve + draft_reserve + prompt_stage_bytes(sizes, prefill_chunk, cpu.is_some());
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
            let mut compute = Compute::new(
                (s.hidden * 2).max(s.q_width()),
                s.vocab.max((layers[0].query.rows + 2 * (s.kv_width() + s.kv_heads * s.conv)) * prefill_chunk
                    + 2 * s.kv_width() * (s.conv - 1)),
                s.hidden,
                s.intermediate,
                if packed { 0 } else { expert_bytes },
                if packed { [0; 2] } else { sizes },
                if packed { [0; 2] } else { visits },
                s.top_k,
                prefill_chunk,
                cpu.clone(),
            )?;
            if let Some(store) = &store {
                let end = expert_layer_start + expert_layer_count;
                let threads = cpu.as_ref().map_or(1, |pool| pool.current_num_threads().div_ceil(gpus));
                let engine = MoeEngine::packed(store, expert_layer_start..end, id, expert_bytes, s.top_k,
                    threads, prefill_chunk, expert_threads.is_none() && expert_bytes < all_experts, (s.gate_cap, s.up_cap))?;
                compute.attach_expert_engine(engine, s.hidden, prefill_chunk)?;
                expert_layer_start = end;
            }
            devices.push(Device { id, compute, scratch });
        }
        let norm_embed = read_float(&ckpt, "model.llm.embed_norm.weight", &[s.hidden])?;
        let norm_final = read_float(&ckpt, "model.llm.norm.weight", &[s.hidden])?;
        let query_final = read_float(&ckpt, "model.llm.attn_res.output_query", &[s.hidden])?;
        Ok(Self {
            draft,
            serial_verify: expert_threads.is_some(),
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
            expert_layers: packed.then_some(layer_map),
            context,
            global_kv_host: host_kv,
            kv_dtype,
            vision,
            prefill_chunk,
            profile_prefill: std::env::var_os("BLAZE_INKLING_PROFILE_PREFILL").is_some_and(|value| value == "1"),
            recurrent_qkv: std::env::var_os("BLAZE_INKLING_RECURRENT_QKV").is_some_and(|value| value == "1"),
            batch_qkv: std::env::var_os("BLAZE_INKLING_BATCH_QKV").is_some_and(|value| value == "1"),
            parallel_prefill: std::env::var_os("BLAZE_INKLING_PARALLEL_PREFILL").is_some_and(|value| value == "1"),
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
        let slack = self.draft.as_ref().map_or(0, |draft| draft.limit);
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
                    key: ConvState::new(s.conv, s.kv_width(), slack),
                    value: ConvState::new(s.conv, s.kv_width(), slack),
                    attention: ConvState::new(s.conv, s.hidden, slack),
                    mlp: ConvState::new(s.conv, s.hidden, slack),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Sequence {
            draft: self.draft.as_ref().map(|_| DraftCache::new(capacity, self.kv_dtype)).transpose()?,
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

    pub(super) fn draft_limit(&self) -> usize {
        self.draft.as_ref().map_or(0, |draft| draft.limit)
    }

    pub(super) fn draft_tokens(&mut self, seq: &Sequence, anchor: u32, limit: usize) -> Result<Vec<u32>> {
        let x = self.embedding(anchor)?;
        let device = &mut self.devices[*self.placement.last().unwrap()];
        gpu::set_device(device.id)?;
        self.draft.as_ref().unwrap().propose(&mut device.compute, seq.draft.as_ref().unwrap(), &self.ckpt,
            &self.head, anchor, x, seq.position, self.cfg.unpadded_vocab, limit)
    }

    pub(super) fn verify(&mut self, seq: &mut Sequence, anchor: u32, draft: &[u32], readout: Readout,
                         mut sample: impl FnMut(&[f32]) -> u32) -> Result<(usize,u32)> {
        if self.serial_verify || matches!(readout, Readout::Scores) {
            let mut token = anchor;
            let start = seq.position;
            let mut features = Vec::with_capacity(draft.len() + 1);
            for i in 0..=draft.len() {
                let row = ForwardRow::new(self.embedding(token)?);
                let next = match self.forward_rows(seq, vec![row], readout, Some(&mut features))? {
                    Some(Prediction::Token(id)) => id,
                    Some(Prediction::Scores(scores)) => sample(&scores),
                    _ => anyhow::bail!("missing DSpark target prediction"),
                };
                if draft.get(i) != Some(&next) {
                    let compute = &self.devices[*self.placement.last().unwrap()].compute;
                    self.draft.as_ref().unwrap().append(compute, seq.draft.as_mut().unwrap(), start, &features)?;
                    return Ok((i, next));
                }
                token = next;
            }
            unreachable!("the bonus prediction has no matching draft");
        }
        let rows = std::iter::once(anchor).chain(draft.iter().copied())
            .map(|t| self.embedding(t).map(ForwardRow::new)).collect::<Result<_>>()?;
        let Some(Prediction::Verified(kept, next)) = self.forward_rows(seq, rows, Readout::Verify(draft), None)? else {
            anyhow::bail!("missing DSpark target verdict");
        };
        Ok((kept,next))
    }

    pub fn encode_image(&self, bytes: &[u8]) -> Result<Vec<Vec<u16>>> {
        let vision = self.vision.as_ref().context("Inkling vision is disabled")?;
        let device = &self.devices[*self.placement.last().unwrap()];
        gpu::set_device(device.id)?;
        vision.encode(&device.compute, bytes)
    }

    pub(super) fn forward(&mut self, seq: &mut Sequence, token: u32, readout: Readout) -> Result<Option<Prediction>> {
        let x = self.embedding(token)?;
        self.forward_rows(seq, vec![ForwardRow::new(x)], readout, None)
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
        self.forward_rows(seq, rows, readout, None)
    }

    fn forward_rows(&mut self, seq: &mut Sequence, mut rows: Vec<ForwardRow>, readout: Readout,
                    deferred_features: Option<&mut Vec<Features>>) -> Result<Option<Prediction>> {
        ensure!(
            !rows.is_empty() && rows.len() <= seq.capacity.saturating_sub(seq.position),
            "Inkling chunk exceeds the sequence context limit of {}",
            seq.capacity
        );
        let s = &self.cfg;
        let mut timings = [Duration::ZERO; 5];
        let mut phase_start = (self.profile_prefill && rows.len() > 1).then(Instant::now);
        let mut phase = |index: usize| {
            if let Some(start) = &mut phase_start {
                let now = Instant::now();
                timings[index] += now - *start;
                *start = now;
            }
        };
        let mut active = None;
        let mut bank_ready = false;
        let mut features = seq.draft.as_ref().map(|_| vec![Features::default(); rows.len()]);
        // Visit each physical weight bank once per chunk. Per-visit attention and convolution still consume
        // tokens in order, so batching the traversal has the same state transitions as token-wise decoding.
        for (index, (visit, state)) in self.loops.visits.iter().zip(&mut seq.visits).enumerate() {
            if let Some(f) = &mut features && let Some(tap) = [5, 23, 41].iter().position(|&v| v == index) {
                for (dst,row) in f.iter_mut().zip(&rows) { dst[tap].clone_from(&row.x); }
            }
            let id = visit.layer;
            if id == self.loops.core.start {
                bank_ready = false;
            }
            let rank = self.placement[id];
            let device = &mut self.devices[rank];
            if active != Some(rank) {
                gpu::set_device(device.id)?;
                active = Some(rank);
            }
            let compute = &mut device.compute;
            let layer = &self.layers[id];
            let workers = if self.parallel_prefill && rows.len() >= 128
                && !matches!(readout, Readout::Verify(_)) { compute.host_workers() } else { None };
            if !matches!(readout, Readout::Verify(_))
                && let Some(next) = self.loops.visits.get(index + 1)
                && self.placement[next.layer] == rank
            {
                compute.prefetch_experts(&self.ckpt, &self.layers[next.layer].experts, id, next.layer, rows.len())?;
            }
            let mut residuals = Vec::with_capacity(rows.len());
            let mut activations = Vec::with_capacity(rows.len());
            let mut q_inputs = Vec::with_capacity(rows.len());
            let mut k_inputs = Vec::with_capacity(rows.len());
            let mut v_inputs = Vec::with_capacity(rows.len());
            let controller = (visit.pass > 0).then(|| layer.loop_inputs.as_ref().unwrap());
            let recurrent = controller.zip(layer.device_qkv_norms.as_ref())
                .filter(|_| self.recurrent_qkv && rows.len() == 1 && compute.recurrent_qkv_fits(s.hidden));
            let restart = id == self.loops.core.start;
            let end = id == self.loops.core.end;
            let prepare_qkv = |row: &mut ForwardRow| {
                let ForwardRow { x, blocks, recurrent } = row;
                if restart {
                    recurrent.enter(visit.pass, x, blocks, |hidden| normalized(hidden, None, s.eps));
                }
                if end {
                    recurrent.leave();
                }
                let mut qkv = if id == 0 {
                    vec![x.clone()]
                } else {
                    mix(blocks, x, &layer.res_query, s.eps)
                };
                if qkv.len() == 1 && controller.is_some() {
                    qkv.resize(3, qkv[0].clone());
                }
                qkv
            };
            let mut qkv_rows: Vec<_> = if let Some(pool) = &workers {
                pool.install(|| rows.par_iter_mut().map(prepare_qkv).collect())
            } else {
                rows.iter_mut().map(prepare_qkv).collect()
            };
            let count = rows.len();
            let on_device = count > 1 && self.prefill_chunk >= s.conv && !matches!(readout, Readout::Verify(_));
            let normalized_on_gpu = if recurrent.is_none() && let Some(c) = controller {
                let norms = layer.device_qkv_norms.as_ref()
                    .filter(|_| self.batch_qkv && on_device && layer.device_clip.is_some()
                        && layer.kv_taps.is_some());
                c.mix_attention_rows(compute, &mut qkv_rows, &rows, norms, &mut bank_ready, workers.as_deref(), s.eps)?
            } else { false };
            for qkv in qkv_rows.iter().filter(|_| recurrent.is_none() && !normalized_on_gpu) {
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
            phase(0);
            let block_start = id % s.block == 0;
            let (attention, residual_ready) = if on_device
                && let (Some(clip), Some(weights)) = (&layer.device_clip, &layer.kv_taps)
            {
                let conv = AttentionConv {
                    weights,
                    history: [&state.key.bf16_history(seq.position, s.conv), &state.value.bf16_history(seq.position, s.conv)],
                    dynamic: None,
                    taps: s.conv,
                    scale: layer.k_conv.scale,
                    clip: s.qk_clip,
                    eps: s.eps,
                };
                let clip = AttentionClip {
                    weights: clip,
                    relative: s.relative,
                    threshold: s.qk_clip,
                    relative_threshold: s.rel_clip,
                    log_floor: s.log_floor,
                    log_alpha: if layer.local { 0.0 } else { s.log_alpha },
                };
                let previous: Vec<_> = if block_start || layer.attn_conv_device.is_none() { Vec::new() }
                    else { rows.iter().map(|row| row.x.as_slice()).collect() };
                let history = layer.attn_conv_device.as_ref().map(|_| state.attention.bf16_history(seq.position, s.conv));
                let residual = layer.attn_conv_device.as_ref().zip(history.as_ref()).map(|(weights, history)| AttentionResidual {
                    weights,
                    history,
                    previous: &previous,
                    scale: layer.attn_conv.scale,
                });
                let tail = state.key.rows.len().min(count);
                let output = if let (Some(k), Some(v)) = (&layer.k_conv.dynamic, &layer.v_conv.dynamic) {
                    let inputs = if normalized_on_gpu { ProjectionInputs::Recurrent(count) }
                        else { ProjectionInputs::Host([&q_refs, &k_refs, &v_refs, &k_refs, &v_refs]) };
                    device.scratch.project_attention(compute, &state.kv,
                        [&layer.query, &layer.key, &layer.value, k, v], inputs,
                        &layer.bias, &layer.output, seq.position, conv, clip, tail, residual)
                } else {
                    let inputs = if normalized_on_gpu { ProjectionInputs::Recurrent(count) }
                        else { ProjectionInputs::Host([&q_refs, &k_refs, &v_refs]) };
                    device.scratch.project_attention(compute, &state.kv,
                        [&layer.query, &layer.key, &layer.value], inputs,
                        &layer.bias, &layer.output, seq.position, conv, clip, tail, residual)
                };
                if normalized_on_gpu && output.is_err() {
                    let _ = compute.stream.sync();
                }
                let (output, [key, value, attn]) = output?;
                let start = seq.position + count - tail;
                state.key.remember_bf16(start, s.kv_width(), &key);
                state.value.remember_bf16(start, s.kv_width(), &value);
                state.attention.remember_bf16(start, s.hidden, &attn);
                (output, layer.attn_conv_device.is_some())
            } else {
                let (qkv, conv) = if let Some((controller, norms)) = recurrent {
                    let x = std::array::from_fn(|i| qkv_rows[0][i].as_slice());
                    let messages: [_; 3] = std::array::from_fn(|i| mix_normalized(rows[0].recurrent.messages(), &controller.queries[i]));
                    let inputs: [_; 3] = std::array::from_fn(|i| Controller::input(x[i], &messages[i], s.eps));
                    let controllers = controller.controllers.first_chunk::<3>().unwrap();
                    let down = controllers.each_ref().map(|c| &c.down);
                    let up = controllers.each_ref().map(|c| {
                        let GateWeights::Device(w) = &c.up else { unreachable!() };
                        w
                    });
                    let messages = messages.each_ref().map(Vec::as_slice);
                    let inputs = inputs.each_ref().map(Vec::as_slice);
                    if let (Some(k), Some(v)) = (&layer.k_conv.dynamic, &layer.v_conv.dynamic) {
                        let [q, key, value, k_taps, v_taps] = compute.project_recurrent(
                            [&layer.query, &layer.key, &layer.value, k, v], down, up, x, messages, inputs, norms, s.eps,
                        )?;
                        ([q, key, value], Some([k_taps, v_taps]))
                    } else {
                        let qkv = compute.project_recurrent(
                            [&layer.query, &layer.key, &layer.value], down, up, x, messages, inputs, norms, s.eps,
                        )?;
                        (qkv, None)
                    }
                } else if let (Some(k), Some(v)) = (&layer.k_conv.dynamic, &layer.v_conv.dynamic) {
                    let [q, key, value, k_taps, v_taps] = compute.project_rows(
                        [&layer.query, &layer.key, &layer.value, k, v],
                        [&q_refs, &k_refs, &v_refs, &k_refs, &v_refs], matches!(readout, Readout::Verify(_)),
                    )?;
                    ([q, key, value], Some([k_taps, v_taps]))
                } else {
                    (compute.project_rows([&layer.query, &layer.key, &layer.value], [&q_refs, &k_refs, &v_refs], matches!(readout, Readout::Verify(_)))?, None)
                };
                let [q_batch, k_batch, v_batch] = qkv;
                let kv_weights = layer.kv_taps.as_ref().filter(|_| count > 1);
                let kv_history = kv_weights.map(|_| [
                    state.key.bf16_history(seq.position, s.conv),
                    state.value.bf16_history(seq.position, s.conv),
                ]);
                let mut queries = Vec::with_capacity(count);
                for offset in 0..count {
                    let position = seq.position + offset;
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
                    if kv_weights.is_some() {
                        queries.push(AttentionRow { q, gate, sink, relative: rel,
                            key: (&k_batch[kv_range.clone()]).into(), value: (&v_batch[kv_range]).into() });
                        continue;
                    }
                    let key: Vec<_> = k_batch[kv_range.clone()].iter().copied().map(from_bf16).collect();
                    let value: Vec<_> = v_batch[kv_range].iter().copied().map(from_bf16).collect();
                    let taps = |i: usize| conv.as_ref().map(|w| {
                        let width = s.kv_heads * s.conv;
                        &w[i][offset * width..(offset + 1) * width]
                    });
                    let key = layer.k_conv.run(&mut state.key, &key, taps(0), position, s.dim);
                    let value = layer.v_conv.run(&mut state.value, &value, taps(1), position, s.dim);
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
                    queries.push(AttentionRow { q, gate, sink, relative: rel, key: key.into(), value: value.into() });
                }
                let kv_conv = kv_weights.zip(kv_history.as_ref()).map(|(weights, history)| AttentionConv {
                    weights,
                    history: history.each_ref().map(Vec::as_slice),
                    dynamic: conv.as_ref().map(|c| c.each_ref().map(Vec::as_slice)),
                    taps: s.conv,
                    scale: layer.k_conv.scale,
                    clip: s.qk_clip,
                    eps: s.eps,
                });
                let attention = device.scratch.run_rows(
                    compute, &state.kv, &queries, &layer.bias, &layer.output, seq.position, kv_conv,
                    matches!(readout, Readout::Verify(_)),
                )?;
                if kv_weights.is_some() {
                    state.key.remember_bf16(seq.position, s.kv_width(), &k_batch);
                    state.value.remember_bf16(seq.position, s.kv_width(), &v_batch);
                }
                (attention, false)
            };
            phase(1);
            let prepare_mlp = |offset: usize, row: &mut ForwardRow| {
                let position = seq.position + offset;
                let attn = &attention[offset * s.hidden..][..s.hidden];
                let mut residual = if residual_ready {
                    attn.to_vec()
                } else {
                    layer.attn_conv.apply(attn, None, s.conv.min(position + 1), s.dim, |lag| {
                        if let Some(earlier) = offset.checked_sub(lag) {
                            &attention[earlier * s.hidden..][..s.hidden]
                        } else {
                            &state.attention.rows[(position - lag) % state.attention.rows.len()]
                        }
                    })
                };
                if block_start {
                    row.blocks.push(normalized(&row.x, None, s.eps));
                } else if !residual_ready {
                    residual.iter_mut().zip(&row.x).for_each(|(v, &x)| *v += x);
                }
                let mixed = mix(&row.blocks, &residual, &layer.res_mlp, s.eps).remove(0);
                let message = controller.map(|c| mix_normalized(row.recurrent.messages(), &c.queries[3]));
                let activation = if message.is_none() {
                    normalized(&mixed, Some(&layer.norm_mlp), s.eps).into_iter().map(round).collect()
                } else {
                    mixed
                };
                (residual, activation, message)
            };
            let prepared: Vec<_> = if let Some(pool) = &workers {
                pool.install(|| rows.par_iter_mut().enumerate().map(|(i, row)| prepare_mlp(i, row)).collect())
            } else {
                rows.iter_mut().enumerate().map(|(i, row)| prepare_mlp(i, row)).collect()
            };
            if !residual_ready {
                state.attention.remember_f32(seq.position, s.hidden, &attention);
            }
            let mut inputs = Vec::new();
            let mut mixed = Vec::new();
            let mut messages_batch = Vec::new();
            let limit = compute.gate_batch_limit(s.hidden);
            let mut gated_logits = None;
            let retain_activations = workers.is_some() && compute.retain_prefill_activations()
                && layer.gated_norm.is_some() && matches!(layer.router, Router::Device(_))
                && controller.is_some_and(|c| matches!(c.controllers[3].up, GateWeights::Device(_)));
            for (offset, (residual, mut mlp_in, message)) in prepared.into_iter().enumerate() {
                if let (Some(c), Some(msg)) = (controller, message) {
                    let gate = &c.controllers[3];
                    if let (Some(norm), GateWeights::Device(up), Router::Device(router)) =
                        (&layer.gated_norm, &gate.up, &layer.router)
                    {
                        if workers.is_none() {
                            inputs.extend(Controller::input(&mlp_in, &msg, s.eps));
                        }
                        mixed.extend(mlp_in);
                        messages_batch.extend(msg);
                        residuals.push(residual);
                        if offset+1 == count || mixed.len() >= limit*s.hidden {
                            let retain_at = retain_activations.then(|| offset + 1 - mixed.len() / s.hidden);
                            let (activation, logits) = compute.gated_router(
                                &gate.down, up, norm, router, workers.is_none().then_some(inputs.as_slice()),
                                &mixed, &messages_batch, s.eps, retain_at,
                            )?;
                            if count == 1 { activations.push(activation); }
                            else { activations.extend(activation.chunks_exact(s.hidden).map(<[f32]>::to_vec)); }
                            gated_logits.get_or_insert_with(Vec::new).extend(logits);
                            inputs.clear();
                            mixed.clear();
                            messages_batch.clear();
                        }
                        continue;
                    }
                    gate.merge(compute, &mut mlp_in, &msg, s.eps)?;
                    mlp_in = normalized(&mlp_in, Some(&layer.norm_mlp), s.eps).into_iter().map(round).collect();
                }
                activations.push(mlp_in);
                residuals.push(residual);
            }
            let route_rows = |logits: &[f32]| {
                let route = |row: &[f32]| layer.routes(row.iter().copied(), s);
                if let Some(pool) = &workers {
                    pool.install(|| logits.par_chunks_exact(s.experts).map(route).collect())
                } else {
                    logits.chunks_exact(s.experts).map(route).collect()
                }
            };
            let routes: Vec<_> = match (&layer.router, gated_logits) {
                (_, Some(logits)) => route_rows(&logits),
                (Router::Device(w), None) => route_rows(&compute.router_logits(w, &activations)?),
                (Router::Host(w), None) => activations.iter().map(|x| {
                    layer.routes(w.chunks_exact(s.hidden).map(|row| dot(row, x)), s)
                }).collect(),
            };
            phase(2);
            let mlp_history = layer.mlp_conv_device.as_ref().filter(|_| count > 1)
                .map(|_| state.mlp.bf16_history(seq.position, s.conv));
            let finish = layer.mlp_conv_device.as_ref().zip(mlp_history.as_ref()).map(|(weights, history)| MlpResidual {
                weights, history, residuals: &residuals, taps: s.conv,
                tail: state.mlp.rows.len().min(count), scale: layer.mlp_conv.scale,
            });
            let moe = if let Some(layers) = &self.expert_layers {
                compute.hybrid_moe(layers[id], (!retain_activations).then_some(activations.as_slice()),
                    &routes, s.hidden, s.top_k, matches!(readout, Readout::Verify(_)))?
            } else if matches!(readout, Readout::Verify(_)) {
                let mut output = Vec::with_capacity(rows.len()*s.hidden);
                for (x, route) in activations.iter().zip(&routes) {
                    output.extend(compute.moe(&self.ckpt,&layer.experts,id,x,route,s.gate_cap,s.up_cap)?);
                }
                MoeOutput::Routed(output)
            } else { compute.moe_batch(
                &self.ckpt,
                &layer.experts,
                id,
                (!retain_activations).then_some(activations.as_slice()),
                &routes,
                (s.gate_cap, s.up_cap),
                s.experts,
                finish,
            )? };
            phase(3);
            match moe {
                MoeOutput::Routed(mut moe) => {
                    for (offset, ((row, residual), moe)) in rows.iter_mut().zip(residuals).zip(moe.chunks_exact_mut(s.hidden)).enumerate() {
                        moe.iter_mut().for_each(|v| *v = round(*v));
                        let mlp = layer.mlp_conv.run(&mut state.mlp, moe, None, seq.position + offset, s.dim);
                        row.x = residual.into_iter().zip(mlp).map(|(a, b)| round(a + b)).collect();
                    }
                }
                MoeOutput::Residual { hidden, history } => {
                    let start = seq.position + count - history.len() / s.hidden;
                    state.mlp.remember_bf16(start, s.hidden, &history);
                    for (row, computed) in rows.iter_mut().zip(hidden.chunks_exact(s.hidden)) {
                        row.x.iter_mut().zip(computed).for_each(|(dst, &src)| *dst = from_bf16(src));
                    }
                }
            }
            phase(4);
        }
        if phase_start.is_some() {
            let [qkv_input, attention, routing, experts, mlp_residual] = timings.map(|time| time.as_secs_f64() * 1e3);
            crate::eprintln!("Inkling prefill {} rows: QKV input {qkv_input:.2} ms, attention {attention:.2} ms, routing {routing:.2} ms, experts {experts:.2} ms, MLP residual {mlp_residual:.2} ms", rows.len());
        }
        let normalize = |row: &ForwardRow| {
            let x = mix(&row.blocks, &row.x, &self.query_final, s.eps).remove(0);
            normalized(&x, Some(&self.norm_final), s.eps)
        };
        if let Some(f) = &mut features {
            for (dst,row) in f.iter_mut().zip(&rows) { dst[3] = normalize(row); }
        }
        let compute = &mut self.devices[*self.placement.last().unwrap()].compute;
        let mut consumed = rows.len();
        let result = if let Readout::Verify(draft) = readout {
            let mut verdict = None;
            for (i, f) in features.as_ref().unwrap().iter().enumerate() {
                let next = compute.argmax(&self.head,&f[3],s.unpadded_vocab)?;
                if draft.get(i) != Some(&next) {
                    consumed = i + 1;
                    verdict = Some(Prediction::Verified(i, next));
                    break;
                }
            }
            verdict
        } else if matches!(readout, Readout::None) {
            None
        } else {
            let row = rows.last().unwrap();
            let x = features.as_ref().map_or_else(|| normalize(row), |f| f.last().unwrap()[3].clone());
            // The model's tanh softcap is monotonic, so greedy decoding can select on the BF16 head output.
            if matches!(readout, Readout::Greedy) {
                Some(Prediction::Token(compute.argmax(&self.head, &x, s.unpadded_vocab)?))
            } else {
                let mut logits = compute.linear(&self.head, &x)?;
                logits.truncate(s.unpadded_vocab);
                cpu::softcap(&mut logits, s.logit_cap);
                Some(Prediction::Scores(logits))
            }
        };
        if let (Some(draft), Some(cache), Some(f)) = (&self.draft, &mut seq.draft, features) {
            if let Some(pending) = deferred_features {
                pending.extend(f.into_iter().take(consumed));
            } else {
                draft.append(compute, cache, seq.position, &f[..consumed])?;
            }
        }
        seq.position += consumed;
        Ok(result)
    }
}

struct ForwardRow {
    x: Vec<f32>,
    blocks: Vec<Vec<f32>>,
    recurrent: LoopResiduals<Vec<f32>>,
}

impl ForwardRow {
    fn new(x: Vec<f32>) -> Self {
        Self {
            x,
            blocks: Vec::new(),
            recurrent: LoopResiduals::new(),
        }
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
            let input = DevBuf::from_slice(&routed).unwrap();
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
            let rounded = routed.iter().copied().map(to_bf16).collect::<Vec<_>>();
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
