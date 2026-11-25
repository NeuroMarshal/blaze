//! The model's own `config.json`, read into what the graph builder needs. Unknown fields stay reachable through
//! `raw`; shapes that the config leaves implicit (e.g. the gated attention output) are taken from the weights.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use std::ops::Range;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Softmax attention over the KV cache.
    Full,
    /// Gated DeltaNet linear attention with a recurrent state.
    Linear,
}

#[derive(Debug, Clone)]
pub struct AttnConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone)]
pub struct GdnConfig {
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
    /// Gate of the output norm: sigmoid(z) when the config's output_gate_type is "sigmoid", else silu(z).
    pub sigmoid_gate: bool,
}

#[derive(Debug, Clone)]
pub enum MlpConfig {
    Dense { intermediate: usize },
    Moe { experts: usize, top_k: usize, intermediate: usize, shared_intermediate: usize, norm_topk: bool },
}

#[derive(Debug, Clone)]
pub enum RopeScaling {
    None,
    Yarn { factor: f64, original_max: usize, beta_fast: f64, beta_slow: f64 },
}

#[derive(Debug, Clone)]
pub struct RopeConfig {
    pub theta: f64,
    /// Rotated leading dims of each head (head_dim * partial_rotary_factor).
    pub rotary_dim: usize,
    pub mrope_section: Vec<usize>,
    pub mrope_interleaved: bool,
    pub scaling: RopeScaling,
}

/// Hyper-connections (Qwen4-Exp): `count` residual streams of `hidden` each, low-rank gated mixing.
#[derive(Debug, Clone)]
pub struct HcConfig {
    pub count: usize,
    pub lowrank: usize,
}

/// Per-layer n-gram embedding (Qwen4-Exp PLE).
#[derive(Debug, Clone)]
pub struct PleConfig {
    /// Decoder layers that add a PLE embedding to their streams.
    pub layers: Vec<usize>,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub embed_dim: usize,
    pub conv_kernel: usize,
}

/// Slots of one row of the QSA selection table (the selected keys of one query row).
pub const QSA_TABLE_STRIDE: usize = 2056;

/// Qwen sparse attention (compressed indexer) of the softmax layers.
#[derive(Debug, Clone)]
pub struct QsaConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub budget: usize,
    pub compress_ratio: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerVisit {
    pub layer: usize,
    pub pass: usize,
}

#[derive(Debug, Clone)]
pub struct InklingLoop {
    pub core: Range<usize>,
    pub passes: usize,
    pub gate_rank: usize,
    pub variable: bool,
    pub window: usize,
    local: Vec<bool>,
    /// Indexed in execution order. KV and convolution state are indexed by visit, weights by layer.
    pub visits: Vec<LayerVisit>,
}

impl InklingLoop {
    fn parse(t: &Value, layers: usize) -> Result<Self> {
        let c = t.get("loop_config").context("Inkling: missing loop_config")?;
        let core = need(c, "core_start")?..need(c, "core_end")?;
        let passes = need(c, "num_passes")?;
        let gate_rank = need(c, "gate_rank")?;
        let window = need(t, "sliding_window_size")?;
        anyhow::ensure!(core.start < core.end && core.end <= layers, "Inkling: invalid core layer range {core:?}");
        anyhow::ensure!(passes > 0 && gate_rank > 0 && window > 0, "Inkling: loop passes, gate rank and sliding window must be positive");
        let ids = t.get("local_layer_ids").and_then(Value::as_array).context("Inkling: missing local_layer_ids")?;
        let mut local = vec![false; layers];
        for id in ids {
            let id = id.as_u64().and_then(|id| usize::try_from(id).ok()).context("Inkling: invalid local attention layer ID")?;
            let slot = local.get_mut(id).with_context(|| format!("Inkling: local attention layer {id} is out of range"))?;
            anyhow::ensure!(!*slot, "Inkling: duplicate local attention layer {id}");
            *slot = true;
        }
        let count = core.len().checked_mul(passes).and_then(|n| n.checked_add(layers - core.len()))
            .context("Inkling: loop visit count overflows")?;
        let visit = |layer| LayerVisit { layer, pass: 0 };
        let mut visits = Vec::new();
        visits.try_reserve_exact(count).context("Inkling: allocating layer visits")?;
        visits.extend((0..core.start).map(visit));
        for pass in 0..passes {
            visits.extend(core.clone().map(|layer| LayerVisit { layer, pass }));
        }
        visits.extend((core.end..layers).map(visit));
        Ok(Self { core, passes, gate_rank, variable: c["is_varloop"].as_bool().unwrap_or(false), window, local, visits })
    }

    pub fn is_local(&self, layer: usize) -> bool {
        self.local[layer]
    }

    pub fn global_visits(&self) -> usize {
        self.visits.iter().filter(|v| !self.is_local(v.layer)).count()
    }
}

pub fn is_inkling_loop(root: &Value) -> bool {
    root["model_type"].as_str() == Some("inkling2_loop")
        || root["text_config"]["model_type"].as_str() == Some("inkling2_loop_text")
        || root["model_type"].as_str() == Some("inkling2_loop_text")
        || matches!(root["architectures"][0].as_str(), Some("Inkling2LoopForConditionalGeneration" | "Inkling2LoopForCausalLM"))
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub hidden: usize,
    pub layers: Vec<LayerKind>,
    pub vocab: usize,
    pub rms_eps: f32,
    pub tie_embeddings: bool,
    pub max_positions: usize,
    pub attn: AttnConfig,
    pub gdn: Option<GdnConfig>,
    pub mlp: MlpConfig,
    pub rope: RopeConfig,
    pub mtp_layers: usize,
    pub hc: Option<HcConfig>,
    pub ple: Option<PleConfig>,
    pub qsa: Option<QsaConfig>,
    pub inkling_loop: Option<InklingLoop>,
    pub eos: Option<i64>,
    /// The text part of the config (`text_config` of multimodal checkpoints).
    pub raw: Value,
}

fn usize_of(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

fn need(v: &Value, key: &str) -> Result<usize> {
    usize_of(v, key).ok_or_else(|| anyhow!("config: missing integer `{key}`"))
}

impl ModelConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::from_json(&Self::read_json(dir)?)
    }

    pub(crate) fn read_json(dir: &Path) -> Result<Value> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn from_json(root: &Value) -> Result<Self> {
        let architecture = root
            .get("architectures")
            .and_then(|a| a.get(0))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let t = root.get("text_config").unwrap_or(root).clone();
        let hidden = need(&t, "hidden_size")?;
        let n = need(&t, "num_hidden_layers")?;
        let inkling_loop = is_inkling_loop(root).then(|| InklingLoop::parse(&t, n)).transpose()?;
        let layers = match t.get("layer_types").and_then(Value::as_array) {
            Some(kinds) => kinds
                .iter()
                .map(|k| match k.as_str() {
                    // Qwen4-Exp names its softmax layers by their sparse indexer (QsaConfig)
                    Some("full_attention" | "attention" | "qwen_sparse_attention") => Ok(LayerKind::Full),
                    Some("linear_attention") => Ok(LayerKind::Linear),
                    other => Err(anyhow!("config: unknown layer type {other:?}")),
                })
                .collect::<Result<Vec<_>>>()?,
            None => match usize_of(&t, "full_attention_interval").filter(|&n| n > 0) {
                Some(every) => (0..n).map(|i| if (i + 1) % every == 0 { LayerKind::Full } else { LayerKind::Linear }).collect(),
                None => vec![LayerKind::Full; n],
            },
        };
        if layers.len() != n {
            bail!("config: {} layer types for {n} layers", layers.len());
        }
        let heads = need(&t, "num_attention_heads")?;
        anyhow::ensure!(heads > 0, "config: num_attention_heads must be positive");
        let attn = AttnConfig {
            heads,
            kv_heads: usize_of(&t, "num_key_value_heads").unwrap_or(heads),
            head_dim: usize_of(&t, "head_dim").unwrap_or(hidden / heads),
        };
        let gdn = if layers.contains(&LayerKind::Linear) {
            Some(GdnConfig {
                k_heads: need(&t, "linear_num_key_heads")?,
                v_heads: need(&t, "linear_num_value_heads")?,
                k_dim: need(&t, "linear_key_head_dim")?,
                v_dim: need(&t, "linear_value_head_dim")?,
                conv_kernel: usize_of(&t, "linear_conv_kernel_dim").unwrap_or(4),
                sigmoid_gate: t.get("output_gate_type").and_then(Value::as_str) == Some("sigmoid"),
            })
        } else {
            None
        };
        let mlp = if inkling_loop.is_some() {
            let experts = need(&t, "n_routed_experts")?;
            let top_k = need(&t, "num_experts_per_tok")?;
            anyhow::ensure!(top_k > 0 && top_k <= experts, "Inkling: invalid top-k {top_k} for {experts} experts");
            MlpConfig::Moe {
                experts,
                top_k,
                intermediate: need(&t, "intermediate_size")?,
                shared_intermediate: 0,
                norm_topk: true,
            }
        } else {
            match usize_of(&t, "num_experts") {
                Some(experts) if experts > 0 => MlpConfig::Moe {
                    experts,
                    top_k: need(&t, "num_experts_per_tok")?,
                    intermediate: need(&t, "moe_intermediate_size")?,
                    shared_intermediate: usize_of(&t, "shared_expert_intermediate_size").unwrap_or(0),
                    norm_topk: t.get("norm_topk_prob").and_then(Value::as_bool).unwrap_or(true),
                },
                _ => MlpConfig::Dense { intermediate: need(&t, "intermediate_size")? },
            }
        };
        let rope = if inkling_loop.is_some() {
            RopeConfig { theta: 0.0, rotary_dim: 0, mrope_section: Vec::new(), mrope_interleaved: false, scaling: RopeScaling::None }
        } else {
            Self::rope(&t, attn.head_dim)?
        };
        let hc = usize_of(&t, "hc_count").filter(|&c| c > 1).map(|count| HcConfig {
            count,
            lowrank: usize_of(&t, "hc_lowrank").unwrap_or(16),
        });
        let ple = match t.get("ple_layer_ids").and_then(Value::as_array) {
            Some(ids) if !ids.is_empty() => Some(PleConfig {
                // SGLang's _has_ple: layer_id + 1 in ple_layer_ids
                layers: ids.iter().filter_map(Value::as_u64).filter(|&i| i >= 1).map(|i| i as usize - 1).collect(),
                ngram_size: need(&t, "ngram_size")?,
                heads_per_ngram: need(&t, "heads_per_ngram")?,
                embed_dim: need(&t, "ple_embed_dim")?,
                conv_kernel: usize_of(&t, "ple_conv_kernel_size").unwrap_or(4),
            }),
            _ => None,
        };
        let qsa = usize_of(&t, "indexer_n_heads").map(|heads| -> Result<QsaConfig> {
            let (budget, ratio) = (need(&t, "indexer_budget")?, need(&t, "indexer_compress_ratio")?);
            // a select row holds up to (budget / ratio) * ratio + ratio - 1 slots (blz_qsa_select: top blocks of
            // `ratio` keys and the trailing partial block) in a table row of QSA_TABLE_STRIDE
            let blocks = budget / ratio.max(1);
            if ratio == 0 || blocks == 0 || blocks > 2048 || blocks * ratio + ratio - 1 > QSA_TABLE_STRIDE {
                anyhow::bail!("indexer_budget {budget} / indexer_compress_ratio {ratio}: a selection of at most 2048 blocks and \
                               {QSA_TABLE_STRIDE} slots is supported");
            }
            Ok(QsaConfig { heads, head_dim: need(&t, "indexer_head_dim")?, budget, compress_ratio: ratio })
        });
        let qsa = match qsa {
            Some(q) => Some(q?),
            None => None,
        };
        let eos = t.get("eos_token_id").or_else(|| root.get("eos_token_id")).and_then(Value::as_i64);
        Ok(Self {
            architecture,
            hidden,
            layers,
            vocab: need(&t, "vocab_size")?,
            rms_eps: t.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            tie_embeddings: root
                .get("tie_word_embeddings")
                .or_else(|| t.get("tie_word_embeddings"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            max_positions: usize_of(&t, "max_position_embeddings")
                .or_else(|| inkling_loop.as_ref().and_then(|_| usize_of(&t, "model_max_length")))
                .unwrap_or(32768),
            attn,
            gdn,
            mlp,
            rope,
            mtp_layers: usize_of(&t, "mtp_num_hidden_layers")
                .or_else(|| inkling_loop.as_ref().and_then(|_| usize_of(&root["mtp_config"], "num_hidden_layers")))
                .unwrap_or(0),
            hc,
            ple,
            qsa,
            inkling_loop,
            eos,
            raw: t,
        })
    }

    fn rope(t: &Value, head_dim: usize) -> Result<RopeConfig> {
        // Newer configs nest everything in rope_parameters; older ones use rope_theta / rope_scaling.
        let p = t.get("rope_parameters").or_else(|| t.get("rope_scaling")).cloned().unwrap_or(Value::Null);
        let f = |k: &str| p.get(k).and_then(Value::as_f64).or_else(|| t.get(k).and_then(Value::as_f64));
        let theta = f("rope_theta").unwrap_or(10000.0);
        let partial = f("partial_rotary_factor").unwrap_or(1.0);
        let mrope_section = p
            .get("mrope_section")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
            .unwrap_or_default();
        let scaling = match p.get("rope_type").or_else(|| p.get("type")).and_then(Value::as_str) {
            Some("yarn") => RopeScaling::Yarn {
                factor: f("factor").ok_or_else(|| anyhow!("yarn rope without factor"))?,
                original_max: p
                    .get("original_max_position_embeddings")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| anyhow!("yarn rope without original_max_position_embeddings"))? as usize,
                beta_fast: f("beta_fast").unwrap_or(32.0),
                beta_slow: f("beta_slow").unwrap_or(1.0),
            },
            None | Some("default") => RopeScaling::None,
            Some(other) => bail!("rope type {other} is not supported yet"),
        };
        Ok(RopeConfig {
            theta,
            rotary_dim: ((head_dim as f64) * partial).round() as usize,
            mrope_section,
            mrope_interleaved: p.get("mrope_interleaved").and_then(Value::as_bool).unwrap_or(false),
            scaling,
        })
    }

    pub fn full_layers(&self) -> usize {
        self.layers.iter().filter(|k| **k == LayerKind::Full).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inkling_loop_visits_keep_weight_ids_and_separate_cache_slots() {
        let mut text = serde_json::json!({
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64},
            "sliding_window_size": 1024,
            "local_layer_ids": (0..24).filter(|i| i % 6 != 5).collect::<Vec<_>>(),
        });
        let layout = InklingLoop::parse(&text, 24).unwrap();
        assert_eq!(layout.visits.len(), 48);
        assert_eq!(layout.global_visits(), 8);
        for (i, pass) in [6, 18, 30].into_iter().zip(0..3) {
            assert_eq!(layout.visits[i], LayerVisit { layer: 6, pass });
        }
        assert_eq!(layout.visits[42], LayerVisit { layer: 18, pass: 0 });
        text["local_layer_ids"] = serde_json::json!([0, 0]);
        assert!(InklingLoop::parse(&text, 24).is_err());
    }
}
