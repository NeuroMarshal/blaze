//! The model's own `config.json`, read into what the graph builder needs. Unknown fields stay reachable through
//! `raw`; shapes that the config leaves implicit (e.g. the gated attention output) are taken from the weights.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
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
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let root: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Self::from_json(&root)
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
        let layers = match t.get("layer_types").and_then(Value::as_array) {
            Some(kinds) => kinds
                .iter()
                .map(|k| match k.as_str() {
                    Some("full_attention") => Ok(LayerKind::Full),
                    Some("linear_attention") => Ok(LayerKind::Linear),
                    other => Err(anyhow!("config: unknown layer type {other:?}")),
                })
                .collect::<Result<Vec<_>>>()?,
            None => match usize_of(&t, "full_attention_interval") {
                Some(every) => (0..n).map(|i| if (i + 1) % every == 0 { LayerKind::Full } else { LayerKind::Linear }).collect(),
                None => vec![LayerKind::Full; n],
            },
        };
        if layers.len() != n {
            bail!("config: {} layer types for {n} layers", layers.len());
        }
        let heads = need(&t, "num_attention_heads")?;
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
            })
        } else {
            None
        };
        let mlp = match usize_of(&t, "num_experts") {
            Some(experts) if experts > 0 => MlpConfig::Moe {
                experts,
                top_k: need(&t, "num_experts_per_tok")?,
                intermediate: need(&t, "moe_intermediate_size")?,
                shared_intermediate: usize_of(&t, "shared_expert_intermediate_size").unwrap_or(0),
                norm_topk: t.get("norm_topk_prob").and_then(Value::as_bool).unwrap_or(true),
            },
            _ => MlpConfig::Dense { intermediate: need(&t, "intermediate_size")? },
        };
        let rope = Self::rope(&t, attn.head_dim)?;
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
            max_positions: usize_of(&t, "max_position_embeddings").unwrap_or(32768),
            attn,
            gdn,
            mlp,
            rope,
            mtp_layers: usize_of(&t, "mtp_num_hidden_layers").unwrap_or(0),
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
