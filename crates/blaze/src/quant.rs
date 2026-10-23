//! Per-module storage formats and activation calibration read from checkpoint tensors and configuration.

use crate::weights::{Checkpoint, Dtype};
use anyhow::{Result, anyhow, bail};
use regex::Regex;
use serde_json::Value;

/// How a quantized layer expects its input activations (what the checkpoint was calibrated for).
#[derive(Debug, Clone, PartialEq)]
pub enum ActQuant {
    /// 16-bit activations (weight-only quantization).
    None,
    /// FP8 E4M3 per token group, scales computed on the fly.
    Fp8Dynamic { group: Option<usize> },
    /// FP8 E4M3 with a calibrated per-tensor scale (`input_scale`).
    Fp8Static,
    /// NVFP4 activations with a calibrated global scale (`input_scale` / `input_global_scale`).
    Nvfp4,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Format {
    /// Unquantized weight in its stored 16-bit dtype.
    Dense(Dtype),
    /// FP8 E4M3 weight with one scale per `[block_n, block_k]` block (`weight_scale` or `weight_scale_inv`).
    Fp8Block { block_n: usize, block_k: usize, scale_dtype: Dtype, inverse: bool },
    /// FP8 E4M3 weight with one scale per output channel or per tensor.
    Fp8Channel { per_tensor: bool },
    /// MXFP8 (OCP microscaling, ModelOpt MXFP8): E4M3 codes with an E8M0 scale (uint8) per 32 values of a row.
    Mxfp8,
    /// NVFP4: E2M1 codes two per byte, an E4M3 scale per `group` values, an FP32 global scale.
    /// `global_inverse`: compressed-tensors stores the reciprocal (`weight_global_scale`), ModelOpt the factor
    /// itself (`weight_scale_2`).
    Nvfp4 { group: usize, global_inverse: bool },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scheme {
    pub format: Format,
    pub act: ActQuant,
}

/// Names of a module's tensors in the checkpoint.
#[derive(Debug, Clone)]
pub struct WeightNames {
    pub weight: String,
    pub scale: Option<String>,
    pub global: Option<String>,
    pub input_scale: Option<String>,
}

pub struct QuantConfig {
    method: String,
    rules: Vec<(Vec<Matcher>, ActQuant)>,
    ignore: Vec<Matcher>,
    default_act: ActQuant,
}

enum Matcher {
    Regex(Regex),
    Glob(Regex),
    Exact(String),
    /// A module class name ("Linear"): matches every linear layer.
    AnyLinear,
}

impl Matcher {
    fn parse(s: &str) -> Result<Self> {
        Ok(if let Some(re) = s.strip_prefix("re:") {
            Self::Regex(Regex::new(re)?)
        } else if s.contains('*') {
            let pattern = format!("^{}$", regex::escape(s).replace(r"\*", ".*"));
            Self::Glob(Regex::new(&pattern)?)
        } else if s == "Linear" {
            Self::AnyLinear
        } else {
            Self::Exact(s.to_string())
        })
    }

    fn matches(&self, module: &str) -> bool {
        match self {
            Self::Regex(r) => r.is_match(module),
            Self::Glob(r) => r.is_match(module),
            Self::Exact(s) => module == s || module.ends_with(&format!(".{s}")),
            Self::AnyLinear => true,
        }
    }
}

fn act_of(group: &Value) -> ActQuant {
    let Some(a) = group.get("input_activations").filter(|a| !a.is_null()) else {
        return ActQuant::None;
    };
    let bits = a.get("num_bits").and_then(Value::as_u64).unwrap_or(16);
    let dynamic = a.get("dynamic").and_then(Value::as_bool).unwrap_or(false);
    match (bits, a.get("type").and_then(Value::as_str)) {
        (4, _) => ActQuant::Nvfp4,
        (8, Some("float")) if dynamic => ActQuant::Fp8Dynamic { group: a.get("group_size").and_then(Value::as_u64).map(|g| g as usize) },
        (8, Some("float")) => ActQuant::Fp8Static,
        _ => ActQuant::None,
    }
}

impl QuantConfig {
    pub fn from_model_config(root: &Value, hf_quant: Option<&Value>) -> Result<Self> {
        let q = root.get("quantization_config").cloned().or_else(|| hf_quant.and_then(|h| h.get("quantization").cloned()));
        let Some(q) = q else {
            return Ok(Self { method: "none".into(), rules: vec![], ignore: vec![], default_act: ActQuant::None });
        };
        let method = q
            .get("quant_method")
            .and_then(Value::as_str)
            .unwrap_or(if hf_quant.is_some() { "modelopt" } else { "unknown" })
            .to_string();
        let mut rules = Vec::new();
        let mut default_act = ActQuant::None;
        if let Some(groups) = q.get("config_groups").and_then(Value::as_object) {
            for g in groups.values() {
                let targets = g
                    .get("targets")
                    .and_then(Value::as_array)
                    .map(|t| t.iter().filter_map(Value::as_str).map(Matcher::parse).collect::<Result<Vec<_>>>())
                    .transpose()?
                    .unwrap_or_default();
                rules.push((targets, act_of(g)));
            }
        } else {
            // ModelOpt: one algorithm for every quantized layer.
            let algo = q.get("quant_algo").and_then(Value::as_str).unwrap_or("");
            default_act = match algo {
                "NVFP4" | "NVFP4_AWQ" => ActQuant::Nvfp4,
                "FP8" => ActQuant::Fp8Static,
                "FP8_PB_WO" | "W4A16_NVFP4" | "W4A16_AWQ" | "MXFP8" | "" => ActQuant::None,
                other => bail!("ModelOpt quant_algo {other} is not supported yet"),
            };
        }
        let ignore = q
            .get("ignore")
            .or_else(|| q.get("exclude_modules"))
            .and_then(Value::as_array)
            .map(|t| t.iter().filter_map(Value::as_str).map(Matcher::parse).collect::<Result<Vec<_>>>())
            .transpose()?
            .unwrap_or_default();
        Ok(Self { method, rules, ignore, default_act })
    }

    pub fn method(&self) -> &str {
        &self.method
    }

    fn act_for(&self, module: &str) -> ActQuant {
        if self.ignore.iter().any(|m| m.matches(module)) {
            return ActQuant::None;
        }
        for (targets, act) in &self.rules {
            if targets.iter().any(|m| m.matches(module)) {
                return act.clone();
            }
        }
        self.default_act.clone()
    }

    /// The scheme and tensor names of the linear module `module` (its name without `.weight`).
    pub fn resolve(&self, ckpt: &Checkpoint, module: &str) -> Result<(Scheme, WeightNames)> {
        let has = |suffix: &str| ckpt.get(&format!("{module}.{suffix}")).map(|t| t.dtype);
        let name = |suffix: &str| format!("{module}.{suffix}");
        let input_scale = ["input_scale", "input_global_scale"].iter().find(|s| has(s).is_some()).map(|s| name(s));
        let (format, names) = if has("weight_packed") == Some(Dtype::U8) && has("weight_global_scale").is_some() {
            // compressed-tensors nvfp4-pack-quantized
            let w = ckpt.info(&name("weight_packed"))?;
            let s = ckpt.info(&name("weight_scale"))?;
            let group = (w.shape[1] * 2) / s.shape[1];
            (
                Format::Nvfp4 { group, global_inverse: true },
                WeightNames { weight: name("weight_packed"), scale: Some(name("weight_scale")), global: Some(name("weight_global_scale")), input_scale },
            )
        } else if has("weight") == Some(Dtype::U8) && has("weight_scale_2").is_some() {
            // ModelOpt NVFP4
            let w = ckpt.info(&name("weight"))?;
            let s = ckpt.info(&name("weight_scale"))?;
            let group = (w.shape[1] * 2) / s.shape[1];
            (
                Format::Nvfp4 { group, global_inverse: false },
                WeightNames { weight: name("weight"), scale: Some(name("weight_scale")), global: Some(name("weight_scale_2")), input_scale },
            )
        } else if has("weight") == Some(Dtype::F8E4m3) && has("weight_scale") == Some(Dtype::U8) {
            // ModelOpt MXFP8: E8M0 scales [n, k / 32]
            let w = ckpt.info(&name("weight"))?;
            let s = ckpt.info(&name("weight_scale"))?;
            if s.shape.len() != 2 || s.shape[0] != w.shape[0] || s.shape[1] * 32 != w.shape[1] {
                bail!("{module}: MXFP8 scales {:?} for a {:?} weight", s.shape, w.shape);
            }
            (Format::Mxfp8, WeightNames { weight: name("weight"), scale: Some(name("weight_scale")), global: None, input_scale })
        } else if has("weight") == Some(Dtype::F8E4m3) {
            let w = ckpt.info(&name("weight"))?;
            let (scale_name, inverse) = if has("weight_scale_inv").is_some() { (name("weight_scale_inv"), true) } else { (name("weight_scale"), false) };
            let s = ckpt.info(&scale_name)?;
            let format = if s.shape.len() == 2 && s.shape[1] > 1 {
                Format::Fp8Block {
                    block_n: w.shape[0].div_ceil(s.shape[0]),
                    block_k: w.shape[1].div_ceil(s.shape[1]),
                    scale_dtype: s.dtype,
                    inverse,
                }
            } else {
                Format::Fp8Channel { per_tensor: s.numel() == 1 }
            };
            (format, WeightNames { weight: name("weight"), scale: Some(scale_name), global: None, input_scale })
        } else if let Some(d @ (Dtype::Bf16 | Dtype::F16 | Dtype::F32)) = has("weight") {
            (Format::Dense(d), WeightNames { weight: name("weight"), scale: None, global: None, input_scale: None })
        } else {
            return Err(anyhow!("{module}: no weight in a known format"));
        };
        let act = if matches!(format, Format::Dense(_)) { ActQuant::None } else { self.act_for(module) };
        Ok((Scheme { format, act }, names))
    }
}

impl Format {
    pub fn describe(&self) -> String {
        match self {
            Format::Dense(d) => format!("{d:?}").to_lowercase(),
            Format::Fp8Block { block_n, block_k, .. } => format!("fp8 block {block_n}x{block_k}"),
            Format::Fp8Channel { per_tensor: true } => "fp8 per-tensor".into(),
            Format::Fp8Channel { per_tensor: false } => "fp8 per-channel".into(),
            Format::Mxfp8 => "mxfp8".into(),
            Format::Nvfp4 { group, .. } => format!("nvfp4 g{group}"),
        }
    }
}
