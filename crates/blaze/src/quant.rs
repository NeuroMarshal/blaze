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
    /// MXFP6 codes packed low-bit first, with an E8M0 scale per 32 values.
    Mxfp6(Mxfp6),
    /// NVFP4: E2M1 codes two per byte, an E4M3 scale per `group` values, an FP32 global scale.
    /// `global_inverse`: compressed-tensors stores the reciprocal (`weight_global_scale`), ModelOpt the factor
    /// itself (`weight_scale_2`).
    Nvfp4 { group: usize, global_inverse: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mxfp6 {
    E2M3,
    E3M2,
}

impl Mxfp6 {
    /// All finite FP6 element codes are exactly representable in E4M3.
    pub(crate) fn widen(self, packed: &[u8]) -> Vec<u8> {
        let (bits, bias) = match self {
            Self::E2M3 => (3, 1),
            Self::E3M2 => (2, 3),
        };
        let convert = |code: u8| {
            let sign = (code & 32) << 2;
            let exponent = i32::from((code & 31) >> bits);
            let mantissa = code & ((1 << bits) - 1);
            if exponent == 0 && mantissa == 0 {
                return sign;
            }
            let (exponent, fraction) = if exponent == 0 {
                let top = mantissa.ilog2();
                (1 - bias + top as i32 - bits, (mantissa << (3 - top)) & 7)
            } else {
                (exponent - bias, mantissa << (3 - bits))
            };
            sign | ((exponent + 7) as u8) << 3 | fraction
        };
        packed.chunks_exact(3).flat_map(|word| {
            let bits = u32::from_le_bytes([word[0], word[1], word[2], 0]);
            [0, 6, 12, 18].map(|shift| convert(((bits >> shift) & 63) as u8))
        }).collect()
    }
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
    /// FP8 block checkpoints: `weight_block_size` [rows, columns] of one scale
    block: Option<(usize, usize)>,
    rules: Vec<(Vec<Matcher>, ActQuant)>,
    ignore: Vec<Matcher>,
    default_act: ActQuant,
    mxfp6: Option<Mxfp6>,
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
            return Ok(Self { method: "none".into(), block: None, rules: vec![], ignore: vec![], default_act: ActQuant::None, mxfp6: None });
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
                "FP8_PB_WO" | "W4A16_NVFP4" | "W4A16_AWQ" | "MXFP8" | "MXFP6_E2M3" | "MXFP6_E3M2" | "" => ActQuant::None,
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
        let block = q.get("weight_block_size").and_then(Value::as_array).and_then(|b| match b.as_slice() {
            [n, k] => Some((n.as_u64()? as usize, k.as_u64()? as usize)),
            _ => None,
        });
        let mxfp6 = match q.get("quant_algo").and_then(Value::as_str) {
            Some("MXFP6_E2M3") => Some(Mxfp6::E2M3),
            Some("MXFP6_E3M2") => Some(Mxfp6::E3M2),
            _ => None,
        };
        Ok(Self { method, block, rules, ignore, default_act, mxfp6 })
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
        } else if let Some(encoding) = self.mxfp6
            && has("weight") == Some(Dtype::U8) && has("weight_scale") == Some(Dtype::U8) {
            let w = ckpt.info(&name("weight"))?;
            let s = ckpt.info(&name("weight_scale"))?;
            if w.shape.len() != 2 || s.shape.len() != 2 || w.shape[0] != s.shape[0] || s.shape[1].checked_mul(24) != Some(w.shape[1]) {
                bail!("{module}: MXFP6 expects 24 packed bytes for every 32-value scale");
            }
            (Format::Mxfp6(encoding), WeightNames { weight: name("weight"), scale: Some(name("weight_scale")), global: None, input_scale })
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
                // the block from the config, else the power of two the scale grid implies (a 200-row weight with two
                // scale rows has blocks of 128, not 100); either must give the stored grid
                let (n, k) = (w.shape[0], w.shape[1]);
                let (bn, bk) = self.block.unwrap_or((n.div_ceil(s.shape[0]).next_power_of_two(), k.div_ceil(s.shape[1]).next_power_of_two()));
                if bn == 0 || bk == 0 || n.div_ceil(bn) != s.shape[0] || k.div_ceil(bk) != s.shape[1] {
                    bail!("{module}: FP8 scales {:?} do not match {bn}x{bk} blocks of a {:?} weight", s.shape, w.shape);
                }
                Format::Fp8Block { block_n: bn, block_k: bk, scale_dtype: s.dtype, inverse }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fp6_element_widens_to_e4m3_exactly() {
        let packed = (0..64u32).step_by(4).flat_map(|at| {
            let word = at | ((at + 1) << 6) | ((at + 2) << 12) | ((at + 3) << 18);
            let bytes = word.to_le_bytes();
            [bytes[0], bytes[1], bytes[2]]
        }).collect::<Vec<_>>();
        for (encoding, mantissa_bits, bias) in [(Mxfp6::E2M3, 3, 1), (Mxfp6::E3M2, 2, 3)] {
            for (code, value) in encoding.widen(&packed).into_iter().enumerate() {
                let exponent = ((code & 31) >> mantissa_bits) as i32;
                let fraction = (code & ((1 << mantissa_bits) - 1)) as f32 / (1 << mantissa_bits) as f32;
                let magnitude = if exponent == 0 { fraction * 2f32.powi(1 - bias) }
                    else { (1.0 + fraction) * 2f32.powi(exponent - bias) };
                let expected = if code & 32 == 0 { magnitude } else { -magnitude };
                let expanded = if value & 127 == 0 { 0.0 } else {
                    (1.0 + f32::from(value & 7) / 8.0) * 2f32.powi(i32::from((value & 127) >> 3) - 7)
                };
                let actual = if value & 128 == 0 { expanded } else { -expanded };
                assert_eq!(actual.to_bits(), expected.to_bits(), "{encoding:?}, code {code}");
            }
        }
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
            Format::Mxfp6(encoding) => format!("mxfp6 {encoding:?}"),
            Format::Nvfp4 { group, .. } => format!("nvfp4 g{group}"),
        }
    }
}
