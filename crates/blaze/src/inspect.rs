//! `blaze inspect`: what a checkpoint contains and how blaze reads it (architecture, layers, formats per module).

use crate::config::{LayerKind, MlpConfig, ModelConfig};
use crate::quant::{ActQuant, QuantConfig};
use crate::weights::Checkpoint;
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;

/// Linear modules of the checkpoint: tensors named `<module>.weight` / `<module>.weight_packed` with two dims.
pub fn linear_modules(ckpt: &Checkpoint) -> Vec<String> {
    let mut out: Vec<String> = ckpt
        .tensors
        .iter()
        .filter(|(_, t)| t.shape.len() == 2)
        .filter_map(|(n, _)| n.strip_suffix(".weight_packed").or_else(|| n.strip_suffix(".weight")).map(str::to_string))
        .filter(|m| !m.contains("embed_tokens") && !m.contains("visual") && !m.ends_with("norm"))
        .collect();
    out.sort();
    out.dedup();
    out
}

pub fn run(dir: &Path) -> Result<()> {
    let cfg = ModelConfig::load(dir)?;
    let ckpt = Checkpoint::open(dir)?;
    let root: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let hf_quant = std::fs::read_to_string(dir.join("hf_quant_config.json")).ok().and_then(|s| serde_json::from_str(&s).ok());
    let quant = QuantConfig::from_model_config(&root, hf_quant.as_ref())?;

    let full = cfg.full_layers();
    println!("{} ({}), {} tensors in {} files", dir.display(), cfg.architecture, ckpt.tensors.len(), ckpt.files.len());
    println!(
        "hidden {}, {} layers ({} softmax attention, {} linear attention), vocab {}, max positions {}",
        cfg.hidden,
        cfg.layers.len(),
        full,
        cfg.layers.len() - full,
        cfg.vocab,
        cfg.max_positions
    );
    println!(
        "attention: {} q heads, {} kv heads, head dim {}, rotary dim {} (theta {}, {:?})",
        cfg.attn.heads, cfg.attn.kv_heads, cfg.attn.head_dim, cfg.rope.rotary_dim, cfg.rope.theta, cfg.rope.scaling
    );
    if let Some(g) = &cfg.gdn {
        println!("gated deltanet: {} k heads x {}, {} v heads x {}, conv {}", g.k_heads, g.k_dim, g.v_heads, g.v_dim, g.conv_kernel);
    }
    match &cfg.mlp {
        MlpConfig::Dense { intermediate } => println!("mlp: dense, intermediate {intermediate}"),
        MlpConfig::Moe { experts, top_k, intermediate, shared_intermediate, .. } => {
            println!("mlp: {experts} experts, top {top_k}, intermediate {intermediate}, shared expert {shared_intermediate}")
        }
    }
    println!("mtp layers {}, quantization: {}", cfg.mtp_layers, quant.method());

    let mut by_scheme: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut failures = Vec::new();
    for m in linear_modules(&ckpt) {
        match quant.resolve(&ckpt, &m) {
            Ok((scheme, names)) => {
                let act = match scheme.act {
                    ActQuant::None => "a16".to_string(),
                    ActQuant::Fp8Dynamic { group } => format!("a8 dynamic{}", group.map(|g| format!(" g{g}")).unwrap_or_default()),
                    ActQuant::Fp8Static => "a8 static".into(),
                    ActQuant::Nvfp4 => "a4".into(),
                };
                let bytes: usize = [Some(&names.weight), names.scale.as_ref(), names.global.as_ref()]
                    .into_iter()
                    .flatten()
                    .filter_map(|n| ckpt.get(n))
                    .map(|t| t.bytes())
                    .sum();
                let kind = if m.contains(".experts.") { "experts" } else if m.contains("mtp") { "mtp" } else { "other" };
                let e = by_scheme.entry(format!("{kind:8} {:<18} {act}", scheme.format.describe())).or_default();
                e.0 += 1;
                e.1 += bytes;
            }
            Err(e) => failures.push(e.to_string()),
        }
    }
    println!("\nlinear modules by format:");
    for (k, (n, bytes)) in &by_scheme {
        println!("  {k:<45} {n:>6} modules {:>9.2} GiB", *bytes as f64 / (1u64 << 30) as f64);
    }
    for f in failures.iter().take(10) {
        println!("  unresolved: {f}");
    }
    let layer_kinds: String = cfg.layers.iter().map(|k| if *k == LayerKind::Full { 'F' } else { 'l' }).collect();
    println!("\nlayers: {layer_kinds}");
    Ok(())
}
