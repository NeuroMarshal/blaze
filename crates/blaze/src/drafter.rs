//! The drafters a target can run next to it, chosen by the checkpoint's architecture: EAGLE-4 (one layer, a chain of
//! draft steps) and DFlash2 (a block drafted in one pass).

use crate::dflash::{DFlash, DFlashConfig};
use crate::eagle::{Draft, DraftConfig, DraftPlan, DraftTree};
use crate::gpu::HostMapped;
use crate::model::Model;
use anyhow::{Result, bail};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

pub enum Drafter {
    Eagle(Draft),
    DFlash(DFlash),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Eagle,
    DFlash,
}

/// The drafter kind of a checkpoint.
pub fn kind(dir: &Path) -> Result<Kind> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    match v["architectures"].as_array().and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("") {
        "LlamaForCausalLMEagle4" => Ok(Kind::Eagle),
        "DFlash2DraftModel" => Ok(Kind::DFlash),
        other => bail!("{}: drafter architecture {other} (LlamaForCausalLMEagle4, DFlash2DraftModel supported)", dir.display()),
    }
}

/// VRAM one of `world` ranks gives a drafter, from its checkpoint: the tensors it keeps on the GPU (the shared ones whole,
/// the rest by its shard; not the embedding, which it shares with the target, nor an EAGLE-4 LM head, which stays in
/// pinned RAM), the EAGLE-4 head shard when it goes to VRAM (`head_vram` on or auto), and its KV ring and buffers.
pub fn vram_estimate(dir: &Path, world: usize, head_vram: Option<bool>) -> Result<usize> {
    let ckpt = crate::weights::Checkpoint::open(dir)?;
    let world = world.max(1);
    // modules every rank holds whole (Cut::All in eagle.rs / dflash.rs)
    let whole = ["fc.", "efficient_lm_head_down_proj", "kernel_projection", "candidate_selector", "norm"];
    let mut bytes = 0usize;
    for (name, t) in &ckpt.tensors {
        if name.contains("embed_tokens") || name == "lm_head.weight" {
            continue;
        }
        bytes += if whole.iter().any(|w| name.contains(w)) { t.bytes() } else { t.bytes().div_ceil(world) };
    }
    if matches!(kind(dir)?, Kind::Eagle) && head_vram != Some(false)
        && let Some(lm) = ckpt.get("lm_head.weight")
    {
        bytes += lm.bytes().div_ceil(world);
    }
    Ok(bytes + (256 << 20))
}

/// Target layers whose input the drafter reads.
pub fn capture_layers(dir: &Path) -> Result<Vec<usize>> {
    Ok(match kind(dir)? {
        Kind::Eagle => DraftConfig::load(dir)?.capture_layers(),
        Kind::DFlash => DFlashConfig::load(dir)?.capture_layers(),
    })
}

impl Drafter {
    /// `eagle_head`: the EAGLE-4 LM head table (shared by the ranks; None for DFlash, which uses the target's head).
    pub fn load(dir: &Path, model: &Model, max_items: usize, stride: usize, eagle_head: Option<Arc<HostMapped>>) -> Result<Self> {
        Ok(match kind(dir)? {
            Kind::Eagle => {
                let Some(head) = eagle_head else { bail!("EAGLE-4 drafter without its LM head table") };
                Self::Eagle(Draft::load(dir, model, max_items, stride, head)?)
            }
            Kind::DFlash => Self::DFlash(DFlash::load(dir, model, max_items, stride)?),
        })
    }

    pub fn run(&mut self, model: &Model, plan: &DraftPlan) -> Result<Vec<Vec<i64>>> {
        match self {
            Self::Eagle(d) => d.run(model, plan),
            Self::DFlash(d) => d.run(model, plan),
        }
    }

    /// Draft trees (EAGLE-4 only).
    pub fn run_tree(&mut self, model: &Model, plan: &DraftPlan) -> Result<Vec<DraftTree>> {
        match self {
            Self::Eagle(d) => d.run_tree(model, plan),
            Self::DFlash(_) => bail!("draft trees need an EAGLE-4 drafter"),
        }
    }

    /// Attention window: the prompt positions a drafter needs extended (the last `window` ones).
    pub fn window(&self) -> usize {
        match self {
            Self::Eagle(d) => d.cfg.window,
            Self::DFlash(d) => d.cfg.window,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Eagle(d) => d.describe(),
            Self::DFlash(d) => d.describe(),
        }
    }
}
