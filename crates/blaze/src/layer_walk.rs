use anyhow::{Context, Result, ensure};
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerVisit {
    pub layer: usize,
    pub pass: usize,
}

pub(crate) fn repeated_layers(layers: usize, core: Range<usize>, passes: usize) -> Result<Vec<LayerVisit>> {
    ensure!(core.start < core.end && core.end <= layers, "invalid repeated layer range {core:?}");
    ensure!(passes > 0, "layer walk requires at least one pass");
    let count = core.len().checked_mul(passes).and_then(|n| n.checked_add(layers - core.len()))
        .context("layer visit count overflows")?;
    let mut visits = Vec::new();
    visits.try_reserve_exact(count).context("allocating layer visits")?;
    let visit = |layer| LayerVisit { layer, pass: 0 };
    visits.extend((0..core.start).map(visit));
    for pass in 0..passes {
        visits.extend(core.clone().map(|layer| LayerVisit { layer, pass }));
    }
    visits.extend((core.end..layers).map(visit));
    Ok(visits)
}
