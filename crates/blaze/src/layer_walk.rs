use anyhow::{Context, Result, ensure};
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerVisit {
    pub layer: usize,
    pub pass: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoopEntry {
    Save,
    Rewind {
        first_message: usize,
        previous_blocks: usize,
    },
}

#[derive(Clone, Copy)]
pub(crate) struct ResidualVisit {
    pub depth: usize,
    pub bank_slot: Option<usize>,
    pub loop_entry: Option<LoopEntry>,
    pub messages: usize,
}

/// Plan the residual bank independently of where the tensors live.
pub(crate) fn residual_visits(
    visits: &[LayerVisit],
    core_start: usize,
    layers_per_block: usize,
) -> Result<Vec<ResidualVisit>> {
    ensure!(layers_per_block > 0, "residual blocks must contain layers");
    let mut depth = 0usize;
    let mut entry_depth = None;
    let mut messages = 0;
    visits
        .iter()
        .map(|visit| {
            let loop_entry = if visit.layer != core_start {
                None
            } else if visit.pass == 0 {
                entry_depth = Some(depth);
                Some(LoopEntry::Save)
            } else {
                let entry = entry_depth.context("loop pass has no entry state")?;
                messages = depth
                    .checked_sub(entry)
                    .filter(|&n| n > 0)
                    .context("loop pass has no residual blocks")?;
                depth = entry;
                Some(LoopEntry::Rewind {
                    first_message: entry + 1,
                    previous_blocks: messages - 1,
                })
            };
            let bank_slot = visit
                .layer
                .is_multiple_of(layers_per_block)
                .then_some(depth);
            let state = ResidualVisit {
                depth,
                bank_slot,
                loop_entry,
                messages,
            };
            depth += usize::from(bank_slot.is_some());
            Ok(state)
        })
        .collect()
}

pub(crate) fn repeated_layers(
    layers: usize,
    core: Range<usize>,
    passes: usize,
) -> Result<Vec<LayerVisit>> {
    ensure!(
        core.start < core.end && core.end <= layers,
        "invalid repeated layer range {core:?}"
    );
    ensure!(passes > 0, "layer walk requires at least one pass");
    let count = core
        .len()
        .checked_mul(passes)
        .and_then(|n| n.checked_add(layers - core.len()))
        .context("layer visit count overflows")?;
    let mut visits = Vec::new();
    visits
        .try_reserve_exact(count)
        .context("allocating layer visits")?;
    let visit = |layer| LayerVisit { layer, pass: 0 };
    visits.extend((0..core.start).map(visit));
    for pass in 0..passes {
        visits.extend(core.clone().map(|layer| LayerVisit { layer, pass }));
    }
    visits.extend((core.end..layers).map(visit));
    Ok(visits)
}

pub(crate) fn pipeline_ranges(
    layers: usize, stages: usize, co_located: Option<Range<usize>>,
) -> Result<Vec<Vec<Range<usize>>>> {
    ensure!(stages > 0 && layers >= stages, "{layers} layers for {stages} pipeline stages");
    if let Some(group) = co_located.filter(|_| stages > 1) {
        ensure!(group.start < group.end && group.end <= layers, "invalid co-located range {group:?}");
        ensure!(stages == 2, "co-located layer groups use two pipeline stages");
        let outside = [0..group.start, group.end..layers].into_iter().filter(|range| !range.is_empty()).collect();
        return Ok(vec![vec![group], outside]);
    }
    Ok((0..stages).map(|stage| vec![stage * layers / stages..(stage + 1) * layers / stages]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_keeps_repeated_layers_together() -> Result<()> {
        assert_eq!(pipeline_ranges(10, 3, None)?, [vec![0..3], vec![3..6], vec![6..10]]);
        assert_eq!(pipeline_ranges(24, 2, Some(6..18))?, [vec![6..18], vec![0..6, 18..24]]);
        Ok(())
    }

    #[test]
    fn loop_rewinds_to_entry_and_addresses_only_previous_pass_messages() -> Result<()> {
        let visits = repeated_layers(8, 2..6, 3)?;
        let plan = residual_visits(&visits, 2, 2)?;
        assert_eq!(plan[2].loop_entry, Some(LoopEntry::Save));
        for i in [6, 10] {
            assert_eq!(
                plan[i].loop_entry,
                Some(LoopEntry::Rewind {
                    first_message: 2,
                    previous_blocks: 1
                })
            );
            assert_eq!(
                (plan[i].depth, plan[i].messages, plan[i].bank_slot),
                (1, 2, Some(1))
            );
        }
        assert_eq!(plan.last().unwrap().depth, 4);
        Ok(())
    }
}
