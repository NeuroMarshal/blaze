use anyhow::{Context, Result, ensure};
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerVisit {
    pub layer: usize,
    pub pass: usize,
}

pub(crate) struct LoopResiduals<T> {
    entry: Option<(T, usize)>,
    messages: Vec<T>,
}

impl<T: Clone> LoopResiduals<T> {
    pub fn new() -> Self {
        Self { entry: None, messages: Vec::new() }
    }

    pub fn enter(&mut self, pass: usize, hidden: &mut T, blocks: &mut Vec<T>, finish: impl FnOnce(&T) -> T) {
        if pass == 0 {
            self.entry = Some((hidden.clone(), blocks.len()));
            return;
        }
        let (entry, depth) = self.entry.as_ref().expect("repeated layer visited before its first pass");
        self.messages.clear();
        self.messages.extend_from_slice(&blocks[*depth + 1..]);
        self.messages.push(finish(hidden));
        blocks.truncate(*depth);
        hidden.clone_from(entry);
    }

    pub fn messages(&self) -> &[T] {
        &self.messages
    }

    pub fn leave(&mut self) {
        self.entry = None;
        self.messages.clear();
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_restores_entry_and_replaces_previous_pass_messages() {
        let mut residuals = LoopResiduals::new();
        let mut blocks = vec![1];
        let mut hidden = 2;
        residuals.enter(0, &mut hidden, &mut blocks, |_| unreachable!());
        for pass in 1..3 {
            blocks.extend([2, 2 + pass]);
            hidden += pass;
            residuals.enter(pass, &mut hidden, &mut blocks, |&x| x * 2);
            assert_eq!(hidden, 2);
            assert_eq!(&blocks, &[1]);
            assert_eq!(residuals.messages(), &[2 + pass, (2 + pass) * 2]);
        }
        residuals.leave();
        assert!(residuals.messages().is_empty());
    }
}
