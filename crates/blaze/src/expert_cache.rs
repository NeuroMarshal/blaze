use anyhow::{Result, ensure};
use std::collections::HashMap;

pub(crate) type ExpertId = (usize, usize);

#[derive(Clone, Copy, Default)]
struct Heat {
    count: usize,
    epoch: usize,
}

impl Heat {
    fn at(self, epoch: usize) -> usize {
        self.count >> epoch.saturating_sub(self.epoch).min(usize::BITS as usize - 1)
    }

    fn observe(&mut self, epoch: usize, count: usize) {
        *self = Self { count: self.at(epoch).saturating_add(count), epoch };
    }
}

struct Resident<Record> {
    value: Record,
    slot: usize,
    last_used: usize,
    heat: Heat,
}

pub(crate) struct ExpertVisit {
    pub tick: usize,
    pub epoch: usize,
    pub observations: Option<usize>,
    pub preserve_group: bool,
}

pub(crate) struct ExpertResidency<Record> {
    free: Vec<usize>,
    records: HashMap<ExpertId, Resident<Record>>,
    hits: HashMap<ExpertId, Heat>,
}

impl<Record> ExpertResidency<Record> {
    pub fn new(slots: usize) -> Self {
        Self {
            free: (0..slots).rev().collect(),
            records: HashMap::new(),
            hits: HashMap::new(),
        }
    }

    pub fn get(&self, key: ExpertId) -> Option<&Record> {
        self.records.get(&key).map(|entry| &entry.value)
    }

    pub fn touch(&mut self, key: ExpertId, tick: usize) {
        if let Some(entry) = self.records.get_mut(&key) {
            entry.last_used = tick;
        }
    }

    pub fn cold_scores(&self, layer: usize, epoch: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.hits.iter().filter_map(move |(&key, heat)| {
            (key.0 == layer && !self.records.contains_key(&key)).then_some((key.1, heat.at(epoch)))
        })
    }

    pub fn admit(
        &mut self, key: ExpertId, visit: ExpertVisit, before_evict: impl FnOnce() -> Result<()>,
        load: impl FnOnce(usize) -> Result<Record>,
    ) -> Result<bool> {
        let ExpertVisit { tick, epoch, observations, preserve_group } = visit;
        let heat = if let Some(count) = observations {
            let score = self.hits.entry(key).or_default();
            score.observe(epoch, count);
            *score
        } else { Heat::default() };
        if let Some(entry) = self.records.get_mut(&key) {
            entry.last_used = tick;
            entry.heat = heat;
            return Ok(true);
        }
        if observations.is_some() && heat.count < 2 {
            return Ok(false);
        }
        let slot = if let Some(slot) = self.free.pop() {
            slot
        } else {
            let victim = self.records.iter().filter(|(_, entry)| !preserve_group || entry.last_used != tick)
                .min_by_key(|(key, entry)| (entry.heat.at(epoch), entry.last_used, **key));
            let Some((&old, entry)) = victim else {
                ensure!(observations.is_some(), "GPU expert slots exhausted");
                return Ok(false);
            };
            let old_count = entry.heat.at(epoch);
            if observations.is_some() && heat.count < old_count.saturating_add(16.max(old_count / 4)) {
                return Ok(false);
            }
            before_evict()?;
            self.records.remove(&old).unwrap().slot
        };
        match load(slot) {
            Ok(value) => {
                self.records.insert(key, Resident { value, slot, last_used: tick, heat });
                Ok(true)
            }
            Err(error) => {
                self.free.push(slot);
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_group_keeps_its_slots_and_failed_upload_releases_slot() {
        let mut cache = ExpertResidency::new(1);
        let visit = |tick, observations| ExpertVisit { tick, epoch: 0, observations, preserve_group: true };
        let old = (1, 2);
        let new = (1, 3);
        assert!(cache.admit(old, visit(1, Some(20)), || Ok(()), Ok).unwrap());
        assert!(!cache.admit(new, visit(1, Some(80)), || Ok(()), Ok).unwrap());
        assert_eq!(cache.get(old), Some(&0));
        assert!(cache.admit(new, visit(2, Some(20)), || Ok(()), |_| anyhow::bail!("test upload")).is_err());
        assert_eq!(cache.get(old), None);
        assert!(cache.admit(new, visit(3, None), || Ok(()), Ok).unwrap());
        assert_eq!(cache.get(new), Some(&0));
    }
}
