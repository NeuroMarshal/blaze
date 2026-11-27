use std::cmp::Ordering;

pub(crate) fn rank_experts<T>(candidates: &mut [T], limit: usize, compare: impl Fn(&T, &T) -> Ordering) -> &mut [T] {
    let count = limit.min(candidates.len());
    if (1..candidates.len()).contains(&count) {
        candidates.select_nth_unstable_by(count, &compare);
    }
    let selected = &mut candidates[..count];
    selected.sort_unstable_by(compare);
    selected
}

pub(crate) unsafe extern "C" fn rank_expert_ids(
    ids: *mut usize, len: usize, scores: *const f64, experts: usize, limit: usize, hottest: std::ffi::c_int,
) -> usize {
    if len == 0 || experts == 0 {
        return 0;
    }
    // HybridMoe passes distinct candidate and score arrays.
    let ids = unsafe { std::slice::from_raw_parts_mut(ids, len) };
    let scores = unsafe { std::slice::from_raw_parts(scores, experts) };
    if ids.iter().any(|&id| id >= scores.len()) {
        return 0;
    }
    rank_experts(ids, limit, |&a, &b| {
        let order = scores[a].total_cmp(&scores[b]);
        let order = if hottest != 0 { order.reverse() } else { order };
        order.then(a.cmp(&b))
    }).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_swap_candidates_by_heat_with_stable_ties() {
        let scores = [13.0, 2.0, 13.0, 6.0, 1.0];
        let mut ids = [2, 3, 1, 0, 4];
        for (direction, expected) in [(1, [0, 2]), (0, [4, 1])] {
            let count = unsafe { rank_expert_ids(ids.as_mut_ptr(), ids.len(), scores.as_ptr(), scores.len(), 2, direction) };
            assert_eq!(count, 2);
            assert_eq!(&ids[..count], &expected);
        }
    }

}
