/// Rows of the last target step to extend the draft over, for one sequence.
#[derive(Clone, Debug)]
pub struct DraftItem {
    /// sequence row (the draft's KV ring)
    pub row: usize,
    /// first row of the item in the last target step (its captured hidden states)
    pub aux_row: usize,
    /// the token after each extended row (the target's input at the next position)
    pub tokens: Vec<i64>,
    /// position of the first extended row
    pub pos0: usize,
}

/// One draft call: extend every item, then draft `k` tokens after each (0: extend only), or with `tree` = (K, D) a
/// tree of depth D with K frontier rows per item instead of the chain.
#[derive(Clone, Debug, Default)]
pub struct DraftPlan {
    pub items: Vec<DraftItem>,
    pub k: usize,
    pub tree: Option<(usize, usize)>,
    /// MTP head: per item the token ids its draft head may propose (ascending; empty: the whole vocabulary). Parallel to
    /// `items` or empty.
    pub cands: Vec<Vec<i32>>,
}

/// Prompt-lookup draft: the longest suffix of `ctx` of max_n .. min_n tokens that occurred earlier (most recent
/// occurrence, within the last 64K tokens), and up to k tokens that followed it.
pub fn propose(ctx: &[i64], max_n: usize, min_n: usize, k: usize) -> Vec<i64> {
    let len = ctx.len();
    let lo = len.saturating_sub(1 << 16);
    for n in (min_n..=max_n).rev() {
        if len <= n {
            continue;
        }
        let tail = &ctx[len - n..];
        // most recent earlier occurrence whose continuation exists
        let mut i = len - n;
        while i > lo {
            i -= 1;
            if ctx[i..i + n] == *tail {
                let start = i + n;
                let end = (start + k).min(len);
                if start < end {
                    return ctx[start..end].to_vec();
                }
            }
        }
    }
    Vec::new()
}

pub(crate) fn match_picks<T: Eq + Copy>(
    draft: &[T],
    picks: impl IntoIterator<Item = T>,
) -> (usize, T) {
    let mut picks = picks.into_iter();
    let mut last = None;
    let emitted = visit_verified_chain(draft, || {
        let pick = picks.next().expect("one target prediction per row");
        last = Some(pick);
        Some((pick, false))
    });
    (emitted - 1, last.expect("verification always reads the anchor prediction"))
}

pub(crate) fn visit_verified_chain<T: Eq>(draft: &[T], mut read: impl FnMut() -> Option<(T, bool)>) -> usize {
    let mut emitted = 0;
    while let Some((token, finished)) = read() {
        let matched = draft.get(emitted) == Some(&token);
        emitted += 1;
        if finished || !matched {
            break;
        }
    }
    emitted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_prefix_includes_a_target_prediction_on_match_and_rejection() {
        for (draft, picks, expected) in [
            (&[2, 3][..], &[2, 3, 4][..], (2, 4)),
            (&[2, 3][..], &[2, 4, 5][..], (1, 4)),
            (&[2, 3][..], &[9, 4, 5][..], (0, 9)),
        ] {
            let mut sampled = 0;
            let picks = picks.iter().copied().inspect(|_| sampled += 1);
            assert_eq!(match_picks(draft, picks), expected);
            assert_eq!(sampled, expected.0 + 1);
        }
    }

    #[test]
    fn finished_request_does_not_read_the_rest_of_the_verified_chain() {
        let mut reads = 0;
        let emitted = visit_verified_chain(&[2, 3], || {
            reads += 1;
            Some((2, true))
        });
        assert_eq!((emitted, reads), (1, 1));
    }
}
