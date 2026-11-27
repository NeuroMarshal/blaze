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

pub(crate) fn match_picks<T: Eq + Copy>(draft: &[T], picks: &[T]) -> (usize, T) {
    debug_assert_eq!(picks.len(), draft.len() + 1);
    let kept = draft
        .iter()
        .zip(picks)
        .take_while(|(proposal, pick)| proposal == pick)
        .count();
    (kept, picks[kept])
}

pub(crate) fn match_chain<T: Eq + Copy, E>(
    draft: &[T],
    mut pending: T,
    mut predict: impl FnMut(T) -> Result<T, E>,
) -> Result<(usize, T), E> {
    for kept in 0..=draft.len() {
        pending = predict(pending)?;
        if draft.get(kept) != Some(&pending) {
            return Ok((kept, pending));
        }
    }
    unreachable!("the bonus token always ends verification")
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
            assert_eq!(match_picks(draft, picks), expected);
            let mut next = picks.iter().copied();
            assert_eq!(
                match_chain(draft, 1, |_| Ok::<_, ()>(next.next().unwrap())),
                Ok(expected)
            );
        }
    }
}
