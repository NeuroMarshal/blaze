//! Draft trees ranked by path probability and measured acceptance.
//! Ancestor ordering keeps each selected prefix a valid verification tree.

use crate::eagle::DraftTree;

/// eagle_harness core/draft_tree.py DEPTH_DECAY
const DEPTH_DECAY: f64 = 0.9999;

/// The verify rows of a tree: row 0 the pending token, then the chosen nodes in node order (parents first).
#[derive(Clone, Debug, Default)]
pub struct VerifyRows {
    pub tokens: Vec<i64>,
    /// parent row of each row (-1 for row 0)
    pub parents: Vec<i32>,
    /// draft path probability of each row (1 for row 0)
    pub scores: Vec<f32>,
    /// the path's draft tokens before each row's own (row 0: none) are the ancestors' tokens
    pub depth: Vec<usize>,
}

impl VerifyRows {
    /// A lone pending token (no draft).
    pub fn single(pending: i64) -> Self {
        Self { tokens: vec![pending], parents: vec![-1], scores: vec![1.0], depth: vec![0] }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The draft tokens on the path to row r, row r's included (row 0's pending token not).
    pub fn path_tokens(&self, r: usize) -> Vec<i64> {
        let mut out = Vec::new();
        let mut x = r as i32;
        while x > 0 {
            out.push(self.tokens[x as usize]);
            x = self.parents[x as usize];
        }
        out.reverse();
        out
    }

    /// Walk from row 0: `pick(row)` gives the token emitted after a row (and whether the request stops there); the walk
    /// moves to the child row with that token while there is one. Returns the path's rows (row 0 first): their inputs
    /// are kept, one token was emitted after each.
    pub fn accept(&self, mut pick: impl FnMut(usize) -> (i64, bool)) -> Vec<usize> {
        let mut path = vec![0usize];
        loop {
            let cur = *path.last().unwrap();
            let (tok, stop) = pick(cur);
            if stop {
                break;
            }
            match (cur + 1..self.len()).find(|&c| self.parents[c] == cur as i32 && self.tokens[c] == tok) {
                Some(c) => path.push(c),
                None => break,
            }
        }
        path
    }
}

/// Nodes of `t` by decayed score, best first (ties: the lower node), at depths <= `max_depth`, with their scores.
pub fn ranked(t: &DraftTree, max_depth: usize) -> Vec<(f32, usize)> {
    let mut nodes: Vec<(f64, usize)> = (0..t.tokens.len())
        .filter(|&n| t.depth_of(n) <= max_depth && t.scores[n] > 0.0)
        .map(|n| {
            let d = t.depth_of(n);
            let decay = if d >= 2 { DEPTH_DECAY.powi(d as i32) } else { 1.0 };
            (t.scores[n] as f64 * decay, n)
        })
        .collect();
    nodes.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    nodes.into_iter().map(|(_, n)| (t.scores[n], n)).collect()
}

/// The verify rows of `nodes` (an ancestor-closed set, e.g. a prefix of `ranked`) after the pending token.
pub fn rows_of(t: &DraftTree, pending: i64, nodes: &[usize]) -> VerifyRows {
    let mut sorted = nodes.to_vec();
    sorted.sort_unstable();
    let mut v = VerifyRows::single(pending);
    let mut row_of = std::collections::HashMap::new();
    for &n in &sorted {
        let p = t.parents[n];
        let prow = if p < 0 { 0 } else { *row_of.get(&(p as usize)).expect("node set not closed under ancestors") };
        row_of.insert(n, v.tokens.len() as i32);
        v.tokens.push(t.tokens[n]);
        v.parents.push(prow);
        v.scores.push(t.scores[n]);
        v.depth.push(t.depth_of(n));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> DraftTree {
        // K = 2, depth 2: depth 1 nodes 0, 1; children of the frontier (0, 1): nodes 2..6
        DraftTree {
            tokens: vec![10, 11, 20, 21, 22, 23],
            scores: vec![0.6, 0.3, 0.5, 0.05, 0.2, 0.1],
            parents: vec![-1, -1, 0, 0, 1, 1],
            top_k: 2,
            depth: 2,
        }
    }

    #[test]
    fn ranking_is_ancestor_closed() {
        let t = tree();
        let r = ranked(&t, 8);
        assert_eq!(r.iter().map(|x| x.1).collect::<Vec<_>>(), vec![0, 2, 1, 4, 5, 3]);
        let v = rows_of(&t, 7, &[0, 2, 1, 4]);
        assert_eq!(v.tokens, vec![7, 10, 11, 20, 22]);
        assert_eq!(v.parents, vec![-1, 0, 0, 1, 2]);
        assert_eq!(v.path_tokens(4), vec![11, 22]);
    }

    #[test]
    fn accept_walks_the_matching_children() {
        let t = tree();
        let v = rows_of(&t, 7, &[0, 1, 2, 4]);
        // after row 0 the target says 11 (row 2), after it 22 (row 4), then anything
        let picks = [11i64, 0, 22, 0, 99];
        assert_eq!(v.accept(|r| (picks[r], false)), vec![0, 2, 4]);
        assert_eq!(v.accept(|_| (5, false)), vec![0]);
        assert_eq!(v.accept(|r| (picks[r], r == 2)), vec![0, 2]);
    }
}
