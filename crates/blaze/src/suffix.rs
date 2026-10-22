//! Suffix drafting follows repeated prompt/output spans, using a token-pair position index.
//! Candidate drafts are ranked by expected accepted length, as in Arctic Inference suffix decoding.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Multiplicative hash of the pair key (keys are already well spread token pairs).
#[derive(Default)]
struct PairHasher(u64);

impl Hasher for PairHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = (x ^ (x >> 29)).wrapping_mul(0xbf58476d1ce4e5b9);
    }
}

const SEP: i32 = -1;

/// Suffix drafting limits (vLLM's suffix_decoding_* options).
#[derive(Clone, Copy, Debug)]
pub struct SuffixConfig {
    /// longest context suffix matched
    pub max_depth: usize,
    /// draft tokens per matched suffix token (a match of n tokens drafts at most factor * n)
    pub max_spec_factor: f32,
    /// draft probability floor
    pub min_token_prob: f32,
    /// most recent occurrences of a token pair examined
    pub max_occurrences: usize,
}

impl Default for SuffixConfig {
    fn default() -> Self {
        Self { max_depth: 24, max_spec_factor: 1.0, min_token_prob: 0.1, max_occurrences: 256 }
    }
}

/// Token sequences (separated) with the positions of every token pair.
#[derive(Default)]
pub struct SuffixIndex {
    tokens: Vec<i32>,
    pairs: HashMap<u64, Vec<u32>, BuildHasherDefault<PairHasher>>,
}

fn key(a: i32, b: i32) -> u64 {
    ((a as u32 as u64) << 32) | b as u32 as u64
}

impl SuffixIndex {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn push(&mut self, t: i32) {
        if let Some(&p) = self.tokens.last()
            && p != SEP
        {
            self.pairs.entry(key(p, t)).or_default().push(self.tokens.len() as u32);
        }
        self.tokens.push(t);
    }

    pub fn extend(&mut self, ts: impl IntoIterator<Item = i32>) {
        for t in ts {
            self.push(t);
        }
    }

    /// Ends the current sequence (continuations never cross it).
    pub fn end_seq(&mut self) {
        if self.tokens.last().is_some_and(|&t| t != SEP) {
            self.tokens.push(SEP);
        }
    }

    /// Keeps the newest sequences holding about `keep` tokens (the index is rebuilt).
    pub fn trim(&mut self, keep: usize) {
        if self.tokens.len() <= keep {
            return;
        }
        let cut = self.tokens.len() - keep;
        let start = self.tokens[cut..].iter().position(|&t| t == SEP).map_or(self.tokens.len(), |i| cut + i + 1);
        let rest = self.tokens.split_off(start);
        self.tokens.clear();
        self.pairs.clear();
        for t in rest {
            if t == SEP { self.end_seq() } else { self.push(t) }
        }
    }

    /// Best draft continuing `ctx` (at most `max_len` tokens) and its expected accepted tokens (sum of the cumulative
    /// probabilities). Positions without a continuation are skipped, so ctx may be the end of this very corpus.
    pub fn draft(&self, ctx: &[i32], max_len: usize, cfg: &SuffixConfig) -> (Vec<i32>, f32) {
        let n = ctx.len();
        if n < 2 || max_len == 0 {
            return (Vec::new(), 0.0);
        }
        let Some(pos) = self.pairs.get(&key(ctx[n - 2], ctx[n - 1])) else { return (Vec::new(), 0.0) };
        // occurrences (position of the pair's second token) with a continuation, newest first, and match lengths
        let mut occ: Vec<(usize, usize)> = Vec::new();
        for &p in pos.iter().rev() {
            let p = p as usize;
            if p + 1 >= self.tokens.len() || self.tokens[p + 1] == SEP {
                continue;
            }
            let mut m = 2;
            while m < cfg.max_depth && m < n && m <= p && self.tokens[p - m] == ctx[n - 1 - m] && self.tokens[p - m] != SEP {
                m += 1;
            }
            occ.push((p, m));
            if occ.len() >= cfg.max_occurrences {
                break;
            }
        }
        let mut lens: Vec<usize> = occ.iter().map(|o| o.1).collect();
        lens.sort_unstable();
        lens.dedup();
        let (mut best, mut best_score) = (Vec::new(), 0.0f32);
        let mut counts: HashMap<i32, usize> = HashMap::new();
        for &l in &lens {
            let limit = ((cfg.max_spec_factor * l as f32).ceil() as usize).min(max_len);
            let mut live: Vec<usize> = occ.iter().filter(|o| o.1 >= l).map(|o| o.0 + 1).collect();
            let (mut draft, mut score, mut prob) = (Vec::new(), 0.0f32, 1.0f32);
            while draft.len() < limit && !live.is_empty() {
                counts.clear();
                for &q in &live {
                    *counts.entry(self.tokens[q]).or_insert(0) += 1;
                }
                // most frequent next token; ties to the newest occurrence (live is newest first)
                let top = counts.values().copied().max().unwrap_or(0);
                let Some(&tok) = live.iter().map(|&q| &self.tokens[q]).find(|t| counts[*t] == top) else { break };
                let p = prob * top as f32 / live.len() as f32;
                if p < cfg.min_token_prob {
                    break;
                }
                prob = p;
                score += p;
                draft.push(tok);
                live.retain(|&q| self.tokens[q] == tok);
                for q in live.iter_mut() {
                    *q += 1;
                }
                live.retain(|&q| q < self.tokens.len() && self.tokens[q] != SEP);
            }
            if score > best_score {
                (best, best_score) = (draft, score);
            }
        }
        (best, best_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeats_a_seen_continuation() {
        let mut ix = SuffixIndex::default();
        let text = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3];
        ix.extend(text);
        let (d, s) = ix.draft(&text, 8, &SuffixConfig::default());
        // match "1 2 3" (3 tokens) -> at most 3 tokens of continuation
        assert_eq!(d, vec![4, 5, 6]);
        assert!(s > 2.9);
    }

    #[test]
    fn follows_the_majority_and_stops_at_the_floor() {
        let mut ix = SuffixIndex::default();
        for tail in [[7, 8], [7, 9], [7, 8], [5, 5]] {
            ix.extend([1, 2, 3, 4]);
            ix.extend(tail);
            ix.end_seq();
        }
        let cfg = SuffixConfig { min_token_prob: 0.5, ..Default::default() };
        // 3 of 4 continue with 7 (0.75), then 2 of 3 with 8 (0.5)
        let (d, _) = ix.draft(&[0, 1, 2, 3, 4], 8, &cfg);
        assert_eq!(d, vec![7, 8]);
        let cfg = SuffixConfig { min_token_prob: 0.6, ..Default::default() };
        assert_eq!(ix.draft(&[0, 1, 2, 3, 4], 8, &cfg).0, vec![7]);
    }

    #[test]
    fn trim_keeps_the_newest_sequences() {
        let mut ix = SuffixIndex::default();
        ix.extend([1, 2, 3]);
        ix.end_seq();
        ix.extend([4, 5, 6]);
        ix.end_seq();
        ix.trim(5);
        assert_eq!(ix.len(), 4);
        assert_eq!(ix.draft(&[4, 5], 4, &SuffixConfig::default()).0, vec![6]);
        assert!(ix.draft(&[1, 2], 4, &SuffixConfig::default()).0.is_empty());
    }
}
