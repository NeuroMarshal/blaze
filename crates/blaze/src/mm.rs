//! Image placeholder embeddings, prefix-cache keys and Qwen3-VL M-RoPE positions.
//! Text after an image advances by max(grid height, grid width), rather than its patch count.

use std::sync::Arc;

/// One image of a prompt.
#[derive(Clone)]
pub struct ImageEmbeds {
    /// token index of the first image placeholder token
    pub start: usize,
    /// merged grid (the placeholder tokens are gh * gw, row by row)
    pub grid_h: usize,
    pub grid_w: usize,
    /// content hash (prefix-cache key)
    pub hash: u64,
    /// [grid_h * grid_w, hidden] bf16, the encoder's output
    pub emb: Arc<Vec<u16>>,
}

impl ImageEmbeds {
    pub fn tokens(&self) -> usize {
        self.grid_h * self.grid_w
    }
}

/// The images of a prompt in token order, with the position bookkeeping of M-RoPE.
pub struct MmInput {
    pub images: Vec<ImageEmbeds>,
    pub hidden: usize,
    /// position of each image's first token (temporal stream)
    pos0: Vec<i64>,
    /// position minus token index after each image (the tokens that follow it)
    shift_after: Vec<i64>,
}

/// A run of image rows inside one step: `count` rows of `emb` from row `row0` go to step rows `at..at + count`.
#[derive(Clone)]
pub struct ImgSeg {
    pub at: usize,
    pub emb: Arc<Vec<u16>>,
    pub row0: usize,
    pub count: usize,
}

impl MmInput {
    pub fn new(images: Vec<ImageEmbeds>, hidden: usize) -> Self {
        let mut pos0 = Vec::with_capacity(images.len());
        let mut shift_after = Vec::with_capacity(images.len());
        let mut shift = 0i64;
        for im in &images {
            pos0.push(im.start as i64 + shift);
            shift += im.grid_h.max(im.grid_w) as i64 - im.tokens() as i64;
            shift_after.push(shift);
        }
        Self { images, hidden, pos0, shift_after }
    }

    /// Image tokens of the prompt.
    pub fn image_tokens(&self) -> usize {
        self.images.iter().map(ImageEmbeds::tokens).sum()
    }

    /// Positions (temporal, height, width) of the token at index `j`.
    pub fn pos3(&self, j: usize) -> [i64; 3] {
        let k = self.images.partition_point(|im| im.start <= j);
        if k == 0 {
            return [j as i64; 3];
        }
        let im = &self.images[k - 1];
        let off = j - im.start;
        if off < im.tokens() {
            let p0 = self.pos0[k - 1];
            return [p0, p0 + (off / im.grid_w) as i64, p0 + (off % im.grid_w) as i64];
        }
        [j as i64 + self.shift_after[k - 1]; 3]
    }

    /// The temporal position of token `j` (all three streams for a text token).
    pub fn pos_t(&self, j: usize) -> i64 {
        self.pos3(j)[0]
    }

    /// Does any token of `start..start + len` belong to an image (so that the three streams differ)?
    pub fn touches_image(&self, start: usize, len: usize) -> bool {
        self.images.iter().any(|im| im.start < start + len && start < im.start + im.tokens())
    }

    /// The runs of image rows within tokens `start..start + len` of the prompt, as rows of the step starting at `at`.
    pub fn segments(&self, start: usize, len: usize, at: usize) -> Vec<ImgSeg> {
        let mut out = Vec::new();
        for im in &self.images {
            let (lo, hi) = (im.start.max(start), (im.start + im.tokens()).min(start + len));
            if lo < hi {
                out.push(ImgSeg { at: at + (lo - start), emb: im.emb.clone(), row0: lo - im.start, count: hi - lo });
            }
        }
        out
    }

    /// Prefix-cache key of a token list: the image placeholder rows replaced by values that depend on the image's content
    /// and the row (outside the vocabulary), so that a cached prefix matches only a prompt with the same images.
    pub fn keyed(&self, tokens: &mut [i64]) {
        for im in &self.images {
            for i in 0..im.tokens() {
                if let Some(t) = tokens.get_mut(im.start + i) {
                    let h = im.hash.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).rotate_left(17).wrapping_mul(0xD6E8_FEB8_6659_FD93);
                    *t = ((h >> 2) | 1 << 62) as i64;
                }
            }
        }
    }
}

/// Is `id` a prefix-cache key of an image row (see `MmInput::keyed`) rather than a vocabulary id?
pub fn is_image_key(id: i64) -> bool {
    id >= 1 << 62
}

/// FNV-1a over bytes (an image's content hash; collisions only cost a wrong prefix-cache hit between two different
/// images of the same size, so 64 bits of a simple hash are enough together with the position mix in `keyed`).
pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for chunk in data.chunks(8) {
        let mut w = [0u8; 8];
        w[..chunk.len()].copy_from_slice(chunk);
        h = (h ^ u64::from_le_bytes(w)).wrapping_mul(0x0000_0100_0000_01B3).rotate_left(23);
    }
    h ^ data.len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn im(start: usize, gh: usize, gw: usize) -> ImageEmbeds {
        ImageEmbeds { start, grid_h: gh, grid_w: gw, hash: start as u64, emb: Arc::new(Vec::new()) }
    }

    #[test]
    fn text_before_and_after_one_image() {
        // 3 text tokens, then a 2 x 4 image at index 3 (8 tokens), then text from index 11
        let mm = MmInput::new(vec![im(3, 2, 4)], 8);
        assert_eq!(mm.pos3(2), [2, 2, 2]);
        assert_eq!(mm.pos3(3), [3, 3, 3]);
        assert_eq!(mm.pos3(4), [3, 3, 4]);
        assert_eq!(mm.pos3(7), [3, 4, 3]);
        assert_eq!(mm.pos3(10), [3, 4, 6]);
        // the text after continues from p0 + max(gh, gw) = 7
        assert_eq!(mm.pos3(11), [7, 7, 7]);
        assert_eq!(mm.pos3(12), [8, 8, 8]);
    }

    #[test]
    fn two_images() {
        let mm = MmInput::new(vec![im(1, 2, 2), im(7, 3, 1)], 8);
        // first image tokens 1..5 at p0 = 1, text from index 5 at position 3
        assert_eq!(mm.pos3(5), [3, 3, 3]);
        assert_eq!(mm.pos3(6), [4, 4, 4]);
        // second image at index 7: p0 = 5 (index 7 + shift -2)
        assert_eq!(mm.pos3(7), [5, 5, 5]);
        assert_eq!(mm.pos3(9), [5, 7, 5]);
        assert_eq!(mm.pos3(10), [8, 8, 8]);
    }

    #[test]
    fn segments_cut_by_the_step() {
        let mm = MmInput::new(vec![im(3, 2, 4)], 8);
        let s = mm.segments(5, 4, 100);
        assert_eq!(s.len(), 1);
        assert_eq!((s[0].at, s[0].row0, s[0].count), (100, 2, 4));
        assert!(mm.segments(11, 3, 0).is_empty());
        assert!(mm.touches_image(0, 4) && !mm.touches_image(11, 5));
    }

    #[test]
    fn keys_differ_by_image() {
        let (a, b) = (MmInput::new(vec![im(1, 1, 2)], 8), MmInput::new(vec![ImageEmbeds { hash: 99, ..im(1, 1, 2) }], 8));
        let (mut ta, mut tb) = (vec![5i64; 4], vec![5i64; 4]);
        a.keyed(&mut ta);
        b.keyed(&mut tb);
        assert_eq!(ta[0], 5);
        assert_ne!(ta[1], tb[1]);
        assert_ne!(ta[1], ta[2]);
        assert!(ta[1] > 1 << 40);
    }
}
