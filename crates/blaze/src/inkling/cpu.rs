use super::cuda::{from_bf16, to_bf16};
use super::weights::{ExpertSlice, Experts, PackedMatrix, Packing};
use crate::weights::Checkpoint;
use anyhow::Result;
use rayon::prelude::*;

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn e4m3(code: u8) -> f32 {
    let sign = (code as u32 & 128) << 24;
    let bits = code & 127;
    let mantissa = bits as u32 & 7;
    let exponent = bits as u32 >> 3;
    let value = if bits == 127 {
        f32::NAN
    } else if exponent == 0 {
        mantissa as f32 / 512.0
    } else {
        f32::from_bits((exponent + 120) << 23 | mantissa << 20)
    };
    f32::from_bits(value.to_bits() | sign)
}

fn e8m0(scale: u8) -> f32 {
    match scale {
        0 => f32::from_bits(1 << 22),
        255 => f32::NAN,
        e => f32::from_bits((e as u32) << 23),
    }
}

fn lookup() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| e4m3(i as u8)))
}

pub(super) struct Routed<'a> {
    gate_up: ExpertSlice<'a>,
    down: ExpertSlice<'a>,
}

impl<'a> Routed<'a> {
    pub fn read(ckpt: &'a Checkpoint, weights: &Experts, expert: usize) -> Result<Self> {
        Ok(Self {
            gate_up: weights.gate_up.expert(ckpt, expert)?,
            down: weights.down.expert(ckpt, expert)?,
        })
    }

    pub fn run(&self, weights: &Experts, input: &[f32], caps: (f32, f32)) -> Vec<f32> {
        let gate = matvec(&weights.gate_up, &self.gate_up, input);
        let mut activation = vec![0.0; gate.len() / 2];
        let done = swiglu_vectors(&gate, caps, &mut activation);
        for (dst, row) in activation[done..].iter_mut().zip(gate[done * 2..].chunks_exact(2)) {
            *dst = swiglu_scalar(row[0], row[1], caps);
        }
        matvec(&weights.down, &self.down, &activation)
    }
}

fn swiglu_scalar(g: f32, u: f32, (gate, up): (f32, f32)) -> f32 {
    let g_cap = gate * (g / gate).tanh();
    let u_cap = up * (u / up).tanh();
    from_bf16(to_bf16(g_cap * u_cap / (1.0 + (-g).exp())))
}

fn swiglu_vectors(gate: &[f32], caps: (f32, f32), output: &mut [f32]) -> usize {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        let end = output.len() / 8 * 8;
        // Loads consume 16 input values per 8 output values.
        unsafe { swiglu_avx(&gate[..end * 2], caps, &mut output[..end]) };
        return end;
    }
    0
}

pub(super) fn controller_mix(x: &mut [f32], msg: &[f32], forget: &[f32], gain: &[f32]) {
    let mut start = 0;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        start = x.len() / 8 * 8;
        // All four operands contain one value per channel.
        unsafe {
            controller_mix_avx(&mut x[..start], &msg[..start], &forget[..start], &gain[..start]);
        }
    }
    for (((x, &msg), &forget), &gain) in x[start..]
        .iter_mut()
        .zip(&msg[start..])
        .zip(&forget[start..])
        .zip(&gain[start..])
    {
        *x = (1.0 - forget.tanh()) * *x + gain.tanh() * msg;
    }
}

pub(super) fn softcap(x: &mut [f32], cap: f32) {
    let mut start = 0;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        start = x.len() / 8 * 8;
        unsafe {
            softcap_avx(&mut x[..start], cap);
        }
    }
    for x in &mut x[start..] {
        *x = cap * (*x / cap).tanh();
    }
}

fn matvec(matrix: &PackedMatrix, data: &ExpertSlice<'_>, input: &[f32]) -> Vec<f32> {
    let (cols, rows) = (matrix.cols, matrix.rows);
    let (code, scale, alpha) = match matrix.packing {
        Packing::Nvfp4 => (cols / 2, cols / 16, data.alpha.unwrap()),
        Packing::Mxfp8 => (cols, cols / 32, 1.0),
    };
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    let mut output = vec![0.0; rows];
    output.par_chunks_mut(64).enumerate().for_each(|(page, out)| {
        let row = page * 64;
        let codes = &data.codes[row * code..];
        let scales = &data.scales[row * scale..];
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            // The row shape was validated against the checkpoint. Every vector load stays within its row.
            unsafe {
                match matrix.packing {
                    Packing::Nvfp4 => nvfp4_rows(codes, scales, input, alpha, out),
                    Packing::Mxfp8 => mxfp8_rows(codes, scales, input, out),
                }
            }
            return;
        }
        for ((dst, codes), scales) in out.iter_mut().zip(codes.chunks_exact(code)).zip(scales.chunks_exact(scale)) {
            let value = scalar_dot(matrix.packing, codes, scales, input);
            *dst = from_bf16(to_bf16(value * alpha));
        }
    });
    output
}

fn scalar_dot(packing: Packing, codes: &[u8], scales: &[u8], input: &[f32]) -> f32 {
    let table = lookup();
    match packing {
        Packing::Nvfp4 => codes
            .iter()
            .enumerate()
            .map(|(i, &u)| {
                let s = table[scales[i / 8] as usize];
                (E2M1[(u & 15) as usize] * input[2 * i] + E2M1[(u >> 4) as usize] * input[2 * i + 1]) * s
            })
            .sum(),
        Packing::Mxfp8 => codes
            .iter()
            .enumerate()
            .map(|(i, &u)| table[u as usize] * e8m0(scales[i / 32]) * input[i])
            .sum(),
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn exp_avx(x: __m256) -> __m256 {
    let f = _mm256_set1_ps;
    let x = _mm256_min_ps(_mm256_max_ps(x, f(-80.0)), f(80.0));
    let n = _mm256_cvtps_epi32(_mm256_mul_ps(x, f(std::f32::consts::LOG2_E)));
    let r = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(n), f(std::f32::consts::LN_2), x);
    let mut p = f(1.0 / 720.0);
    for c in [1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
        p = _mm256_fmadd_ps(p, r, f(c));
    }
    let two_n = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23));
    _mm256_mul_ps(two_n, p)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn tanh_avx(x: __m256) -> __m256 {
    let sign_bit = _mm256_set1_ps(-0.0);
    let sign = _mm256_and_ps(x, sign_bit);
    let mag = _mm256_andnot_ps(sign_bit, x);
    let exp = exp_avx(_mm256_mul_ps(mag, _mm256_set1_ps(-2.0)));
    let one = _mm256_set1_ps(1.0);
    _mm256_xor_ps(_mm256_div_ps(_mm256_sub_ps(one, exp), _mm256_add_ps(one, exp)), sign)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn controller_mix_avx(x: &mut [f32], msg: &[f32], forget: &[f32], gain: &[f32]) {
    let one = _mm256_set1_ps(1.0);
    for (((x, msg), forget), gain) in x
        .chunks_exact_mut(8)
        .zip(msg.chunks_exact(8))
        .zip(forget.chunks_exact(8))
        .zip(gain.chunks_exact(8))
    {
        let (xv, mv, fv, gv) = unsafe {
            (
                _mm256_loadu_ps(x.as_ptr()),
                _mm256_loadu_ps(msg.as_ptr()),
                _mm256_loadu_ps(forget.as_ptr()),
                _mm256_loadu_ps(gain.as_ptr()),
            )
        };
        let keep = _mm256_mul_ps(xv, _mm256_sub_ps(one, tanh_avx(fv)));
        let add = _mm256_mul_ps(mv, tanh_avx(gv));
        unsafe {
            _mm256_storeu_ps(x.as_mut_ptr(), _mm256_add_ps(keep, add));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softcap_avx(x: &mut [f32], cap: f32) {
    let inv = _mm256_set1_ps(cap.recip());
    let cap = _mm256_set1_ps(cap);
    for row in x.chunks_exact_mut(8) {
        let x = unsafe { _mm256_loadu_ps(row.as_ptr()) };
        let y = _mm256_mul_ps(tanh_avx(_mm256_mul_ps(x, inv)), cap);
        unsafe {
            _mm256_storeu_ps(row.as_mut_ptr(), y);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_avx(gate: &[f32], (gate_cap, up_cap): (f32, f32), output: &mut [f32]) {
    let permutation = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    let one = _mm256_set1_ps(1.0);
    let g_scale = _mm256_set1_ps(gate_cap.recip());
    let u_scale = _mm256_set1_ps(up_cap.recip());
    for (values, output) in gate.chunks_exact(16).zip(output.chunks_exact_mut(8)) {
        let a = _mm256_permutevar8x32_ps(unsafe { _mm256_loadu_ps(values.as_ptr()) }, permutation);
        let b = _mm256_permutevar8x32_ps(unsafe { _mm256_loadu_ps(values.as_ptr().add(8)) }, permutation);
        let g = _mm256_permute2f128_ps::<0x20>(a, b);
        let u = _mm256_permute2f128_ps::<0x31>(a, b);
        let g_cap = _mm256_mul_ps(tanh_avx(_mm256_mul_ps(g, g_scale)), _mm256_set1_ps(gate_cap));
        let u_cap = _mm256_mul_ps(tanh_avx(_mm256_mul_ps(u, u_scale)), _mm256_set1_ps(up_cap));
        let denom = _mm256_add_ps(one, exp_avx(_mm256_xor_ps(g, _mm256_set1_ps(-0.0))));
        let activated = _mm256_div_ps(_mm256_mul_ps(g_cap, u_cap), denom);
        let bits = _mm256_castps_si256(activated);
        let tie = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
        let rounded = _mm256_add_epi32(bits, _mm256_add_epi32(tie, _mm256_set1_epi32(0x7fff)));
        let bf16 = _mm256_castsi256_ps(_mm256_and_si256(rounded, _mm256_set1_epi32(!0xffff)));
        unsafe { _mm256_storeu_ps(output.as_mut_ptr(), bf16) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn nvfp4_rows(codes: &[u8], scales: &[u8], input: &[f32], alpha: f32, out: &mut [f32]) {
    let code_width = input.len() / 2;
    let blocks = input.len() / 16;
    let nibble = _mm_set1_epi8(15);
    let lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    let table = lookup();
    for ((dst, codes), scales) in out
        .iter_mut()
        .zip(codes.chunks_exact(code_width))
        .zip(scales.chunks_exact(blocks))
    {
        let mut acc_a = _mm256_setzero_ps();
        let mut acc_b = _mm256_setzero_ps();
        for ((packed, &scale), x) in codes.chunks_exact(8).zip(scales).zip(input.chunks_exact(16)) {
            let w = unsafe { _mm_loadl_epi64(packed.as_ptr().cast()) };
            let lo = _mm_and_si128(w, nibble);
            let hi = _mm_and_si128(_mm_srli_epi16(w, 4), nibble);
            let values = _mm_shuffle_epi8(lut, _mm_unpacklo_epi8(lo, hi));
            let left = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(values));
            let right = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(values, 8)));
            let scale = _mm256_set1_ps(table[scale as usize] * 0.5);
            let xa = unsafe { _mm256_loadu_ps(x.as_ptr()) };
            let xb = unsafe { _mm256_loadu_ps(x.as_ptr().add(8)) };
            acc_a = _mm256_fmadd_ps(xa, _mm256_mul_ps(left, scale), acc_a);
            acc_b = _mm256_fmadd_ps(xb, _mm256_mul_ps(right, scale), acc_b);
        }
        *dst = from_bf16(to_bf16(unsafe { horizontal(_mm256_add_ps(acc_a, acc_b)) } * alpha));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mxfp8_rows(codes: &[u8], scales: &[u8], input: &[f32], out: &mut [f32]) {
    let blocks = input.len() / 32;
    let low = _mm256_set1_epi32(7);
    let exp_mask = _mm256_set1_epi32(15);
    let sign_bit = _mm256_set1_epi32(128);
    let zero = _mm256_setzero_si256();
    let exp_bias = _mm256_set1_epi32(120);
    let subnormal = _mm256_set1_ps(1.0 / 512.0);
    for ((dst, codes), scales) in out
        .iter_mut()
        .zip(codes.chunks_exact(input.len()))
        .zip(scales.chunks_exact(blocks))
    {
        let mut acc = [_mm256_setzero_ps(); 4];
        for ((packed, &scale), x) in codes.chunks_exact(32).zip(scales).zip(input.chunks_exact(32)) {
            let scale = _mm256_set1_ps(e8m0(scale));
            for (i, acc) in acc.iter_mut().enumerate() {
                let packed = unsafe { _mm_loadl_epi64(packed.as_ptr().add(i * 8).cast()) };
                let codes = _mm256_cvtepu8_epi32(packed);
                let sign = _mm256_slli_epi32(_mm256_and_si256(codes, sign_bit), 24);
                let exponent = _mm256_and_si256(_mm256_srli_epi32(codes, 3), exp_mask);
                let mantissa = _mm256_and_si256(codes, low);
                let normal = _mm256_or_si256(
                    _mm256_slli_epi32(_mm256_add_epi32(exponent, exp_bias), 23),
                    _mm256_slli_epi32(mantissa, 20),
                );
                let sub = _mm256_castps_si256(_mm256_mul_ps(_mm256_cvtepi32_ps(mantissa), subnormal));
                let size = _mm256_blendv_epi8(normal, sub, _mm256_cmpeq_epi32(exponent, zero));
                let invalid = _mm256_cmpeq_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(127)), _mm256_set1_epi32(127));
                let values = _mm256_castsi256_ps(_mm256_or_si256(
                    _mm256_or_si256(size, sign),
                    _mm256_and_si256(invalid, _mm256_set1_epi32(0x7fc0_0000)),
                ));
                let x = unsafe { _mm256_loadu_ps(x.as_ptr().add(i * 8)) };
                *acc = _mm256_fmadd_ps(x, _mm256_mul_ps(values, scale), *acc);
            }
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        *dst = from_bf16(to_bf16(unsafe { horizontal(sum) }));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal(x: __m256) -> f32 {
    let x = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    _mm_cvtss_f32(_mm_add_ss(x, _mm_shuffle_ps::<1>(x, x)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_tail_and_simd_match_scalar() {
        let values: Vec<_> = (0..23).map(|i| (i as f32 * 0.17).sin()).collect();
        let f: Vec<_> = values.iter().map(|&v| v * 0.21).collect();
        let mut output = values.clone();
        controller_mix(&mut output, &values, &f, &f);
        for (actual, expected) in output.iter().zip(&values) {
            assert!((actual - expected).abs() < 2e-6);
        }
        let scalar: Vec<_> = values.iter().map(|v| 30.0 * (v / 30.0).tanh()).collect();
        softcap(&mut output, 30.0);
        for (actual, expected) in output.iter().zip(scalar) {
            assert!((actual - expected).abs() < 4e-6, "{actual} vs {expected}");
        }
    }

    #[test]
    fn vector_swiglu_matches_scalar_within_bf16_precision() {
        let values: Vec<_> = (0..512).map(|i| (i as f32 * 1.17).sin() * 45.0).collect();
        let mut output = vec![0.0; values.len() / 2];
        let caps = (4.0, 25.0);
        let done = swiglu_vectors(&values, caps, &mut output);
        for (input, got) in values.chunks_exact(2).zip(&output[..done]) {
            let want = swiglu_scalar(input[0], input[1], caps);
            assert!((got - want).abs() <= (want.abs() * 0.008).max(1e-5), "{got} vs {want}");
        }
    }

    #[test]
    fn packed_avx_rows_match_scalar() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let input: Vec<_> = (0..128).map(|i| (i % 7) as f32 - 3.0).collect();
            for (packing, code_len, scales) in [(Packing::Nvfp4, 64, vec![0x38; 8]), (Packing::Mxfp8, 128, vec![127; 4])] {
                let codes: Vec<_> = (0..code_len).map(|i| (i % 64) as u8).collect();
                let mut actual = [0.0];
                unsafe {
                    match packing {
                        Packing::Nvfp4 => nvfp4_rows(&codes, &scales, &input, 1.0, &mut actual),
                        Packing::Mxfp8 => mxfp8_rows(&codes, &scales, &input, &mut actual),
                    }
                }
                let expected = from_bf16(to_bf16(scalar_dot(packing, &codes, &scales, &input)));
                assert_eq!(actual[0], expected, "{packing:?}");
            }
        }
    }
}
