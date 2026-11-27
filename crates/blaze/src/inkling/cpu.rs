#[cfg(test)]
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

#[cfg(all(test, target_arch = "x86_64"))]
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
}
