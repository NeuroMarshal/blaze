pub(crate) fn embedding_rows(
    prev: &[i64],
    tokens: &[i64],
    eos: i64,
    hpn: usize,
    mult: &[i64],
    sizes: &[i64],
    offsets: &[i64],
) -> Vec<i64> {
    let mut win: Vec<i64> = vec![eos, eos];
    for (i, &t) in prev.iter().rev().take(2).enumerate() {
        win[1 - i] = t;
    }
    win.extend_from_slice(tokens);
    let mut ids = Vec::with_capacity(tokens.len() * 2 * hpn);
    for i in 0..tokens.len() {
        let (w0, w1, w2) = (win[i], win[i + 1], win[i + 2]);
        let reach = if w1 == eos {
            0
        } else if w0 == eos {
            1
        } else {
            2
        };
        let t1 = (if reach >= 1 { w1 } else { eos }) as u64;
        let t2 = (if reach >= 2 { w0 } else { eos }) as u64;
        let m2 = (w2 as u64).wrapping_mul(mult[0] as u64) ^ t1.wrapping_mul(mult[1] as u64);
        let m3 = m2 ^ t2.wrapping_mul(mult[2] as u64);
        for h in 0..2 * hpn {
            let mix = (if h < hpn { m2 } else { m3 }) as i64;
            ids.push(mix.rem_euclid(sizes[h]) + offsets[h]);
        }
    }
    ids
}
