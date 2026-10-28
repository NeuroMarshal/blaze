#!/usr/bin/env python3
"""How the NVFP4 KV cache treats a model's real K and V (BLAZE_DUMP of one prefill chunk, model.rs attention()).

Per softmax layer: the share of blocks whose E4M3 scale is subnormal or zero at the global scale g = 1 (blaze's
default), the relative squared error of K and V, and the relative error of the causal attention output on the
model's own queries (the last --queries rows of the chunk, every head) against the exact cache, for:

  plain      g = 1, the four-over-six scale search (as blz_nvfp4_kv_store)
  g-tuned    the best power-of-two global scale per layer for K and V (an oracle of what a calibrated g gains)
  rotated    the randomized Hadamard rotation of --kv-rotate, g = 1
  rot+g      both

Usage: BLAZE_DUMP=/tmp/dump <one eager prefill of <= chunk tokens>; python3 scripts/kv_quant_study.py /tmp/dump
"""
import argparse
import json
import os
import sys

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e4m3_values():
    v = []
    for b in range(0x7F):  # 0x7F is NaN
        e, m = (b >> 3) & 15, b & 7
        v.append(m * 2.0**-9 if e == 0 else (8 + m) * 2.0 ** (e - 10))
    return np.array(v, dtype=np.float64)


E4M3 = e4m3_values()  # code -> value, increasing


def to_e4m3(x):
    """Nearest E4M3 code of non-negative x (saturating at 448)."""
    x = np.minimum(x, 448.0)
    i = np.clip(np.searchsorted(E4M3, x), 1, len(E4M3) - 1)
    lo, hi = E4M3[i - 1], E4M3[i]
    return np.where(x - lo <= hi - x, i - 1, i)


def e2m1_round(a):
    """E2M1 magnitudes of |values| already divided by the scale (ties to the even code, as the store)."""
    idx = ((a > 0.25).astype(np.int8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0))
    return E2M1[idx]


def nvfp4(x, g):
    """x [..., dim] -> (dequantized x, scale codes) with the four-over-six search of blz_nvfp4_kv_store."""
    shape = x.shape
    b = x.reshape(-1, 16).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    lo = to_e4m3(amax / (6.0 * g))
    cands = [lo, to_e4m3(amax / (4.0 * g))] + [lo + s for s in (-1, 1, 2, 3, 4, 5)]
    best_err = np.full(len(b), np.inf, dtype=np.float32)
    best_q = np.zeros_like(b)
    best_code = np.zeros(len(b), dtype=np.int64)
    for c in cands:
        valid = (c >= 0) & (c <= 0x7E)
        cc = np.clip(c, 0, 0x7E)
        t = (E4M3[cc] * g).astype(np.float32)[:, None]
        inv = np.where(t > 0, 1.0 / np.where(t > 0, t, 1.0), 0.0)
        v = b * inv
        q = np.sign(v) * e2m1_round(np.abs(v)) * t
        err = ((q - b) ** 2).sum(axis=1)
        better = valid & (err < best_err)
        best_err = np.where(better, err, best_err)
        best_q[better] = q[better]
        best_code = np.where(better, cc, best_code)
    return best_q.reshape(shape), best_code


def rht_signs(dim):
    i = np.arange(dim, dtype=np.uint64)
    x = (i * np.uint64(0x9E3779B1) + np.uint64(0x7F4A7C15)) & np.uint64(0xFFFFFFFF)
    for shift, mul in ((15, 0x2C1B3C6D), (12, 0x297A2D39)):
        x ^= x >> np.uint64(shift)
        x = (x * np.uint64(mul)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(15)
    return np.where(x & np.uint64(1), -1.0, 1.0).astype(np.float32)


def hadamard(dim):
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < dim:
        h = np.block([[h, h], [h, -h]])
    return h


def rotate(x, signs, h):
    return (x * signs) @ h.T / np.sqrt(x.shape[-1])


def bf16(path, cols):
    raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32).reshape(-1, cols)


def attention(q, k, v, group, scale, q0):
    """Causal attention of query rows q0.. (q [nq, heads, d]) over k, v [t, kv_heads, d]; float64 softmax."""
    nq, heads, _ = q.shape
    out = np.empty_like(q, dtype=np.float64)
    for h in range(heads):
        s = (q[:, h, :].astype(np.float64) @ k[:, h // group, :].T.astype(np.float64)) * scale
        pos = np.arange(k.shape[0])[None, :]
        s = np.where(pos <= (q0 + np.arange(nq))[:, None], s, -np.inf)
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=1, keepdims=True)
        out[:, h, :] = p @ v[:, h // group, :].astype(np.float64)
    return out


def rel(a, b):
    return float(np.sqrt(((a - b) ** 2).sum() / max((b**2).sum(), 1e-30)))


def best_g(x, quant):
    """The power-of-two global scale with the lowest squared error (2^-14 .. 2^4)."""
    best = (np.inf, 1.0)
    for e in range(-14, 5):
        g = 2.0**e
        err = ((quant(x, g) - x) ** 2).sum()
        best = min(best, (err, g))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--queries", type=int, default=256, help="last rows of the chunk used as queries")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.dump, "attn_meta.json")))
    heads, kvh, d, ld = meta["heads"], meta["kv_heads"], meta["head_dim"], meta["ld"]
    group, scale = heads // kvh, 1.0 / np.sqrt(d)
    signs, h = rht_signs(d), hadamard(d)
    deq = lambda x, g: nvfp4(x, g)[0]
    configs = ("plain", "g-tuned", "rotated", "rot+g")
    totals = {c: [] for c in configs}
    print(f"{'layer':>5} {'tokens':>6} | {'K sub/zero %':>12} {'V sub/zero %':>12} | "
          + " | ".join(f"{c:>8} K   V   out" for c in configs))
    for slot in range(meta["layers"]):
        qp, pp = os.path.join(args.dump, f"q_{slot}.bin"), os.path.join(args.dump, f"proj_{slot}.bin")
        if not (os.path.exists(qp) and os.path.exists(pp)):
            continue
        proj = bf16(pp, ld)
        t = proj.shape[0]
        q = bf16(qp, heads * d).reshape(t, heads, d)
        k = proj[:, meta["k_offset"]:meta["k_offset"] + kvh * d].reshape(t, kvh, d)
        v = proj[:, meta["v_offset"]:meta["v_offset"] + kvh * d].reshape(t, kvh, d)
        q0 = max(0, t - args.queries)
        exact = attention(q[q0:], k, v, group, scale, q0)
        _, kc = nvfp4(k, 1.0)
        _, vc = nvfp4(v, 1.0)
        sub = lambda c: (100.0 * np.mean(c < 8), 100.0 * np.mean(c == 0))
        row = f"{slot:>5} {t:>6} | {sub(kc)[0]:>6.1f}/{sub(kc)[1]:<5.1f} {sub(vc)[0]:>6.1f}/{sub(vc)[1]:<5.1f} |"
        for c in configs:
            rot = c.startswith("rot")
            kk, vv, qq = (rotate(k, signs, h), rotate(v, signs, h), rotate(q[q0:], signs, h)) if rot else (k, v, q[q0:])
            gk = best_g(kk, deq) if c.endswith("g") else 1.0
            gv = best_g(vv, deq) if c.endswith("g") else 1.0
            kq, vq = deq(kk, gk), deq(vv, gv)
            out = attention(qq, kq, vq, group, scale, q0)
            if rot:  # back from the rotated basis
                out = (out @ h) / np.sqrt(d) * signs
            e = (rel(kq, kk), rel(vq, vv), rel(out, exact))
            totals[c].append(e)
            row += f" {e[0]:.4f} {e[1]:.4f} {e[2]:.4f} |"
        print(row, flush=True)
    if not totals["plain"]:
        sys.exit("no q_<slot>.bin / proj_<slot>.bin in " + args.dump)
    print("mean  " + " " * 28 + "|" + "".join(
        " {:.4f} {:.4f} {:.4f} |".format(*np.mean(totals[c], axis=0)) for c in configs))


if __name__ == "__main__":
    main()
