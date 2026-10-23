#!/usr/bin/env python3
"""Offline MXFP8 quantization of an EAGLE-4 drafter (ModelOpt checkpoint format, quant_algo MXFP8), calibrated.

The drafter's linear layers (fc, q/k/v, o, gate/up, down) become E4M3 codes with an E8M0 scale per 32 inputs; the
norms, the low-rank head and the LM head stay BF16. Calibration data are the target's real features: blaze run with
BLAZE_DRAFT_DUMP=dir BLAZE_DRAFT_DUMP_CALLS=n dumps its first n drafting calls (captured hidden states, tokens,
positions) per run in dir/00000, dir/00001, ...; every run directory given here is replayed through the BF16 draft in
order (teacher forced, KV cache with the sliding window) and the inputs of every linear layer are accumulated into
H = X^T X. Then per layer:

- block scales: per 32 inputs of a row the E8M0 exponent from the block maximum (ceil(log2(amax / 448))) or one step
  below it (finer steps, the largest values clipped), whichever has the smaller error weighted by diag(H);
- GPTQ (Frantar et al.) with those scales: columns quantized in order, each one's error spread over the columns not
  yet quantized through the inverse Hessian (blocks of 128 columns, lazy batch updates), damping 1% of mean diag(H).

The output error on the calibration inputs, relative to the BF16 layer, is printed for round-to-nearest and GPTQ.

    quantize_eagle4.py --draft ~/models/.../Qwen3.8-27B-EAGLE4 --target ~/models/.../Qwen3.8-27B-...-NVFP4 \\
        --calib dump_a dump_b ... --out ~/models/.../Qwen3.8-27B-EAGLE4-MXFP8
"""

import argparse
import glob
import json
import math
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E4M3_MAX = 448.0
LAYERS = {
    "fc": ["fc"],
    "qkv": ["layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj"],
    "o": ["layers.0.self_attn.o_proj"],
    "gu": ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"],
    "down": ["layers.0.mlp.down_proj"],
}


def rmsnorm(x, w, eps):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)


def rope(x, pos, theta, hd):
    inv = theta ** (-torch.arange(0, hd // 2, device=x.device, dtype=torch.float64) * 2 / hd)
    ang = pos.double()[:, None] * inv[None, :]
    cos, sin = ang.cos().float()[:, None, :], ang.sin().float()[:, None, :]
    x1, x2 = x[..., : hd // 2].float(), x[..., hd // 2 :].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(torch.bfloat16)


def mm(x, w):
    return (x.float() @ w.float().t()).to(torch.bfloat16)


class Calibrator:
    """The BF16 draft replayed over dumped calls; accumulates X^T X of every linear layer's input."""

    def __init__(self, cfg, w, embed, device):
        self.cfg, self.w, self.embed, self.dev = cfg, w, embed, device
        self.eps = cfg["rms_norm_eps"]
        self.hd, self.heads, self.kvh = cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.window = cfg["sliding_window"] if cfg.get("use_sliding_window") else 0
        self.theta = (cfg.get("rope_parameters") or {}).get("rope_theta") or cfg.get("rope_theta", 1e6)
        self.H = {}
        self.rows = 0
        self.reset()

    def reset(self):
        self.K, self.V, self.P = [], [], []

    def add(self, name, x):
        x = x.float()
        if name not in self.H:
            self.H[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=self.dev)
        self.H[name].addmm_(x.t(), x)

    def call(self, feats, tokens, pos):
        w, p = self.w, "layers.0."
        parts = [rmsnorm(feats[:, i], w[f"fc_norm.{i}.weight"], self.eps) for i in range(3)]
        fc_in = torch.cat(parts, -1)
        self.add("fc", fc_in)
        hidden = mm(fc_in, w["fc.weight"])
        x = torch.cat([rmsnorm(self.embed[tokens], w[p + "input_layernorm.weight"], self.eps),
                       rmsnorm(hidden, w[p + "hidden_norm.weight"], self.eps)], -1)
        self.add("qkv", x)
        n = len(tokens)
        q = mm(x, w[p + "self_attn.q_proj.weight"]).view(n, self.heads, self.hd)
        k = mm(x, w[p + "self_attn.k_proj.weight"]).view(n, self.kvh, self.hd)
        v = mm(x, w[p + "self_attn.v_proj.weight"]).view(n, self.kvh, self.hd)
        q, k = rope(q, pos, self.theta, self.hd), rope(k, pos, self.theta, self.hd)
        self.K.append(k), self.V.append(v), self.P.append(pos)
        # keep the window's keys only
        Kc, Vc, Pc = torch.cat(self.K), torch.cat(self.V), torch.cat(self.P)
        if self.window and len(Pc) > 2 * self.window:
            keep = Pc > Pc.max() - self.window - n
            Kc, Vc, Pc = Kc[keep], Vc[keep], Pc[keep]
            self.K, self.V, self.P = [Kc], [Vc], [Pc]
        rep = self.heads // self.kvh
        out = torch.empty(n, self.heads, self.hd, dtype=torch.bfloat16, device=self.dev)
        for r0 in range(0, n, 512):
            qq, pp = q[r0 : r0 + 512].float(), pos[r0 : r0 + 512]
            s = torch.einsum("qhd,khd->hqk", qq, Kc.float().repeat_interleave(rep, 1)) / math.sqrt(self.hd)
            vis = Pc[None, :] <= pp[:, None]
            if self.window:
                vis &= Pc[None, :] > pp[:, None] - self.window
            s = s.masked_fill(~vis[None], float("-inf"))
            out[r0 : r0 + 512] = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), Vc.float().repeat_interleave(rep, 1)).to(torch.bfloat16)
        o_in = out.reshape(n, -1)
        self.add("o", o_in)
        resid = (hidden.float() + mm(o_in, w[p + "self_attn.o_proj.weight"]).float()).to(torch.bfloat16)
        x2 = rmsnorm(resid, w[p + "post_attention_layernorm.weight"], self.eps)
        self.add("gu", x2)
        g, u = mm(x2, w[p + "mlp.gate_proj.weight"]), mm(x2, w[p + "mlp.up_proj.weight"])
        act = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16)
        self.add("down", act)
        self.rows += n


def e8m0_rtn(amax):
    """E8M0 exponents (biased) fitting each block maximum into the E4M3 range (vLLM / ModelOpt rule)."""
    return (torch.ceil(torch.log2(amax.clamp(min=2.0 ** -126) / E4M3_MAX)) + 127).clamp(1, 254)


def qdq(w, e):
    """Quantize-dequantize with per-32 biased exponents e ([n, k / 32])."""
    n, k = w.shape
    s = torch.exp2(e.float() - 127).repeat_interleave(32, dim=1)
    q = (w / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return q * s


def choose_scales(w, hdiag):
    """Per block: the max-fitting exponent or one below it, by error weighted with diag(H)."""
    n, k = w.shape
    amax = w.abs().view(n, k // 32, 32).amax(-1)
    e0 = e8m0_rtn(amax)
    best = e0.clone()
    errs = []
    for e in (e0, (e0 - 1).clamp(min=1)):
        err = ((w - qdq(w, e)) ** 2 * hdiag[None, :]).view(n, k // 32, 32).sum(-1)
        errs.append(err)
    best = torch.where(errs[1] < errs[0], (e0 - 1).clamp(min=1), e0)
    return best, (errs[1] < errs[0]).float().mean().item()


def gptq(w, H, e, block=128, damp=0.01):
    """GPTQ with fixed per-32 MXFP8 scales; returns the quantized-dequantized weight (float)."""
    w = w.clone().float()
    n, k = w.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    w[:, dead] = 0
    H += torch.eye(k, device=H.device) * damp * torch.mean(torch.diag(H))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    scale = torch.exp2(e.float() - 127)  # [n, k / 32]
    q_all = torch.zeros_like(w)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        W1 = w[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = W1[:, i]
            s = scale[:, (i1 + i) // 32]
            q = (col / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * s
            Q1[:, i] = q
            err = (col - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        q_all[:, i1:i2] = Q1
        w[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return q_all


def rel_err(w, wq, H):
    """Output error of the layer on the calibration inputs: tr(D H D^T) / tr(W H W^T), D = W - Wq."""
    d = (w - wq).float()
    return (torch.einsum("nk,kj,nj->", d, H, d) / torch.einsum("nk,kj,nj->", w.float(), H, w.float())).item()


def encode(wq, e):
    """E4M3 codes of a quantized weight (exact) and its scales as uint8."""
    s = torch.exp2(e.float() - 127).repeat_interleave(32, dim=1)
    return (wq / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), e.to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--target", required=True, help="target checkpoint (the shared token embedding)")
    ap.add_argument("--calib", nargs="+", required=True, help="BLAZE_DRAFT_DUMP directories (one per run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=65536)
    ap.add_argument("--no-gptq", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; choose a new directory")
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.load(open(os.path.join(a.draft, "config.json")))
    names = [n for group in LAYERS.values() for n in group]
    w = {}
    with safe_open(os.path.join(a.draft, "model.safetensors"), "pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            if k.startswith(("fc", "layers.0.", "norm")):
                w[k] = f.get_tensor(k).to(dev)
    idx = json.load(open(os.path.join(a.target, "model.safetensors.index.json")))["weight_map"]
    ename = next(k for k in idx if k.endswith("embed_tokens.weight") and "mtp" not in k)
    with safe_open(os.path.join(a.target, idx[ename]), "pt", device="cpu") as f:
        embed = f.get_tensor(ename).to(dev)
    cal = Calibrator(cfg, w, embed, dev)
    h = cfg["hidden_size"]
    for run in a.calib:
        cal.reset()
        calls = sorted(glob.glob(os.path.join(run, "[0-9]" * 5))) or [run]
        for c in calls:
            meta = json.load(open(os.path.join(c, "meta.json")))
            tokens = torch.tensor(meta["tokens"], device=dev)
            n = len(tokens)
            feats = torch.frombuffer(bytearray(open(os.path.join(c, "features.bin"), "rb").read()), dtype=torch.bfloat16)
            feats = feats.view(-1, 3, h)[:n].to(dev)
            pos = torch.arange(meta["pos0"], meta["pos0"] + n, device=dev)
            with torch.no_grad():
                cal.call(feats, tokens, pos)
            if cal.rows >= a.max_rows:
                break
        print(f"calibration: {run}: {len(calls)} calls, {cal.rows} rows so far", flush=True)
        if cal.rows >= a.max_rows:
            break
    del embed
    torch.cuda.empty_cache()

    out = {}
    report = []
    for group, mods in LAYERS.items():
        H = cal.H[group] / cal.rows
        hdiag = torch.diag(H)
        for m in mods:
            W = w[m + ".weight"].float()
            e, lowered = choose_scales(W, hdiag)
            rtn = qdq(W, e8m0_rtn(W.abs().view(W.shape[0], -1, 32).amax(-1)))
            searched = qdq(W, e)
            wq = searched if a.no_gptq else gptq(W, H, e)
            codes, scales = encode(wq, e)
            out[m + ".weight"] = codes.cpu()
            out[m + ".weight_scale"] = scales.contiguous().cpu()
            r = (m, rel_err(W, rtn, H), rel_err(W, searched, H), rel_err(W, wq, H), lowered)
            report.append(r)
            print(f"{m}: output error rtn {r[1]:.3e}, scale search {r[2]:.3e} ({100 * lowered:.1f}% blocks lowered), "
                  f"{'search' if a.no_gptq else 'gptq'} {r[3]:.3e}", flush=True)
            del W, rtn, searched, wq
            torch.cuda.empty_cache()

    os.makedirs(a.out)
    with safe_open(os.path.join(a.draft, "model.safetensors"), "pt", device="cpu") as f:
        for k in f.keys():
            if not any(k == m + ".weight" for m in names):
                out[k] = f.get_tensor(k)
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    shutil.copy(os.path.join(a.draft, "config.json"), os.path.join(a.out, "config.json"))
    excluded = sorted({k[: -len(".weight")] for k in out if k.endswith(".weight") and k[: -len(".weight")] not in names})
    json.dump(
        {
            "producer": {"name": "blaze scripts/quantize_eagle4.py"},
            "quantization": {"quant_algo": "MXFP8", "kv_cache_quant_algo": None, "exclude_modules": excluded},
        },
        open(os.path.join(a.out, "hf_quant_config.json"), "w"),
        indent=2,
    )
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(f"MXFP8 (E4M3, E8M0 per 32) of {os.path.basename(os.path.normpath(a.draft))}, offline: "
                f"{'scale search' if a.no_gptq else 'scale search + GPTQ'} on {cal.rows} rows of target features.\n\n")
        f.write("| layer | rtn | scale search | final |\n|---|---|---|---|\n")
        for m, r1, r2, r3, _ in report:
            f.write(f"| {m} | {r1:.3e} | {r2:.3e} | {r3:.3e} |\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
