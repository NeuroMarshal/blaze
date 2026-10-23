#!/usr/bin/env python3
"""Reference check of blaze's EAGLE-4 drafter against a plain torch implementation.

blaze run with BLAZE_DRAFT_DUMP=dir writes the first drafting call: the captured target features of the extended rows
(features.bin, [rows, 3, H] bf16), their tokens, the first position, the drafted chain (meta.json) and the extend's
last normed hidden (first_normed.bin). This script recomputes the same call the way eagle_harness/sglang/models/
llama_eagle4.py defines the draft (fc_norm per aux state, fc, one Llama layer over concat(input_ln(embed),
hidden_norm(hidden)) with a sliding window of W keys ending at the query, final norm, low-rank candidate head re-scored
with the full LM head) and compares.

    python ref_eagle4.py --draft ~/models/.../Qwen3.8-27B-EAGLE4 --target ~/models/.../Qwen3.8-27B-...-NVFP4 --dump dir

--embed own uses the draft checkpoint's embed_tokens instead of the target's (sglang always shares the target's).
"""

import argparse
import json
import math
import os

import torch
from safetensors import safe_open


def load(path, names, device):
    out = {}
    with safe_open(path, "pt", device="cpu") as f:
        for n in names:
            out[n] = f.get_tensor(n).to(device)
    return out


def target_embedding(target_dir, device):
    idx = json.load(open(os.path.join(target_dir, "model.safetensors.index.json")))["weight_map"]
    name = next(k for k in idx if k.endswith("embed_tokens.weight") and "mtp" not in k)
    with safe_open(os.path.join(target_dir, idx[name]), "pt", device="cpu") as f:
        return f.get_tensor(name).to(device)


def rmsnorm(x, w, eps):
    # FP32 throughout, one rounding (sglang's fused RMSNorm)
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)


def rope(x, pos, theta, hd):
    # neox: halves rotated; x [rows, heads, hd]
    inv = theta ** (-torch.arange(0, hd // 2, device=x.device, dtype=torch.float64) * 2 / hd)
    ang = pos.double()[:, None] * inv[None, :]
    cos, sin = ang.cos().float()[:, None, :], ang.sin().float()[:, None, :]
    x1, x2 = x[..., : hd // 2].float(), x[..., hd // 2 :].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(torch.bfloat16)


class Draft:
    def __init__(self, draft_dir, embed, device):
        cfg = json.load(open(os.path.join(draft_dir, "config.json")))
        self.cfg = cfg
        self.eps = cfg["rms_norm_eps"]
        self.h, self.hd = cfg["hidden_size"], cfg["head_dim"]
        self.heads, self.kvh = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.window = cfg["sliding_window"] if cfg.get("use_sliding_window") else 0
        self.theta = (cfg.get("rope_parameters") or {}).get("rope_theta") or cfg.get("rope_theta", 1e6)
        self.norm_output = cfg.get("norm_output", False)
        self.nc = cfg["efficient_lm_head_config"]["num_candidates"]
        p = "layers.0."
        names = ["fc.weight", "norm.weight", "lm_head.weight", "efficient_lm_head.weight",
                 "efficient_lm_head_down_proj.weight", "embed_tokens.weight"] + [f"fc_norm.{i}.weight" for i in range(3)] + [
            p + n for n in ("input_layernorm.weight", "hidden_norm.weight", "post_attention_layernorm.weight",
                            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")]
        self.w = load(os.path.join(draft_dir, "model.safetensors"), names, device)
        self.embed = embed if embed is not None else self.w["embed_tokens.weight"]
        self.k_cache, self.v_cache, self.pos_cache = [], [], []

    def features(self, feats):
        # feats [rows, 3, H] bf16 -> fc(cat(fc_norm_i(f_i)))
        parts = [rmsnorm(feats[:, i], self.w[f"fc_norm.{i}.weight"], self.eps) for i in range(3)]
        return (torch.cat(parts, -1).float() @ self.w["fc.weight"].float().t()).to(torch.bfloat16)

    def layer(self, tokens, hidden, pos):
        w, p = self.w, "layers.0."
        emb = self.embed[tokens]
        x = torch.cat([rmsnorm(emb, w[p + "input_layernorm.weight"], self.eps),
                       rmsnorm(hidden, w[p + "hidden_norm.weight"], self.eps)], -1)
        q = (x.float() @ w[p + "self_attn.q_proj.weight"].float().t()).to(torch.bfloat16).view(-1, self.heads, self.hd)
        k = (x.float() @ w[p + "self_attn.k_proj.weight"].float().t()).to(torch.bfloat16).view(-1, self.kvh, self.hd)
        v = (x.float() @ w[p + "self_attn.v_proj.weight"].float().t()).to(torch.bfloat16).view(-1, self.kvh, self.hd)
        q, k = rope(q, pos, self.theta, self.hd), rope(k, pos, self.theta, self.hd)
        self.k_cache.append(k), self.v_cache.append(v), self.pos_cache.append(pos)
        K, V, P = torch.cat(self.k_cache), torch.cat(self.v_cache), torch.cat(self.pos_cache)
        rep = self.heads // self.kvh
        Kr = K.float().repeat_interleave(rep, 1)
        Vr = V.float().repeat_interleave(rep, 1)
        s = torch.einsum("qhd,khd->hqk", q.float(), Kr) / math.sqrt(self.hd)
        vis = P[None, :] <= pos[:, None]
        if self.window:
            vis &= P[None, :] > pos[:, None] - self.window
        s = s.masked_fill(~vis[None], float("-inf"))
        o = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), Vr).to(torch.bfloat16).reshape(len(tokens), -1)
        o = (o.float() @ w[p + "self_attn.o_proj.weight"].float().t()).to(torch.bfloat16)
        resid = (hidden.float() + o.float()).to(torch.bfloat16)
        x2 = rmsnorm(resid, w[p + "post_attention_layernorm.weight"], self.eps)
        g = x2.float() @ w[p + "mlp.gate_proj.weight"].float().t()
        u = x2.float() @ w[p + "mlp.up_proj.weight"].float().t()
        act = (torch.nn.functional.silu(g.to(torch.bfloat16).float()) * u.to(torch.bfloat16).float()).to(torch.bfloat16)
        mlp = (act.float() @ w[p + "mlp.down_proj.weight"].float().t()).to(torch.bfloat16)
        summed = (resid.float() + mlp.float()).to(torch.bfloat16)
        normed = rmsnorm(summed, w["norm.weight"], self.eps)
        return normed, normed if self.norm_output else summed

    def head(self, normed):
        lr = (normed.float() @ self.w["efficient_lm_head_down_proj.weight"].float().t()).to(torch.bfloat16)
        scores = lr.float() @ self.w["efficient_lm_head.weight"].float().t()
        cand = scores.topk(self.nc, -1).indices
        logits = torch.einsum("rd,rcd->rc", normed.float(), self.w["lm_head.weight"][cand].float())
        best = logits.argmax(-1)
        full = normed.float() @ self.w["lm_head.weight"].float().t()
        return cand.gather(1, best[:, None])[:, 0], full.argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--embed", choices=["target", "own"], default="target")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    h = meta["hidden"]
    tokens = torch.tensor(meta["tokens"], device=dev)
    rows = len(tokens)
    feats = torch.frombuffer(bytearray(open(os.path.join(a.dump, "features.bin"), "rb").read()), dtype=torch.bfloat16)
    feats = feats.view(-1, 3, h)[:rows].to(dev)
    first = torch.frombuffer(bytearray(open(os.path.join(a.dump, "first_normed.bin"), "rb").read()), dtype=torch.bfloat16).to(dev)

    embed = None if a.embed == "own" else target_embedding(a.target, dev)
    d = Draft(a.draft, embed, dev)
    if embed is not None:
        own = d.w["embed_tokens.weight"]
        diff = (own.float() - embed.float()).abs()
        print(f"embedding: draft's own vs the target's: max |diff| {diff.max().item():.4g}, "
              f"rows differing {(diff.amax(-1) > 0).sum().item()} of {own.shape[0]}")
    pos = torch.arange(meta["pos0"], meta["pos0"] + rows, device=dev)
    normed, nxt = d.layer(tokens, d.features(feats), pos)
    last = normed[-1]
    cos = torch.nn.functional.cosine_similarity(last.float(), first.float(), 0).item()
    print(f"extend of {rows} rows at {meta['pos0']}: last normed vs blaze: cos {cos:.6f}, "
          f"max |diff| {(last.float() - first.float()).abs().max().item():.4g}")
    drafts, full = [], []
    tok, hid, p = None, nxt[-1:], pos[-1:] + 1
    t, f = d.head(normed[-1:])
    drafts.append(t.item()), full.append(f.item())
    for _ in range(len(meta["drafts"]) - 1):
        n, hid = d.layer(t, hid, p)
        t, f = d.head(n)
        drafts.append(t.item()), full.append(f.item())
        p = p + 1
    print("blaze drafts:     ", meta["drafts"])
    print("reference drafts: ", drafts)
    print("full-head argmax: ", full)
    same = sum(x == y for x, y in zip(drafts, meta["drafts"]))
    print(f"chain agreement: {same}/{len(drafts)} (first mismatch at {next((i for i, (x, y) in enumerate(zip(drafts, meta['drafts'])) if x != y), None)})")


if __name__ == "__main__":
    main()
