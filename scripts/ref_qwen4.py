# External test reference (not part of the engine): Qwen4-Exp (Qwen3.8 Flash Next) forward in plain torch, written
# from SGLang's qwen4_exp.py / hyperconnection.py / qwen3_5.py formulas, weights read straight from the checkpoint
# (FP8 per-channel and NVFP4 dequantized on the fly, PLE rows from the n-gram shards). Dense attention (equal to QSA
# while the context is shorter than the indexer budget). Writes every block's tensors as bf16 .bin files named like
# blaze's BLAZE_DUMP, and prints the top-5 next tokens.
#   python ref_qwen4.py MODEL_DIR OUT_DIR "prompt"
import json
import math
import os
import sys

import torch
from tokenizers import Tokenizer

M, OUT, PROMPT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
cfg = json.load(open(f"{M}/config.json"))["text_config"]
index = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
_files = {}
P = "model.language_model."
_DT = {"BF16": (torch.bfloat16, 2), "F32": (torch.float32, 4), "F8_E4M3": (torch.float8_e4m3fn, 1), "U8": (torch.uint8, 1),
       "I64": (torch.int64, 8), "F16": (torch.float16, 2)}


class _File:
    """safetensors file read with pread (a 100 GB shard cannot be mapped under the server's overcommit policy)."""

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        n = int.from_bytes(os.pread(self.fd, 8, 0), "little")
        self.header = json.loads(os.pread(self.fd, n, 8))
        self.base = 8 + n

    def shape(self, name):
        return self.header[name]["shape"]

    def rows(self, name, a, b):
        h = self.header[name]
        dt, size = _DT[h["dtype"]]
        row = size * math.prod(h["shape"][1:])
        raw = os.pread(self.fd, row * (b - a), self.base + h["data_offsets"][0] + row * a)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dt).reshape(b - a, *h["shape"][1:])

    def get(self, name):
        h = self.header[name]
        shape = h["shape"]
        if not shape:
            dt, size = _DT[h["dtype"]]
            raw = os.pread(self.fd, size, self.base + h["data_offsets"][0])
            return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dt).reshape(())
        return self.rows(name, 0, shape[0])


def handle(name):
    f = index[name]
    if f not in _files:
        _files[f] = _File(f"{M}/{f}")
    return _files[f]


def T(name):
    return handle(name).get(name).to(dev)


def has(name):
    return name in index


def rb(x):
    return x.to(torch.bfloat16).float()


# Optional: SGL_PASS=<SGLang --debug-tensor-dump Pass*.pt of the same prompt>; every chk() point is compared with it.
_sgl = torch.load(os.environ["SGL_PASS"], weights_only=False) if os.environ.get("SGL_PASS") else None
MAX_LAYERS = int(os.environ.get("MAX_LAYERS", "1000"))


def chk(name, x):
    if _sgl is None or name not in _sgl:
        return
    r = _sgl[name]
    r = (r[0] if isinstance(r, list) else r).float().to(dev).reshape(-1)
    y = x.float().reshape(-1)
    if r.numel() != y.numel():
        print(f"  {name:55s} size {y.numel()} vs sglang {r.numel()}")
        return
    rel = float((y - r).norm() / r.norm().clamp_min(1e-30))
    print(f"  {name:55s} rel {rel:.4f}{'   <--' if rel > 0.03 else ''}", flush=True)


def save(name, x):
    x.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().tofile(f"{OUT}/{name}.bin")


E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=dev)


def weight(mod):
    """Dequantized FP32 weight [N, K] of a linear module."""
    if has(mod + ".weight_packed"):
        packed = T(mod + ".weight_packed")
        n, k2 = packed.shape
        codes = torch.stack([packed & 15, packed >> 4], -1).reshape(n, 2 * k2).long()
        sc = T(mod + ".weight_scale").float().repeat_interleave(16, dim=1)
        g = T(mod + ".weight_global_scale").float()
        return E2M1[codes] * sc / g
    w = T(mod + ".weight")
    if w.dtype == torch.float8_e4m3fn:
        return w.float() * T(mod + ".weight_scale").float()
    return w.float()


_wcache = {}


def lin(x, mod):
    if mod not in _wcache:
        _wcache[mod] = weight(mod)
    return rb(x @ _wcache[mod].t())


def vec(name):
    return T(name).float()


eps = cfg["rms_norm_eps"]
H, HC = cfg["hidden_size"], cfg["hc_count"]
silu = torch.nn.functional.silu


def grouped_norm(x, w, groups):
    t = x.shape[0]
    g = x.view(t, groups, -1)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)
    return rb(g.reshape(t, -1) * (1.0 + w))


def hc_mix(streams, pre):
    n = grouped_norm(streams, vec(pre + "hc_norm.weight"), HC)
    chk(pre.replace(P, "model.") + "hc_norm", n)
    tt = rb(silu((n @ vec(pre + "input_mix_weight_down.weight").t()) / HC))
    u = tt @ vec(pre + "input_mix_weight_up.weight").t()
    out = (torch.sigmoid(u) * n).view(n.shape[0], HC, H).mean(1)
    return rb(out), n


def hc_combine(streams, n, pre, y):
    a = 2 * torch.sigmoid((n @ vec(pre + "block_inject_weight.weight").t()) / HC)
    return rb(streams.view(-1, HC, H) + a[..., None] * y[:, None, :]).reshape(streams.shape[0], -1)


def gdn(x, p):
    kh, vh, dk, dv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    t = x.shape[0]
    qkv = lin(x, p + "in_proj_qkv")
    z = lin(x, p + "in_proj_z")
    b = lin(x, p + "in_proj_b")
    a = lin(x, p + "in_proj_a")
    L = p.replace(P, "model.").replace("linear_attn.", "linear_attn")
    chk(L + ".in_proj_qkvz", torch.cat([qkv, z], -1))
    chk(L + ".in_proj_ba", torch.cat([b, a], -1))
    cw = vec(p + "conv1d.weight")[:, 0, :]
    pad = torch.cat([torch.zeros(3, qkv.shape[1], device=dev), qkv])
    conv = sum(cw[:, j] * pad[j:j + t] for j in range(4))
    conv = rb(silu(rb(conv)))
    q = conv[:, :kh * dk].view(t, kh, dk).repeat_interleave(vh // kh, dim=1)
    k = conv[:, kh * dk:2 * kh * dk].view(t, kh, dk).repeat_interleave(vh // kh, dim=1)
    v = conv[:, 2 * kh * dk:].view(t, vh, dv)
    beta = rb(torch.sigmoid(b))
    g = -torch.exp(vec(p + "A_log")) * torch.nn.functional.softplus(a + vec(p + "dt_bias"))
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    S = torch.zeros(vh, dk, dv, device=dev)
    outs = []
    for i in range(t):
        S = S * torch.exp(g[i])[:, None, None]
        kv = torch.einsum("hkv,hk->hv", S, k[i])
        delta = (v[i] - kv) * beta[i][:, None]
        S = S + k[i][:, :, None] * delta[:, None, :]
        outs.append(torch.einsum("hkv,hk->hv", S, q[i]))
    o = rb(torch.stack(outs))
    chk(L + ".attn", o)
    nw = vec(p + "norm.weight")
    # SGLang's RMSNormGated (norm before gate), activation = config output_gate_type, FP32 inside
    zz = z.view(t, vh, dv)
    act = torch.sigmoid(zz) if cfg.get("output_gate_type") == "sigmoid" else silu(zz)
    o = rb(o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * nw * act)
    chk(L + ".norm", o)
    out = lin(o.reshape(t, -1), p + "out_proj")
    chk(L + ".out_proj", out)
    return out


def attention(x, p, pos):
    nh, kvh, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    t = x.shape[0]
    qg = lin(x, p + "q_proj").view(t, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:]
    k = lin(x, p + "k_proj").view(t, kvh, hd)
    v = lin(x, p + "v_proj").view(t, kvh, hd)
    L = p.replace(P, "model.").replace("self_attn.", "")
    chk(L + "qkv_proj", torch.cat([qg.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1)], -1))

    def hnorm(y, w):
        return rb(y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w))

    q, k = hnorm(q, vec(p + "q_norm.weight")), hnorm(k, vec(p + "k_norm.weight"))
    rd = int(hd * cfg["partial_rotary_factor"])
    theta = cfg["rope_parameters"]["rope_theta"]
    inv = 1.0 / (theta ** (torch.arange(0, rd, 2, device=dev).float() / rd))
    ang = pos[:, None].float() * inv[None]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]

    def rope(y):
        y1, y2 = y[..., :rd // 2], y[..., rd // 2:rd]
        return torch.cat([rb(y1 * cos - y2 * sin), rb(y2 * cos + y1 * sin), y[..., rd:]], -1)

    q, k = rope(q), rope(k)
    k = k.repeat_interleave(nh // kvh, dim=1)
    v = v.repeat_interleave(nh // kvh, dim=1)
    s = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd)
    s = s.masked_fill(torch.triu(torch.ones(t, t, device=dev, dtype=torch.bool), 1), float("-inf"))
    o = rb(torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v))
    chk(L + "attn", o)
    o = rb(o * torch.sigmoid(gate))
    out = lin(o.reshape(t, -1), p + "o_proj")
    chk(L + "o_proj", out)
    return out


def moe(x, p):
    t = x.shape[0]
    logits = lin(x, p + "gate")
    L = p.replace(P, "model.")
    chk(L + "gate", logits)
    prob = torch.softmax(logits, -1)
    tw, ti = prob.topk(cfg["num_experts_per_tok"], -1)
    if cfg.get("norm_topk_prob", True):
        tw = tw / tw.sum(-1, keepdim=True)
    out = torch.zeros(t, H, device=dev)
    for i in range(t):
        for j in range(ti.shape[1]):
            e = p + f"experts.{int(ti[i, j])}."
            hh = rb(silu(lin(x[i:i + 1], e + "gate_proj")) * lin(x[i:i + 1], e + "up_proj"))
            out[i] += tw[i, j] * lin(hh, e + "down_proj")[0]
    out = rb(out)
    if _sgl is not None and L + "topk" in _sgl:
        sw, si = _sgl[L + "topk"][0].to(dev), _sgl[L + "topk"][1].to(dev)
        same = (si.sort(-1).values == ti.sort(-1).values).float().mean()
        print(f"  {L + 'topk':55s} ids agree {float(same):.3f}, weights {[round(float(v), 3) for v in sw[0][:4]]} vs {[round(float(v), 3) for v in tw[0][:4]]}")
    chk(L + "experts", out)
    sh = rb(silu(lin(x, p + "shared_expert.gate_proj")) * lin(x, p + "shared_expert.up_proj"))
    sh = lin(sh, p + "shared_expert.down_proj")
    chk(L + "shared_expert.down_proj", sh)
    sg = torch.sigmoid(lin(x, p + "shared_expert_gate"))
    return rb(out + sg * sh)


def ple(streams, tokens, p):
    t = streams.shape[0]
    eos = cfg["eos_token_id"]
    e = p + "ple.ple_embedding."
    mult = T(e + "layer_multipliers").tolist()
    sizes = T(e + "ngram_heads_vocab_sizes").tolist()
    offs = T(e + "ngram_heads_offsets").tolist()
    hpn = cfg["heads_per_ngram"]
    win = [eos, eos] + tokens
    mask = (1 << 64) - 1

    def s64(v):
        v &= mask
        return v - (1 << 64) if v >= 1 << 63 else v

    ids = []
    for i in range(t):
        w0, w1, w2 = win[i], win[i + 1], win[i + 2]
        reach = 0 if w1 == eos else (1 if w0 == eos else 2)
        t1 = w1 if reach >= 1 else eos
        t2 = w0 if reach >= 2 else eos
        m2 = s64((w2 * mult[0]) ^ (t1 * mult[1]))
        m3 = s64(m2 ^ (t2 * mult[2]))
        for h in range(2 * hpn):
            ids.append(((m2 if h < hpn else m3) % sizes[h]) + offs[h])
    shard_name = e + "ngram_embedding.shard_0.weight"
    shard_rows = handle(shard_name).shape(shard_name)[0]
    rows = []
    for r in ids:
        name = e + f"ngram_embedding.shard_{r // shard_rows}.weight"
        rows.append(handle(name).rows(name, r % shard_rows, r % shard_rows + 1))
    emb = torch.cat(rows).to(dev).float().reshape(t, -1)
    key = lin(emb, p + "ple.key_proj")
    value = lin(emb, p + "ple.value_proj")
    L = p.replace(P, "model.") + "ple."
    chk(L + "key_proj", key)
    chk(L + "value_proj", value)
    kn = grouped_norm(key, vec(p + "ple.norm_key.weight"), HC).view(t, HC, H)
    qn = grouped_norm(streams, vec(p + "ple.norm_query.weight"), HC).view(t, HC, H)
    chk(L + "norm_key", kn)
    chk(L + "norm_query", qn)
    gte = (kn * qn).sum(-1) / math.sqrt(H)
    gte = torch.sigmoid(gte.abs().clamp_min(1e-6).sqrt() * gte.sign())
    gated = rb(gte[..., None] * value[:, None, :]).reshape(t, -1)
    gn = grouped_norm(gated, vec(p + "ple.norm_conv.weight"), HC)
    chk(L + "norm_conv", gn)
    cw = vec(p + "ple.conv1d.weight")[:, 0, :]
    dil = cfg["ngram_size"]
    pad = torch.cat([torch.zeros(3 * dil, gn.shape[1], device=dev), gn])
    conv = sum(cw[:, j] * pad[j * dil:j * dil + t] for j in range(4))
    out = rb(gated + rb(silu(rb(conv))))
    return rb(streams + out)


tok = Tokenizer.from_file(f"{M}/tokenizer.json")
tokens = tok.encode(PROMPT, add_special_tokens=False).ids
print("tokens", tokens)
t = len(tokens)
pos = torch.arange(t, device=dev)
emb_name = P + "embed_tokens.weight"
emb = torch.cat([handle(emb_name).rows(emb_name, i, i + 1) for i in tokens]).to(dev).float()
chk("model.embed_tokens", emb)
streams = emb.repeat(1, HC)
ple_layers = {i - 1 for i in cfg["ple_layer_ids"]}
for li, kind in enumerate(cfg["layer_types"]):
    if li >= MAX_LAYERS:
        break
    p = P + f"layers.{li}."
    if li in ple_layers:
        streams = ple(streams, tokens, p)
        save(f"l{li}_ple", streams)
    x, n = hc_mix(streams, p + "attn_hyper_connection.")
    save(f"l{li}_attn_in", x)
    y = gdn(x, p + "linear_attn.") if kind == "linear_attention" else attention(x, p + "self_attn.", pos)
    save(f"l{li}_attn_out", y)
    streams = hc_combine(streams, n, p + "attn_hyper_connection.", y)
    x, n = hc_mix(streams, p + "mlp_hyper_connection.")
    y = moe(x, p + "mlp.")
    save(f"l{li}_mlp_out", y)
    streams = hc_combine(streams, n, p + "mlp_hyper_connection.", y)
    save(f"l{li}_streams", streams)
    _wcache.clear()
    print(f"layer {li} ({kind}) done", flush=True)
x, _ = hc_mix(streams, P + "hyper_connection_mixer.")
logits = x[-1:] @ T("lm_head.weight").float().t()
top = logits[0].topk(5)
print("top-5:", [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
