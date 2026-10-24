#!/usr/bin/env python3
"""Reference for blaze's MTP drafter: the Hugging Face Qwen3.5 decoder layer in fp32 on the CPU, with the MTP head's
wiring of vLLM (qwen3_5_mtp) and SGLang (Qwen3_5ForCausalLMMTP):

    x = fc(concat(norm_e(embed(token)), norm_h(hidden)))      zero-centered norms, the embedding first
    x = layer(x)                                              one full-attention decoder layer, causal over the rows
    out = mtp.norm(x);  logits = lm_head(out)

Input: the directory written by `BLAZE_MTP_DUMP=dir blaze eagle-test --mtp-window N ...` (the first drafting call of one
sequence: the tokens after each extended row, the position of the first row, the target's final normed hidden state of
every row, blaze's drafts and its normed output after the extend and each chain step). The reference extends over the same
rows and follows blaze's draft tokens (teacher forcing, so that every step compares like with like); it also reports the
drafts of its own free-running chain. Prints cosine similarity, max abs difference and relative L2 of the normed output,
and the agreement of the logits (cosine, max abs, argmax) computed with the shared LM head from both outputs.

  python scripts/ref_mtp.py MODEL_DIR DUMP_DIR       (a venv with torch, transformers >= 5, safetensors; CPU only)
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding


class Reader:
    """Tensors of a (sharded) checkpoint by name."""

    def __init__(self, d):
        self.d = d
        index = os.path.join(d, "model.safetensors.index.json")
        self.map = json.load(open(index))["weight_map"] if os.path.exists(index) else None

    def file(self, name):
        return os.path.join(self.d, self.map[name] if self.map else "model.safetensors")

    def get(self, name):
        with safe_open(self.file(name), "pt") as f:
            return f.get_tensor(name)

    def rows(self, name, ids):
        with safe_open(self.file(name), "pt") as f:
            sl = f.get_slice(name)
            return torch.cat([sl[i:i + 1] for i in ids], 0)


def rd_bf16(path, cols):
    raw = torch.frombuffer(bytearray(open(path, "rb").read()), dtype=torch.bfloat16)
    return raw.reshape(-1, cols).float()


def main():
    model_dir, dump = sys.argv[1], sys.argv[2]
    meta = json.load(open(os.path.join(dump, "meta.json")))
    tokens, pos0, drafts, h = meta["tokens"], meta["pos0"], meta["drafts"], meta["hidden"]
    rows = len(tokens)
    hidden = rd_bf16(os.path.join(dump, "hidden.bin"), h)[:rows]
    normed = [rd_bf16(os.path.join(dump, f"normed_{j}.bin"), h)[0] for j in range(len(drafts))]
    print(f"{rows} rows from position {pos0}, blaze drafts {drafts}")

    cfg = Qwen3_5TextConfig(**{k: v for k, v in AutoConfig.from_pretrained(model_dir).text_config.to_dict().items()
                               if k not in ("model_type",)})
    cfg.layer_types = ["full_attention"]
    cfg.num_hidden_layers = 1
    cfg._attn_implementation = "eager"
    rd = Reader(model_dir)
    layer = Qwen3_5DecoderLayer(cfg, 0).float().eval()
    sd = {k[len("mtp.layers.0."):]: rd.get(k).float() for k in rd.map if k.startswith("mtp.layers.0.")}
    print("layer tensors:", sorted(sd))
    layer.load_state_dict(sd, strict=True)
    norm_e, norm_h, norm = (Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps).float() for _ in range(3))
    norm_e.weight.data = rd.get("mtp.pre_fc_norm_embedding.weight").float()
    norm_h.weight.data = rd.get("mtp.pre_fc_norm_hidden.weight").float()
    norm.weight.data = rd.get("mtp.norm.weight").float()
    fc = rd.get("mtp.fc.weight").float()
    rope = Qwen3_5TextRotaryEmbedding(cfg)
    lm_head = rd.get("lm_head.weight")  # bf16 [vocab, h]

    def embed(ids):
        return rd.rows("model.language_model.embed_tokens.weight", ids).float()

    @torch.no_grad()
    def mtp(ids, hid, positions):
        """The MTP head over a whole sequence of rows (causal): the normed output of every row."""
        x = torch.cat([norm_e(embed(ids)), norm_h(hid)], -1) @ fc.T
        pos = torch.tensor(positions)[None, None, :].expand(3, 1, -1)
        cos, sin = rope(x[None], pos)
        t = len(ids)
        mask = torch.full((t, t), float("-inf")).triu(1)[None, None]
        y = layer(x[None], position_embeddings=(cos, sin), attention_mask=mask)[0]
        return norm(y)

    @torch.no_grad()
    def logits(v):
        out = torch.empty(lm_head.shape[0])
        for i in range(0, lm_head.shape[0], 32768):
            out[i:i + 32768] = lm_head[i:i + 32768].float() @ v
        return out

    def stats(a, b):
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        return cos, (a - b).abs().max().item(), ((a - b).norm() / b.norm()).item()

    # extend over the dump's rows, then follow blaze's drafts
    ids, hid, positions = list(tokens), hidden.clone(), list(range(pos0, pos0 + rows))
    ok = 0
    free_ids, free_hid, free_pos = list(tokens), hidden.clone(), list(positions)
    for j in range(len(drafts)):
        out = mtp(ids, hid, positions)[-1]
        c, mx, rl = stats(normed[j], out)
        lb, lr = logits(normed[j]), logits(out)
        lc, lmx, _ = stats(lb, lr)
        am_b, am_r = int(lb.argmax()), int(lr.argmax())
        agree = am_b == am_r
        ok += agree
        print(f"step {j}: normed output cos {c:.5f}, max abs {mx:.4f}, rel L2 {rl:.4f} | logits (head applied to each) cos "
              f"{lc:.5f}, max abs {lmx:.3f} | argmax blaze-out {am_b}, ref-out {am_r}, blaze draft {drafts[j]}"
              f" -> {'agree' if agree and drafts[j] == am_r else 'DIFFER'}")
        # teacher forcing: the next row takes blaze's draft token and this step's reference output
        ids.append(drafts[j])
        hid = torch.cat([hid, out[None]], 0)
        positions.append(positions[-1] + 1)
        # free-running reference chain
        fout = mtp(free_ids, free_hid, free_pos)[-1]
        ftok = int(logits(fout).argmax())
        print(f"        free-running reference draft {j}: {ftok}")
        free_ids.append(ftok)
        free_hid = torch.cat([free_hid, fout[None]], 0)
        free_pos.append(free_pos[-1] + 1)
    print(f"argmax agreement {ok}/{len(drafts)}")


if __name__ == "__main__":
    main()
