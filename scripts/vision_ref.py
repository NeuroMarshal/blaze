#!/usr/bin/env python3
"""Reference for blaze's vision encoder: the Hugging Face Qwen3.5 image processor and vision tower on the CPU.

For every image it writes <out>/<name>/ref.safetensors (a directory per image: blaze reads checkpoints by directory) with
  pixel_values  f32 [patches, 1536]  (the processor's output: resized, normalized, patchified, temporal pair)
  grid_thw      i64 [1, 3]
  pos_embed     f32 [patches, hidden]   patch embedding + learned position embedding (bf16 model)
  block0        f32 [patches, hidden]   after the first block
  last_hidden   f32 [patches, hidden]   after the last block
  embeds        f32 [patches / 4, out_hidden]  the merged embeddings the language model receives (bf16 model)
  embeds_f32    f32 [patches / 4, out_hidden]  the same with the weights upcast to fp32 (the noise floor of bf16)
blaze's `vision-check` command reads these files and compares its encoder against them.

  python scripts/vision_ref.py MODEL_DIR OUT_DIR IMAGE...      (a throwaway venv with torch, transformers, pillow,
                                                               torchvision, safetensors; CPU only)
"""
import os
import sys
import time

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


def load_vision(model_dir, dtype):
    cfg = AutoConfig.from_pretrained(model_dir).vision_config
    model = Qwen3_5VisionModel(cfg).to(dtype).eval()
    prefix = "model.visual."
    sd = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = f.get_tensor(k).to(dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return model


@torch.no_grad()
def run(model, pixel_values, grid_thw, keep):
    """The vision tower's forward with the intermediate states a debugging comparison needs."""
    hooks = []
    blocks = model.blocks
    if keep is not None:
        hooks.append(blocks[0].register_forward_hook(lambda m, i, o: keep.__setitem__("block0", o.float().clone())))
        hooks.append(blocks[-1].register_forward_hook(lambda m, i, o: keep.__setitem__("last_hidden", o.float().clone())))
        # patch embedding + position embedding = the input of block 0
        hooks.append(blocks[0].register_forward_pre_hook(
            lambda m, args, kwargs: keep.__setitem__("pos_embed", (args[0] if args else kwargs["hidden_states"]).float().clone()),
            with_kwargs=True))
    out = model(pixel_values.to(next(model.parameters()).dtype), grid_thw=grid_thw)
    for h in hooks:
        h.remove()
    return out.pooler_output


def main():
    model_dir, out_dir, images = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    proc = AutoImageProcessor.from_pretrained(model_dir)
    bf = load_vision(model_dir, torch.bfloat16)
    f32 = load_vision(model_dir, torch.float32) if os.environ.get("REF_FP32", "1") == "1" else None
    torch.set_num_threads(os.cpu_count())
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        batch = proc(images=[img], return_tensors="pt")
        pv, grid = batch["pixel_values"], batch["image_grid_thw"]
        print(f"{name}: {img.size[0]}x{img.size[1]} -> grid {grid.tolist()} pixel_values {tuple(pv.shape)}", flush=True)
        keep = {}
        t0 = time.time()
        emb = run(bf, pv, grid, keep)
        print(f"  bf16 tower {time.time() - t0:.1f} s, embeds {tuple(emb.shape)}", flush=True)
        tensors = {"pixel_values": pv.float().contiguous(), "grid_thw": grid.contiguous(), "embeds": emb.float().contiguous(),
                   **{k: v.contiguous() for k, v in keep.items()}}
        if f32 is not None:
            t0 = time.time()
            emb32 = run(f32, pv, grid, None)
            print(f"  fp32 tower {time.time() - t0:.1f} s", flush=True)
            tensors["embeds_f32"] = emb32.float().contiguous()
            cos = torch.nn.functional.cosine_similarity(emb.float(), emb32.float(), dim=-1)
            print(f"  bf16 vs fp32 reference: cos mean {cos.mean():.5f} min {cos.min():.5f}, max abs diff "
                  f"{(emb.float() - emb32.float()).abs().max():.4f}", flush=True)
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        save_file(tensors, os.path.join(out_dir, name, "ref.safetensors"))


if __name__ == "__main__":
    main()
