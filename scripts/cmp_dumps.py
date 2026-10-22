# External test tool: compare two BLAZE_DUMP-style directories (bf16 .bin per block) block by block, in layer order.
#   python cmp_dumps.py REF_DIR TEST_DIR
import os
import re
import sys

import numpy as np

ref_dir, test_dir = sys.argv[1], sys.argv[2]


def load(path):
    raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32)


def order(name):
    m = re.match(r"l(\d+)_(\w+)\.bin", name)
    stage = ["ple", "attn_in", "attn_out", "mlp_out", "streams"]
    return (int(m.group(1)), stage.index(m.group(2)) if m.group(2) in stage else 9) if m else (999, 0)


for name in sorted(os.listdir(ref_dir), key=order):
    tp = os.path.join(test_dir, name)
    if not os.path.exists(tp):
        continue
    a, b = load(os.path.join(ref_dir, name)), load(tp)
    if a.size != b.size:
        print(f"{name:18s} size {a.size} vs {b.size}")
        continue
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    flag = "" if rel < 0.03 else "   <-- differs"
    print(f"{name:18s} rel {rel:.4f} cos {cos:.5f}{flag}")
