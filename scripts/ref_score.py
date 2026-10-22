# External test client (not part of the engine). # Input-token logprobs of a text from a running SGLang server (teacher forcing), same tokenizer as blaze.
import json, sys, urllib.request
from tokenizers import Tokenizer
model, text_file, out = sys.argv[1:4]
ids = Tokenizer.from_file(model + "/tokenizer.json").encode(open(text_file).read(), add_special_tokens=False).ids
rows = []
for start in range(1, len(ids), 256):
    end = min(len(ids), start + 256)
    body = {"input_ids": ids[:end], "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True,
            "logprob_start_len": start, "top_logprobs_num": 1}
    req = urllib.request.Request("http://127.0.0.1:8000/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    meta = json.load(urllib.request.urlopen(req, timeout=600))["meta_info"]
    lp, top = meta["input_token_logprobs"], meta["input_top_logprobs"]
    # entry i scores token lp[i][1]; find which position of ids it is (the first entry may be the unscored start)
    off = next(o for o in range(-2, 3) if sum(lp[i][1] == ids[start + i + o] for i in range(min(20, len(lp))) if 0 <= start + i + o < len(ids)) >= min(15, len(lp) - 2))
    if start == 1:
        print("alignment offset", off, "entries", len(lp))
    for i in range(len(lp)):
        pos = start + i + off
        if lp[i][0] is None or top[i] is None or not (1 <= pos < len(ids)):
            continue
        # logprob of ids[pos] given ids[:pos] -> row pos - 1 in blaze's numbering
        rows.append({"pos": pos - 1, "target": ids[pos], "logprob": lp[i][0], "argmax": top[i][0][1]})
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(ids), "tokens,", len(rows), "rows")
