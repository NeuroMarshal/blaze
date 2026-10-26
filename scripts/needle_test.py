# External test client (from server-llm-runtime/scripts): long-context retrieval, the same test as the SGLang fork ran.
"""Long-context retrieval: secret codes hidden at several depths of an N-token prompt must all be found.

    python3 scripts/needle_test.py N [--url http://127.0.0.1:8000] [--needles 5]

The filler is varied report sentences (not one repeated phrase), the codes are 6-digit numbers tied to city names.
Prints TTFT, prompt size and which codes the model returned.
"""

import argparse
import json
import os
import random
import re
import time
import urllib.request

CITIES = ["Lisbon", "Osaka", "Tbilisi", "Quito", "Bergen", "Hanoi", "Tallinn", "Accra", "Perth", "Cusco"]
NOUNS = ["turbine", "reservoir", "bridge", "relay", "pipeline", "furnace", "antenna", "rail line", "dam", "depot"]
ADJS = ["northern", "aging", "upgraded", "temporary", "primary", "backup", "coastal", "inland", "eastern", "remote"]
VERBS = ["inspected", "repaired", "measured", "recalibrated", "audited", "photographed", "cleaned", "tested"]
PLACES = ["Glasgow", "Mombasa", "Recife", "Krakow", "Busan", "Nagpur", "Tromso", "Leon", "Izmir", "Darwin"]


def filler(rng: random.Random, words: int) -> list[str]:
    out, n = [], 0
    while n < words:
        s = (f"Report {rng.randint(1000, 9999)}: the {rng.choice(ADJS)} {rng.choice(NOUNS)} near "
             f"{rng.choice(PLACES)} was {rng.choice(VERBS)} on day {rng.randint(1, 365)} and showed "
             f"{rng.randint(10, 990)} units of load, {rng.randint(1, 99)} percent above the previous check.")
        out.append(s)
        n += len(s.split())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens", type=int)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--needles", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(11)
    sentences = filler(rng, int(args.tokens * 0.62))  # ~1.6 tokens per word for this text
    cities = rng.sample(CITIES, args.needles)
    codes = {c: str(rng.randint(100000, 999999)) for c in cities}
    depths = [(i + 0.5) / args.needles for i in range(args.needles)]
    for city, depth in sorted(zip(cities, depths), key=lambda p: -p[1]):
        sentences.insert(int(depth * len(sentences)), f"IMPORTANT: the secret code for {city} is {codes[city]}.")
    question = ("\n\nList the secret code of every city mentioned in the IMPORTANT lines above, one per line, as "
                "'City: code'. Answer with the list only.")
    body = {"model": "default", "max_tokens": 200, "temperature": 0, "presence_penalty": 0.0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": " ".join(sentences) + question}]}
    req = urllib.request.Request(args.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + os.environ["BLAZE_API_KEY"]} if os.environ.get("BLAZE_API_KEY") else {})})
    t0, first, text, usage = time.perf_counter(), None, "", {}
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                first = first or time.perf_counter()
                text += chunk["choices"][0]["delta"]["content"]
    end = time.perf_counter()
    found = {c: bool(re.search(rf"{c}\W+{codes[c]}", text)) for c in cities}
    out = usage.get("completion_tokens", 0)
    print(f"prompt {usage.get('prompt_tokens')} tokens: TTFT {first - t0:.1f} s, decode "
          f"{(out - 1) / max(1e-9, end - first):.1f} tok/s; found {sum(found.values())}/{len(cities)} codes "
          + " ".join(f"{c}@{d:.0%}:{'ok' if found[c] else 'MISS'}" for c, d in zip(cities, depths)))
    if not all(found.values()):
        print("answer:", text.strip()[:400])


if __name__ == "__main__":
    main()
