"""Prefix cache under an agent's traffic: a conversation with a long system prompt grows turn by turn (its history sent
again each time, as opencode / Hermes do), and between the turns side requests with other prompts arrive (titles,
short tool calls) that compete for the cache's snapshots. Per turn: prompt tokens, tokens served from the cache, TTFT.

    python3 scripts/prefix_cache_test.py http://127.0.0.1:8100 [--turns 5] [--side 2] [--system-words 1500]

A correct cache serves every turn from the previous turn's whole conversation (its prompt and answer) or at least from
its prompt; a turn that finds nothing after the first prints MISS.
"""

import argparse
import json
import random
import time
import urllib.request

WORDS = ("module parser kernel cache tensor stream request batch graph token prefix router layer config schema "
         "handler client server buffer memory thread queue event state").split()


def chat(url, messages, max_tokens):
    body = {"model": "default", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, text, usage = time.perf_counter(), None, "", {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:])
            if ch.get("usage"):
                usage = ch["usage"]
            if ch["choices"] and ch["choices"][0]["delta"].get("content"):
                first = first or time.perf_counter()
                text += ch["choices"][0]["delta"]["content"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return text, usage.get("prompt_tokens", 0), cached, (first or time.perf_counter()) - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--side", type=int, default=2, help="side requests between turns")
    ap.add_argument("--system-words", type=int, default=1500)
    ap.add_argument("--answer-tokens", type=int, default=150)
    args = ap.parse_args()
    rng = random.Random(3)
    system = "You are a coding agent. Project notes: " + " ".join(rng.choice(WORDS) for _ in range(args.system_words))
    msgs = [{"role": "system", "content": system}]
    misses = 0
    for turn in range(args.turns):
        msgs.append({"role": "user", "content": f"Step {turn + 1}: describe how the {rng.choice(WORDS)} talks to the "
                                                f"{rng.choice(WORDS)}, in a short paragraph."})
        text, prompt, cached, ttft = chat(args.url, msgs, args.answer_tokens)
        msgs.append({"role": "assistant", "content": text})
        flag = "" if turn == 0 or cached > 0 else "  MISS"
        misses += bool(flag)
        print(f"turn {turn + 1}: prompt {prompt}, cached {cached} ({100 * cached / max(prompt, 1):.0f}%), TTFT {ttft:.2f} s{flag}",
              flush=True)
        for s in range(args.side):
            side = [{"role": "user", "content": f"Give a five-word title for: {' '.join(rng.choice(WORDS) for _ in range(30))}"}]
            _, sp, sc, st = chat(args.url, side, 12)
            print(f"   side {s + 1}: prompt {sp}, cached {sc}, TTFT {st:.2f} s")
    raise SystemExit(1 if misses else 0)


if __name__ == "__main__":
    main()
