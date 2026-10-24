"""Soak test of a running OpenAI-compatible server: N clients for a while with mixed traffic (short and long prompts,
streams and plain answers, cancels mid-stream, tool calls, JSON schemas, repeated conversations for the prefix cache).
Prints a summary every minute (requests, errors by kind, tokens/s, latency percentiles) and checks /health at the end.

    python3 scripts/soak_test.py http://127.0.0.1:8100 [--clients 6] [--minutes 30] [--long-tokens 30000]

Exit status 1 when any request failed in a way a correct server never fails (5xx, broken streams, invalid JSON from a
schema request, a tool call that does not parse) or the server stopped answering.
"""

import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

WORDS = ("alpha beta gamma delta engine kernel tensor cache paging route expert window token stream batch graph "
         "verify draft commit prefix sparse budget layer vector matrix").split()
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}, "score": {"type": "integer"}},
          "required": ["title", "score"], "additionalProperties": False}


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.ok = 0
        self.tokens = 0
        self.errors = {}
        self.fatal = 0
        self.lat = []

    def done(self, tokens, seconds):
        with self.lock:
            self.ok += 1
            self.tokens += tokens
            self.lat.append(seconds)

    def error(self, kind, fatal):
        with self.lock:
            self.errors[kind] = self.errors.get(kind, 0) + 1
            self.fatal += fatal


def post(url, body, timeout=1800):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def filler(rng, words):
    return " ".join(rng.choice(WORDS) for _ in range(words))


def one(url, rng, stats, long_tokens, history):
    kind = rng.choices(["short", "stream", "cancel", "tool", "json", "long", "chat"], [5, 5, 2, 2, 2, 1, 3])[0]
    body = {"model": "default", "temperature": rng.choice([0, 0.7]), "max_tokens": rng.choice([32, 128, 400]),
            "chat_template_kwargs": {"enable_thinking": False}}
    if kind == "long":
        body["messages"] = [{"role": "user", "content": filler(rng, int(long_tokens * 0.6)) + "\nSummarize in one line."}]
    elif kind == "chat" and history:
        msgs = rng.choice(history)
        body["messages"] = msgs + [{"role": "user", "content": "And then? " + filler(rng, 8)}]
    else:
        body["messages"] = [{"role": "user", "content": f"Explain {rng.choice(WORDS)} and {rng.choice(WORDS)} briefly. "
                                                        + filler(rng, rng.randint(5, 200))}]
    if kind == "tool":
        body["tools"], body["tool_choice"] = TOOLS, rng.choice(["auto", "required"])
        body["messages"] = [{"role": "user", "content": "What is the weather in Oslo? Use the tool."}]
    if kind == "json":
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "r", "schema": SCHEMA}}
    stream = kind in ("stream", "cancel")
    body["stream"] = stream
    t0 = time.perf_counter()
    try:
        with post(url, body) as r:
            if stream:
                text, n = "", 0
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    ch = json.loads(line[6:])
                    if ch["choices"] and ch["choices"][0]["delta"].get("content"):
                        text += ch["choices"][0]["delta"]["content"]
                        n += 1
                    if kind == "cancel" and n >= rng.randint(3, 30):
                        break  # the client goes away mid-stream
                stats.done(n, time.perf_counter() - t0)
                return
            out = json.loads(r.read())
        msg = out["choices"][0]["message"]
        if kind == "json":
            j = json.loads(msg["content"])
            if not (isinstance(j, dict) and set(j) == {"title", "score"}):
                stats.error("json schema not followed", 1)
                return
        if kind == "tool" and body["tool_choice"] == "required":
            calls = msg.get("tool_calls") or []
            if not calls:
                stats.error("required tool call missing", 1)
                return
            json.loads(calls[0]["function"]["arguments"])
        if kind in ("short", "chat") and len(history) < 50:
            history.append(body["messages"] + [{"role": "assistant", "content": msg.get("content") or ""}])
        stats.done(out["usage"]["completion_tokens"], time.perf_counter() - t0)
    except urllib.error.HTTPError as e:
        stats.error(f"HTTP {e.code}", e.code >= 500)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        stats.error(f"bad answer ({type(e).__name__})", 1)
    except Exception as e:  # connection errors, timeouts
        stats.error(type(e).__name__, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--clients", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--long-tokens", type=int, default=30000)
    args = ap.parse_args()
    stats, history = Stats(), []
    end = time.time() + 60 * args.minutes

    def client(seed):
        rng = random.Random(seed)
        while time.time() < end:
            one(args.url, rng, stats, args.long_tokens, history)

    threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(args.clients)]
    for t in threads:
        t.start()
    t_start, last_tokens = time.time(), 0
    while any(t.is_alive() for t in threads):
        time.sleep(60 if time.time() + 60 < end else max(1, end - time.time()))
        with stats.lock:
            lat = sorted(stats.lat[-500:])
            p50 = statistics.median(lat) if lat else 0
            p95 = lat[int(0.95 * (len(lat) - 1))] if lat else 0
            rate = (stats.tokens - last_tokens) / 60
            last_tokens = stats.tokens
            print(f"[{(time.time() - t_start) / 60:5.1f} min] {stats.ok} done, errors {stats.errors or 'none'}; "
                  f"{rate:.0f} tok/s; latency p50 {p50:.1f} s p95 {p95:.1f} s", flush=True)
    for t in threads:
        t.join(timeout=1900)
    try:
        alive = urllib.request.urlopen(args.url + "/health", timeout=30).status == 200
    except Exception:
        alive = False
    print(f"total: {stats.ok} requests, {stats.tokens} tokens, errors {stats.errors or 'none'}; server "
          f"{'answers' if alive else 'DOES NOT ANSWER'}")
    raise SystemExit(1 if stats.fatal or not alive else 0)


if __name__ == "__main__":
    main()
