"""KV pool pressure: more long generations at once than a small KV pool holds (start the server with a small
--kv-tokens, e.g. 8192, and --max-seqs 4). Every request must end with an answer (finish "stop" or "length": the
scheduler ends the newest request early rather than failing the step for all), none with an error, and the server
must still answer afterwards.

    python3 scripts/kv_pressure_test.py http://127.0.0.1:8100 [--sessions 4] [--max-tokens 3000]
"""

import argparse
import json
import threading
import urllib.request


def post(url, body, timeout=1800):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=3000)
    args = ap.parse_args()
    out = [None] * args.sessions

    def run(i):
        body = {"model": "default", "messages": [{"role": "user", "content": f"Write a very long story number {i} about a lighthouse keeper, "
                                                  "chapter after chapter, and do not stop."}],
                "max_tokens": args.max_tokens, "temperature": 0.8, "seed": i, "chat_template_kwargs": {"enable_thinking": False}}
        out[i] = post(args.url, body)

    ts = [threading.Thread(target=run, args=(i,)) for i in range(args.sessions)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    bad = 0
    for i, (status, body) in enumerate(out):
        if status != 200:
            bad += 1
            print(f"session {i}: HTTP {status}: {body.get('error', body)}")
            continue
        c = body["choices"][0]
        print(f"session {i}: finish {c['finish_reason']}, {body['usage']['completion_tokens']} tokens")
        bad += c["finish_reason"] not in ("stop", "length")
    status, body = post(args.url, {"model": "default", "messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 8,
                                   "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
    if status != 200:
        bad += 1
        print(f"after the pressure: HTTP {status}")
    print("ok" if bad == 0 else f"FAIL: {bad} bad")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
