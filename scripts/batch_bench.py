# External test client: four concurrent greedy requests (two code rewrites, two essays); aggregate tok/s and the
# outputs (JSON) for comparing runs with and without speculation.
#   python3 scripts/batch_bench.py URL OUT.json
import json, os, sys, time, threading, urllib.request
URL, OUT = sys.argv[1], sys.argv[2]
code = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_test.py")).read()
qs = ["Rewrite this Python file exactly as it is, but rename the variable `msgs` to `messages`. Output only the code.\n\n" + code,
      "Write a detailed essay about the history of the Roman Empire.",
      "Rewrite this Python file exactly as it is, but rename the function `post` to `send`. Output only the code.\n\n" + code,
      "Write a detailed essay about the history of the Ottoman Empire."]
res = {}
def run(i):
    body = {"model": "x", "temperature": 0, "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": qs[i]}]}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    res[i] = (r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"], time.time() - t)
t0 = time.time()
ts = [threading.Thread(target=run, args=(i,)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
dt = time.time() - t0
tok = sum(v[1] for v in res.values())
print(f"4 concurrent: {tok} tokens in {dt:.2f}s = {tok/dt:.1f} tok/s aggregate; per request s: " + " ".join(f"{res[i][2]:.1f}" for i in range(4)))
json.dump({i: res[i][0] for i in range(4)}, open(OUT, "w"))
