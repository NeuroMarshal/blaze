# External test client: a code-editing style request (the answer repeats much of the prompt), timed; greedy output
# saved for comparisons between servers with and without speculative decoding.
import json, sys, time, urllib.request
URL, OUT = sys.argv[1], sys.argv[2]
code = open(__file__).read() * 2
body = {"model": "x", "temperature": 0, "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Rewrite this Python file exactly as it is, but rename the variable `code` to `source`. Output only the code.\n\n" + code}]}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
dt = time.time() - t
txt = r["choices"][0]["message"]["content"]
open(OUT, "w").write(txt)
u = r["usage"]
print(f"{u['completion_tokens']} tokens in {dt:.2f}s ({u['completion_tokens']/dt:.1f} tok/s incl. prefill of {u['prompt_tokens']})")
