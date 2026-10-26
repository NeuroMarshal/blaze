# External test client: speculative decoding speed on a code rewrite (drafts hit), an essay (they miss) and the same
# essay again (earlier outputs help suffix decoding); greedy, streamed, chars/s from the first content chunk.
#   python3 scripts/spec_bench.py URL OUT_PREFIX
import json, os, sys, time, urllib.request
URL = sys.argv[1]
code = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_test.py")).read()
cases = {
  "code": "Rewrite this Python file exactly as it is, but rename the variable `msgs` to `messages`. Output only the code.\n\n" + code,
  "essay": "Write a detailed essay about the history of the Roman Empire.",
}
for name, q in list(cases.items()) + [("essay_again", cases["essay"])]:
    body = {"model": "x", "temperature": 0, "max_tokens": 300, "stream": True, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": q}]}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", **({"Authorization": "Bearer " + os.environ["BLAZE_API_KEY"]} if os.environ.get("BLAZE_API_KEY") else {})})
    t0 = time.time(); first = None; n = 0; text = ""
    for line in urllib.request.urlopen(req, timeout=900):
        line = line.decode().strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]": continue
        j = json.loads(line[5:])
        for c in j.get("choices", []):
            d = c["delta"].get("content") or ""
            if d:
                if first is None: first = time.time()
                text += d; n += 1
    dt = time.time() - first
    print(f"{name}: {len(text)} chars, decode {dt:.2f}s, {len(text)/dt:.0f} chars/s"); open(sys.argv[2] + "." + name, "w").write(text)
