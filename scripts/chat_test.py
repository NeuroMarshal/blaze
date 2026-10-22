# External test client (not part of the engine).
# External test client for blaze serve: tool call, streaming, multi-turn prompt cache, concurrency.
import json, sys, time, threading, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8100"

def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                         "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Paris in celsius? Use the tool."}]
r = post("/v1/chat/completions", {"model": "x", "messages": msgs, "tools": tools, "max_tokens": 600})
m = r["choices"][0]["message"]
print("TOOL finish:", r["choices"][0]["finish_reason"], "| calls:", json.dumps(m.get("tool_calls")), "| content:", repr(m.get("content"))[:80])
print("usage:", r["usage"])
# second turn: tool result appended -> prompt cache should hit the first prompt
msgs2 = msgs + [{"role": "assistant", "content": m.get("content") or "", "tool_calls": m.get("tool_calls")},
                {"role": "tool", "tool_call_id": (m.get("tool_calls") or [{}])[0].get("id", "1"), "content": "{\"temp\": 18, \"sky\": \"cloudy\"}"}]
t = time.time()
r2 = post("/v1/chat/completions", {"model": "x", "messages": msgs2, "tools": tools, "max_tokens": 400})
print("TURN2:", repr(r2["choices"][0]["message"].get("content"))[:200], "| usage:", r2["usage"], f"| {time.time()-t:.1f}s")

# streaming
req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps({"model": "x", "stream": True, "max_tokens": 80,
      "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": "Say hello in 3 languages."}],
      "chat_template_kwargs": {"enable_thinking": False}}).encode(), {"Content-Type": "application/json"})
chunks, text = 0, ""
for line in urllib.request.urlopen(req, timeout=600):
    line = line.decode().strip()
    if not line.startswith("data:"): continue
    d = line[5:].strip()
    if d == "[DONE]": break
    j = json.loads(d); chunks += 1
    for c in j.get("choices", []):
        text += c["delta"].get("content") or ""
    if j.get("usage"): print("stream usage:", j["usage"])
print(f"STREAM {chunks} chunks:", repr(text)[:200])

# concurrency: 4 requests at once
res = {}
def one(i):
    t = time.time()
    r = post("/v1/chat/completions", {"model": "x", "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False},
                                       "messages": [{"role": "user", "content": f"Write a haiku about the number {i}."}]})
    res[i] = (r["usage"]["completion_tokens"], time.time() - t)
th = [threading.Thread(target=one, args=(i,)) for i in range(4)]
t0 = time.time(); [x.start() for x in th]; [x.join() for x in th]
tot = sum(v[0] for v in res.values())
print(f"CONCURRENT 4: {tot} tokens in {time.time()-t0:.1f}s = {tot/(time.time()-t0):.1f} tok/s aggregate", res)
