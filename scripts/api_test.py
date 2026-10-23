#!/usr/bin/env python3
# External test client for the OpenAI-compatible API of `blaze serve` (it works against vLLM / SGLang servers too,
# except for the /metrics checks). Needs the `openai` package (pip install openai).
#
#   scripts/api_test.py --base-url http://127.0.0.1:8101            # everything
#   scripts/api_test.py --base-url http://127.0.0.1:8000 -k tool,stop
#   scripts/api_test.py --list
#
# Sampling is greedy and thinking is switched off (chat_template_kwargs.enable_thinking=false) wherever the test is
# not about reasoning, so that the checks are exact and quick.
import argparse, json, socket, sys, threading, time, traceback, urllib.error, urllib.parse, urllib.request

try:
    import openai
except ImportError:
    sys.exit("api_test.py needs the openai package: pip install openai")

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8100")
ap.add_argument("--api-key", default="none")
ap.add_argument("--model", default=None, help="model id (default: the first of /v1/models)")
ap.add_argument("-k", default="", help="comma-separated substrings: run only the tests whose name has one of them")
ap.add_argument("--skip", default="", help="comma-separated substrings of tests to skip")
ap.add_argument("--list", action="store_true")
ap.add_argument("--expect-prompt-cache", action="store_true", help="the server runs with --prompt-cache-slots: a tool-result turn must hit the cache")
ap.add_argument("--prefill-tokens", type=int, default=0, help="prompt size of the prefill cancellation test (default: 3/4 of the context, at most 12000)")
args = ap.parse_args()

BASE = args.base_url.rstrip("/")
client = openai.OpenAI(base_url=BASE + "/v1", api_key=args.api_key, timeout=300, max_retries=0)
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
MODEL = args.model
MAX_LEN = None


# plumbing

def http(method, path, body=None, headers=None, timeout=300):
    """(status, headers, parsed JSON or text) of one request; HTTP errors are results, not exceptions."""
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        status, rh, raw = r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        status, rh, raw = e.code, e.headers, e.read()
    try:
        return status, rh, json.loads(raw)
    except ValueError:
        return status, rh, raw.decode(errors="replace")


def sse(path, body):
    """Chunks of a streaming response as (list of parsed `data:` JSON objects, saw [DONE])."""
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    out, done = [], False
    with urllib.request.urlopen(req, timeout=300) as r:
        assert "text/event-stream" in r.headers.get("content-type", ""), r.headers.get("content-type")
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                done = True
                break
            out.append(json.loads(d))
    return out, done


def chat(messages, **kw):
    kw.setdefault("temperature", 0)
    kw.setdefault("max_tokens", 200)
    if "extra_body" not in kw and "reasoning" not in kw:
        kw["extra_body"] = NOTHINK
    return client.chat.completions.create(model=MODEL, messages=messages, **kw)


def raw_chat(body):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 200, **NOTHINK, **body}
    return http("POST", "/v1/chat/completions", body)


def metrics():
    status, _, text = http("GET", "/metrics")
    assert status == 200, f"/metrics: {status}"
    return {l.split()[0]: float(l.split()[1]) for l in text.splitlines() if l and not l.startswith("#")}


def wait_for(cond, what, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(0.25)
    raise AssertionError(f"timed out waiting for {what}")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def error_shape(status, body, want_status=400, code=None):
    check(status == want_status, f"status {status}, want {want_status}: {body}")
    check(isinstance(body, dict) and isinstance(body.get("error"), dict), f"not an OpenAI error object: {body}")
    e = body["error"]
    check(set(e) >= {"message", "type", "param", "code"}, f"error keys {sorted(e)}")
    check(isinstance(e["message"], str) and e["message"], "empty error message")
    if code:
        check(e["code"] == code, f"error code {e['code']!r}, want {code!r}")
    return e


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                          "required": ["city"]}}}
TIME = {"type": "function", "function": {"name": "get_time", "description": "Current local time of a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}

TESTS = []


def test(f):
    TESTS.append(f)
    return f


# endpoints

@test
def endpoints_models_version_health():
    ms = client.models.list()
    check(len(ms.data) >= 1, "no models")
    m = ms.data[0]
    check(m.id == MODEL and m.object == "model", m)
    raw = http("GET", "/v1/models")[2]["data"][0]
    check(isinstance(raw["created"], int) and raw["owned_by"], raw)
    check(isinstance(raw.get("max_model_len"), int) and raw["max_model_len"] > 0, f"max_model_len: {raw}")
    one = http("GET", "/v1/models/" + urllib.parse.quote(MODEL, safe="/"))
    check(one[0] == 200 and one[2]["id"] == MODEL and one[2]["max_model_len"] == raw["max_model_len"], one)
    # stable across calls (the id list is a catalogue, not a clock)
    time.sleep(1.1)
    check(http("GET", "/v1/models")[2]["data"][0]["created"] == raw["created"], "created changes between calls")
    error_shape(*http("GET", "/v1/models/no-such-model")[::2], want_status=404, code="model_not_found")
    s, _, v = http("GET", "/version")
    check(s == 200 and isinstance(v.get("version"), str), v)
    check(http("GET", "/health")[0] == 200, "/health")
    check(http("GET", "/ping")[0] == 200, "/ping")


@test
def endpoints_unknown_route_and_method():
    error_shape(*http("GET", "/v1/nope")[::2], want_status=404)
    s, _, b = http("GET", "/v1/chat/completions")
    check(s == 405, f"GET on a POST route: {s}")


@test
def endpoints_cors():
    s, h, _ = http("OPTIONS", "/v1/chat/completions", headers={"Origin": "http://localhost:3000",
                   "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"})
    check(s in (200, 204), f"preflight {s}")
    check(h.get("access-control-allow-origin") in ("*", "http://localhost:3000"), dict(h))
    check("authorization" in h.get("access-control-allow-headers", "").lower() or h.get("access-control-allow-headers") == "*", dict(h))
    s, h, _ = http("GET", "/v1/models", headers={"Origin": "http://localhost:3000"})
    check(h.get("access-control-allow-origin"), "no CORS header on a normal response")


@test
def endpoints_metrics():
    m = metrics()
    for k in ("blaze_requests_total", "blaze_requests_running", "blaze_requests_waiting", "blaze_requests_cancelled_total"):
        check(k in m, f"{k} missing from /metrics")


# chat

@test
def chat_basic_openai_client():
    r = chat([{"role": "user", "content": "Reply with exactly the word: pong"}], user="u-1", max_completion_tokens=20,
             metadata={"a": "b"}, store=False, extra_body={**NOTHINK, "prompt_cache_key": "k"})
    c = r.choices[0]
    check(r.object == "chat.completion" and r.id.startswith("chatcmpl-"), r.id)
    check(c.message.role == "assistant" and "pong" in c.message.content.lower(), c.message.content)
    check(c.finish_reason == "stop", c.finish_reason)
    u = r.usage
    check(u.prompt_tokens > 0 and u.completion_tokens > 0 and u.total_tokens == u.prompt_tokens + u.completion_tokens, u)
    check(r.model, "model")


@test
def chat_model_is_optional_and_unknown_fields_are_ignored():
    s, _, b = http("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say ok"}], "max_tokens": 8,
                   "temperature": 0, "some_future_field": {"x": 1}, **NOTHINK})
    check(s == 200 and b["choices"][0]["message"]["content"], (s, b))


@test
def chat_finish_reason_length():
    r = chat([{"role": "user", "content": "Count from 1 to 100."}], max_tokens=5)
    check(r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 5, (r.choices[0].finish_reason, r.usage))


@test
def chat_stream_shape_and_usage():
    chunks, done = sse("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 40, "stream": True,
                       "stream_options": {"include_usage": True}, **NOTHINK,
                       "messages": [{"role": "user", "content": "Name three colors."}]})
    check(done, "no [DONE]")
    check(all(c["object"] == "chat.completion.chunk" and c["id"] == chunks[0]["id"] for c in chunks), "chunk ids / objects")
    check(chunks[0]["choices"][0]["delta"].get("role") == "assistant", chunks[0])
    check(chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0, chunks[-1])
    fin = [c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
    check(len(fin) == 1 and fin[0]["choices"][0]["finish_reason"] in ("stop", "length"), fin)
    check(chunks.index(fin[0]) == len(chunks) - 2, "the finish chunk is not right before the usage chunk")
    check(all(c.get("usage") is None for c in chunks[:-1]), "usage in a chunk before the last")
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    check(text.strip(), "empty streamed text")
    # same request without include_usage: no usage chunk
    chunks2, _ = sse("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 10, "stream": True, **NOTHINK,
                     "messages": [{"role": "user", "content": "Name three colors."}]})
    check(all(c["choices"] for c in chunks2), "a chunk without choices although usage was not requested")


@test
def chat_stream_openai_client_equals_non_stream():
    msgs = [{"role": "user", "content": "Write one sentence about the sea."}]
    full = chat(msgs, max_tokens=60).choices[0].message.content
    parts, usage = [], None
    for ch in chat(msgs, max_tokens=60, stream=True, stream_options={"include_usage": True}):
        if ch.choices:
            parts.append(ch.choices[0].delta.content or "")
        if ch.usage:
            usage = ch.usage
    check("".join(parts) == full, f"stream {''.join(parts)!r} != non-stream {full!r}")
    check(usage and usage.completion_tokens > 0, usage)


@test
def chat_reasoning_is_split_out():
    r = chat([{"role": "user", "content": "What is 17 * 3? Answer briefly."}], max_tokens=700, extra_body={})
    m = r.choices[0].message
    reasoning = getattr(m, "reasoning_content", None) or (m.model_extra or {}).get("reasoning_content") or (m.model_extra or {}).get("reasoning")
    check(reasoning and "<think>" not in (m.content or "") and "</think>" not in (m.content or ""), (reasoning, m.content))
    check("51" in (m.content or ""), m.content)
    # streamed: reasoning deltas come before content deltas
    seq = []
    for ch in chat([{"role": "user", "content": "What is 17 * 3? Answer briefly."}], max_tokens=700, stream=True, extra_body={}):
        if not ch.choices:
            continue
        d = ch.choices[0].delta
        ex = d.model_extra or {}
        if ex.get("reasoning_content") or ex.get("reasoning"):
            seq.append("r")
        if d.content and d.content.strip():
            seq.append("c")
    check("r" in seq and "c" in seq and seq.index("c") > len(seq) - 1 - seq[::-1].index("r"), f"order {''.join(seq)[:80]}")
    # include_reasoning=false hides it
    r = chat([{"role": "user", "content": "What is 17 * 3? Answer briefly."}], max_tokens=700, extra_body={"include_reasoning": False})
    m = r.choices[0].message
    check(not (m.model_extra or {}).get("reasoning_content") and "51" in (m.content or ""), m)


@test
def chat_separate_reasoning_false_keeps_it_in_the_content():
    msgs = [{"role": "user", "content": "What is 17 * 3? Answer briefly."}]
    split = chat(msgs, max_tokens=700, extra_body={}).choices[0].message
    s, _, b = raw_chat({"messages": msgs, "max_tokens": 700, "separate_reasoning": False, "chat_template_kwargs": {}})
    check(s == 200, b)
    m = b["choices"][0]["message"]
    check("reasoning_content" not in m and "reasoning" not in m, f"reasoning split out although separate_reasoning=false: {list(m)}")
    check("51" in m["content"] and len(m["content"]) > len(split.content or "") + 50, (len(m["content"]), len(split.content or "")))


@test
def chat_stop_token_ids_and_skip_special_tokens():
    msgs = [{"role": "user", "content": "Write the numbers from 1 to 12 separated by commas and spaces, nothing else."}]
    seven = http("POST", "/tokenize", {"prompt": "7"})[2]["tokens"]
    check(len(seven) == 1, seven)
    # the token that ends the generation is not part of the text, as in vLLM, unless the stop string is asked for
    r = chat(msgs, max_tokens=80, extra_body={**NOTHINK, "stop_token_ids": seven})
    c = r.choices[0]
    check(c.finish_reason == "stop" and c.message.content.rstrip().endswith("6,") and "7" not in c.message.content, c.message.content)
    r = chat(msgs, max_tokens=80, extra_body={**NOTHINK, "stop_token_ids": seven, "include_stop_str_in_output": True})
    check(r.choices[0].message.content.endswith("7"), r.choices[0].message.content)
    # special tokens: left out by default; written with skip_special_tokens=false (blaze: the end-of-turn token shows
    # when the stop token is asked for, too)
    ask = [{"role": "user", "content": "Reply with exactly the word: pong"}]
    plain = chat(ask, max_tokens=20).choices[0].message.content
    check("<|im_end|>" not in plain and "<|endoftext|>" not in plain, plain)
    raw = chat(ask, max_tokens=20, extra_body={**NOTHINK, "skip_special_tokens": False, "include_stop_str_in_output": True}).choices[0].message.content
    check("<|im_end|>" in raw or "<|endoftext|>" in raw, raw)


@test
def chat_stop_strings():
    msgs = [{"role": "user", "content": "Write the numbers from 1 to 12 separated by commas and spaces, nothing else."}]
    full = chat(msgs, max_tokens=80).choices[0].message.content
    check("6, 7" in full, f"unexpected baseline {full!r}")
    r = chat(msgs, max_tokens=80, stop=["6, 7"])
    c = r.choices[0]
    check(c.finish_reason == "stop" and c.message.content == full[:full.index("6, 7")], (c.finish_reason, c.message.content))
    # a stop string that spans several tokens, streamed: never part of the output, never a prefix leaked after it
    parts = [ch.choices[0].delta.content or "" for ch in chat(msgs, max_tokens=80, stop=["6, 7"], stream=True) if ch.choices]
    check("".join(parts) == c.message.content, f"streamed {''.join(parts)!r} vs {c.message.content!r}")
    # a stop string that never occurs: the held-back text is flushed at the end
    r2 = chat(msgs, max_tokens=80, stop=["12, 13, 14"])
    check(r2.choices[0].message.content == full, (r2.choices[0].message.content, full))
    parts = [ch.choices[0].delta.content or "" for ch in chat(msgs, max_tokens=80, stop=["12, 13, 14", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13"], stream=True) if ch.choices]
    check("".join(parts) == full, f"flush: {''.join(parts)!r} vs {full!r}")
    # vLLM: include_stop_str_in_output
    r3 = chat(msgs, max_tokens=80, stop=["6, 7"], extra_body={**NOTHINK, "include_stop_str_in_output": True})
    check(r3.choices[0].message.content == full[:full.index("6, 7") + 4], r3.choices[0].message.content)
    # a single string, as OpenAI also allows
    r4 = chat(msgs, max_tokens=80, stop="6, 7")
    check(r4.choices[0].message.content == c.message.content, r4.choices[0].message.content)


@test
def chat_n_choices():
    r = chat([{"role": "user", "content": "Pick a random animal and name it."}], n=3, temperature=1.0, max_tokens=12)
    check([c.index for c in r.choices] == [0, 1, 2], [c.index for c in r.choices])
    check(all(c.message.content for c in r.choices), "empty choice")
    one = chat([{"role": "user", "content": "Pick a random animal and name it."}], max_tokens=12)
    check(r.usage.prompt_tokens == one.usage.prompt_tokens, ("prompt tokens counted once", r.usage, one.usage))
    check(r.usage.completion_tokens >= 3, r.usage)
    # streamed: chunks carry the choice index; every choice ends
    seen, fins = {}, set()
    for ch in chat([{"role": "user", "content": "Pick a random animal and name it."}], n=2, temperature=1.0, max_tokens=12, stream=True):
        for c in ch.choices:
            seen.setdefault(c.index, "")
            seen[c.index] += c.delta.content or ""
            if c.finish_reason:
                fins.add(c.index)
    check(set(seen) == {0, 1} and fins == {0, 1} and all(seen.values()), (seen, fins))
    # seeded choices are reproducible and differ from each other's seed (the first run warms the prompt cache: a cached
    # prefix and a computed one differ in the last bits, so compare cache hits with cache hits)
    chat([{"role": "user", "content": "Invent a name for a cat."}], temperature=1.2, seed=7, max_tokens=10)
    a = chat([{"role": "user", "content": "Invent a name for a cat."}], n=2, temperature=1.2, seed=7, max_tokens=10)
    b = chat([{"role": "user", "content": "Invent a name for a cat."}], n=2, temperature=1.2, seed=7, max_tokens=10)
    check([c.message.content for c in a.choices] == [c.message.content for c in b.choices], "seeded n=2 differs between runs")


@test
def chat_logprobs():
    r = chat([{"role": "user", "content": "Say hi."}], max_tokens=6, logprobs=True, top_logprobs=3)
    lp = r.choices[0].logprobs
    check(lp and lp.content and len(lp.content) >= 2, lp)
    e = lp.content[0]
    check(e.logprob <= 0 and e.token and len(e.top_logprobs) == 3 and e.bytes is not None, e)
    check(e.top_logprobs[0].logprob >= e.top_logprobs[1].logprob, "top_logprobs not sorted")
    # streamed
    n = 0
    for ch in chat([{"role": "user", "content": "Say hi."}], max_tokens=6, logprobs=True, top_logprobs=2, stream=True):
        if ch.choices and ch.choices[0].logprobs and ch.choices[0].logprobs.content:
            n += len(ch.choices[0].logprobs.content)
    check(n >= 2, f"{n} streamed logprob entries")
    error_shape(*raw_chat({"messages": [{"role": "user", "content": "x"}], "logprobs": True, "top_logprobs": 99})[::2], code=None)


@test
def chat_json_schema_and_object():
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}},
              "required": ["name", "age", "tags"], "additionalProperties": False}
    r = chat([{"role": "user", "content": "Describe a fictional person."}], max_tokens=150,
             response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": schema, "strict": True}})
    j = json.loads(r.choices[0].message.content)
    check(set(j) == {"name", "age", "tags"} and isinstance(j["age"], int) and isinstance(j["tags"], list), j)
    r = chat([{"role": "user", "content": "Give me a JSON object with a key 'a'."}], max_tokens=60, response_format={"type": "json_object"})
    check(isinstance(json.loads(r.choices[0].message.content), dict), r.choices[0].message.content)
    # vLLM's guided_* extras
    r = chat([{"role": "user", "content": "Is the sky blue?"}], max_tokens=10, extra_body={**NOTHINK, "guided_choice": ["yes", "no"]})
    check(r.choices[0].message.content in ("yes", "no"), r.choices[0].message.content)
    # streamed, with n=2
    outs = {}
    for ch in chat([{"role": "user", "content": "Describe a fictional person."}], max_tokens=150, n=2, stream=True,
                   response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": schema}}):
        for c in ch.choices:
            outs[c.index] = outs.get(c.index, "") + (c.delta.content or "")
    check(all(set(json.loads(v)) == {"name", "age", "tags"} for v in outs.values()) and len(outs) == 2, outs)


# tools

def tool_args(tc):
    return json.loads(tc.function.arguments)


@test
def tools_single_call_and_result_fed_back():
    msgs = [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]
    r = chat(msgs, tools=[WEATHER], max_tokens=300)
    c = r.choices[0]
    check(c.finish_reason == "tool_calls", c.finish_reason)
    check(c.message.content is None or not c.message.content.strip(), f"content {c.message.content!r} next to a tool call")
    tc = c.message.tool_calls
    check(len(tc) == 1 and tc[0].type == "function" and tc[0].function.name == "get_weather" and tc[0].id, tc)
    check(tool_args(tc[0])["city"].lower() == "paris", tc[0].function.arguments)
    # the result goes back; the model answers in text; the second prompt is mostly cached
    msgs2 = msgs + [{"role": "assistant", "content": c.message.content, "tool_calls": [t.model_dump() for t in tc]},
                    {"role": "tool", "tool_call_id": tc[0].id, "content": json.dumps({"temp_c": 18, "sky": "cloudy"})}]
    r2 = chat(msgs2, tools=[WEATHER], max_tokens=200)
    c2 = r2.choices[0]
    check(c2.finish_reason == "stop" and c2.message.content and not c2.message.tool_calls, c2)
    check("18" in c2.message.content, c2.message.content)
    cached = (r2.usage.prompt_tokens_details.cached_tokens if r2.usage.prompt_tokens_details else 0)
    print(f"      second turn: {cached} of {r2.usage.prompt_tokens} prompt tokens cached")
    check(cached > 0 or not args.expect_prompt_cache, f"no prompt-cache hit on the second turn (usage {r2.usage})")


@test
def tools_parallel_calls():
    msgs = [{"role": "user", "content": "What's the weather in Paris and in Berlin? Call the tool once for each city, both calls in one reply."}]
    r = chat(msgs, tools=[WEATHER], max_tokens=400)
    tc = r.choices[0].message.tool_calls
    check(r.choices[0].finish_reason == "tool_calls" and len(tc) == 2, r.choices[0])
    check({tool_args(t)["city"].lower() for t in tc} == {"paris", "berlin"}, [t.function.arguments for t in tc])
    check(len({t.id for t in tc}) == 2, "tool call ids are not unique")
    # parallel_tool_calls=false: one call
    r = chat(msgs, tools=[WEATHER], max_tokens=400, parallel_tool_calls=False)
    check(len(r.choices[0].message.tool_calls) == 1, r.choices[0].message.tool_calls)


@test
def tools_streaming_deltas_have_the_openai_shape():
    msgs = [{"role": "user", "content": "What's the weather in Paris and in Berlin? Call the tool once for each city, both calls in one reply."}]
    chunks, done = sse("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 400, "stream": True, "messages": msgs,
                                                "tools": [WEATHER], "stream_options": {"include_usage": True}, **NOTHINK})
    check(done, "no [DONE]")
    calls, order = {}, []
    finish = None
    for ch in chunks:
        for c in ch["choices"]:
            if c.get("finish_reason"):
                finish = c["finish_reason"]
            for t in c["delta"].get("tool_calls") or []:
                i = t["index"]
                if i not in calls:
                    # the first delta of a call: id, type and the name (AI SDK clients require them here)
                    check(t.get("id") and t.get("type") == "function" and t["function"].get("name"), f"first delta of call {i}: {t}")
                    calls[i] = {"id": t["id"], "name": t["function"]["name"], "args": ""}
                    order.append(i)
                else:
                    check(not t.get("id") or t["id"] == calls[i]["id"], f"id changes in call {i}")
                calls[i]["args"] += t["function"].get("arguments") or ""
    check(finish == "tool_calls", f"finish_reason {finish}")
    check(order == sorted(order) and len(calls) == 2, order)
    check({json.loads(v["args"])["city"].lower() for v in calls.values()} == {"paris", "berlin"}, calls)
    # the calls of a call-then-text turn come after the text deltas, never mixed into one delta
    for ch in chunks:
        for c in ch["choices"]:
            d = c["delta"]
            check(not (d.get("tool_calls") and d.get("content")), f"content and tool_calls in one delta: {d}")
    # the OpenAI client's own accumulator accepts the stream
    acc = {}
    for ch in chat(msgs, tools=[WEATHER], max_tokens=400, stream=True):
        for c in ch.choices:
            for t in c.delta.tool_calls or []:
                a = acc.setdefault(t.index, {"name": "", "args": ""})
                a["name"] += (t.function.name or "")
                a["args"] += (t.function.arguments or "")
    check(len(acc) == 2 and all(v["name"] == "get_weather" and json.loads(v["args"]) for v in acc.values()), acc)


@test
def tools_tool_choice_none_required_named():
    msgs = [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]
    r = chat(msgs, tools=[WEATHER], tool_choice="none", max_tokens=100)
    check(not r.choices[0].message.tool_calls and "<tool_call>" not in (r.choices[0].message.content or ""), r.choices[0].message)
    # required: a call even when the question does not need one
    r = chat([{"role": "user", "content": "Hello there!"}], tools=[WEATHER, TIME], tool_choice="required", max_tokens=300)
    c = r.choices[0]
    check(c.finish_reason == "tool_calls" and c.message.tool_calls and c.message.tool_calls[0].function.name in ("get_weather", "get_time"), c)
    # named: that function
    r = chat([{"role": "user", "content": "Tell me something about Rome."}], tools=[WEATHER, TIME], max_tokens=300,
             tool_choice={"type": "function", "function": {"name": "get_time"}})
    c = r.choices[0]
    check(c.finish_reason == "tool_calls" and c.message.tool_calls[0].function.name == "get_time", c)
    json.loads(c.message.tool_calls[0].function.arguments)
    # streamed required
    names = []
    for ch in chat([{"role": "user", "content": "Hello there!"}], tools=[WEATHER, TIME], tool_choice="required", max_tokens=300, stream=True):
        for c in ch.choices:
            for t in c.delta.tool_calls or []:
                if t.function and t.function.name:
                    names.append(t.function.name)
    check(names and names[0] in ("get_weather", "get_time"), names)
    # errors
    error_shape(*raw_chat({"messages": msgs, "tool_choice": "required"})[::2])
    error_shape(*raw_chat({"messages": msgs, "tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "nope"}}})[::2])


@test
def tools_multi_turn_agent_loop():
    # two rounds of tools like an agent: weather, then time, then the answer
    msgs = [{"role": "user", "content": "Tell me the weather in Oslo and then its local time. Call one tool at a time."}]
    seen = []
    for _ in range(4):
        r = chat(msgs, tools=[WEATHER, TIME], max_tokens=300)
        m = r.choices[0].message
        if not m.tool_calls:
            break
        msgs.append({"role": "assistant", "content": m.content, "tool_calls": [t.model_dump() for t in m.tool_calls]})
        for t in m.tool_calls:
            seen.append(t.function.name)
            res = {"temp_c": 4} if t.function.name == "get_weather" else {"time": "14:05"}
            msgs.append({"role": "tool", "tool_call_id": t.id, "content": json.dumps(res)})
    check("get_weather" in seen and "get_time" in seen, seen)
    check(m.content and not m.tool_calls, m)


# prompt controls

@test
def chat_continue_final_message_and_generation_prompt():
    msgs = [{"role": "user", "content": "Complete the sentence: The capital of France is"},
            {"role": "assistant", "content": "The capital of France is"}]
    s, _, b = raw_chat({"messages": msgs, "continue_final_message": True, "add_generation_prompt": False, "max_tokens": 12})
    check(s == 200, b)
    text = b["choices"][0]["message"]["content"]
    check("Paris" in text, f"the continuation {text!r}")
    check(not text.lstrip().startswith("The capital"), f"the final message was repeated, not continued: {text!r}")
    # the prompt ends with the message's text, its turn left open
    t = http("POST", "/tokenize", {"messages": msgs, "continue_final_message": True, "add_generation_prompt": False,
                                   "return_token_strs": True, **NOTHINK})[2]
    tail = "".join(t["token_strs"])
    check(tail.endswith("The capital of France is"), repr(tail[-80:]))
    check(not tail.rstrip().endswith("<|im_end|>"), repr(tail[-80:]))
    # both flags set is an error; so is continuing a user message
    error_shape(*raw_chat({"messages": msgs, "continue_final_message": True, "add_generation_prompt": True})[::2])
    error_shape(*raw_chat({"messages": msgs[:1], "continue_final_message": True, "add_generation_prompt": False})[::2])
    # no generation prompt: the template ends after the user turn
    full = "".join(http("POST", "/tokenize", {"messages": msgs[:1], "add_generation_prompt": False, "return_token_strs": True})[2]["token_strs"])
    gen = "".join(http("POST", "/tokenize", {"messages": msgs[:1], "add_generation_prompt": True, "return_token_strs": True})[2]["token_strs"])
    check(gen.startswith(full) and gen.rstrip().endswith("<think>") and "assistant" not in full.split("<|im_end|>")[-1], (full[-60:], gen[-60:]))


@test
def tokenize_detokenize():
    s, _, t = http("POST", "/tokenize", {"prompt": "Hello, world! 你好"})
    check(s == 200 and t["count"] == len(t["tokens"]) > 2 and t["max_model_len"] == MAX_LEN, t)
    s, _, d = http("POST", "/detokenize", {"tokens": t["tokens"]})
    check(s == 200 and d["prompt"] == "Hello, world! 你好", d)
    s, _, t2 = http("POST", "/tokenize", {"prompt": "Hi", "return_token_strs": True})
    check(t2["token_strs"] and "".join(t2["token_strs"]) == "Hi", t2)
    # the chat form counts exactly what a chat request then uses as its prompt
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say hi."}]
    n = http("POST", "/tokenize", {"messages": msgs, **NOTHINK})[2]["count"]
    r = chat(msgs, max_tokens=3)
    check(r.usage.prompt_tokens == n, (r.usage.prompt_tokens, n))
    n2 = http("POST", "/tokenize", {"messages": msgs, "tools": [WEATHER], **NOTHINK})[2]["count"]
    check(n2 > n, "tools do not change the token count")
    error_shape(*http("POST", "/detokenize", {"tokens": [10 ** 9]})[::2])
    error_shape(*http("POST", "/tokenize", {})[::2])


@test
def chat_multimodal_parts_are_text_or_rejected():
    r = chat([{"role": "user", "content": [{"type": "text", "text": "Reply with the word: ok"}]}], max_tokens=10)
    check("ok" in r.choices[0].message.content.lower(), r.choices[0].message.content)
    s, _, b = raw_chat({"messages": [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]})
    error_shape(s, b)


# errors, limits

@test
def errors_bad_requests():
    s, _, b = http("POST", "/v1/chat/completions", b"{not json")
    error_shape(s, b)
    error_shape(*http("POST", "/v1/chat/completions", [1, 2])[::2])
    error_shape(*http("POST", "/v1/chat/completions", {"model": MODEL})[::2])
    error_shape(*raw_chat({"messages": []})[::2])
    e = error_shape(*raw_chat({"messages": [{"role": "user", "content": "x"}], "temperature": -1})[::2])
    check(e["param"] == "temperature", e)
    error_shape(*raw_chat({"messages": [{"role": "user", "content": "x"}], "n": 0})[::2])
    error_shape(*raw_chat({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0})[::2])
    error_shape(*raw_chat({"messages": [{"role": "user", "content": "x"}], "top_p": 1.5})[::2])
    error_shape(*raw_chat({"messages": [{"role": "user", "content": "x"}], "stop_token_ids": [10 ** 9]})[::2])
    error_shape(*raw_chat({"messages": [{"role": "user", "content": "x"}], "logit_bias": {"999999999": 5}})[::2])
    error_shape(*raw_chat({"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_schema"}})[::2])
    error_shape(*http("POST", "/v1/completions", {"model": MODEL, "prompt": {"a": 1}})[::2])
    error_shape(*http("POST", "/v1/completions", {"model": MODEL, "prompt": [10 ** 9], "max_tokens": 4})[::2])
    # the server is still fine
    check(chat([{"role": "user", "content": "Say ok"}], max_tokens=5).choices[0].message.content, "no answer after the errors")


@test
def errors_context_overflow():
    big = "the " * (MAX_LEN + 50)
    s, _, b = raw_chat({"messages": [{"role": "user", "content": big}], "max_tokens": 5})
    e = error_shape(s, b, code="context_length_exceeded")
    check(str(MAX_LEN) in e["message"] and "prompt" in e["message"].lower(), e["message"])
    s, _, b = raw_chat({"messages": [{"role": "user", "content": big}], "max_tokens": 5, "stream": True})
    error_shape(s, b, code="context_length_exceeded")
    s, _, b = http("POST", "/v1/completions", {"model": MODEL, "prompt": big, "max_tokens": 5})
    error_shape(s, b, code="context_length_exceeded")
    # the prompt fits, max_tokens does not: the answer is cut at the end of the context, not an error
    s, _, b = raw_chat({"messages": [{"role": "user", "content": "Say ok"}], "max_tokens": MAX_LEN * 4})
    check(s == 200 and b["choices"][0]["finish_reason"] in ("stop", "length"), (s, b))
    # truncate_prompt_tokens keeps the tail and lets the request through
    s, _, b = http("POST", "/v1/completions", {"model": MODEL, "prompt": big, "max_tokens": 3, "truncate_prompt_tokens": 100})
    check(s == 200 and b["usage"]["prompt_tokens"] == 100, (s, b))
    # the tokenizer endpoint reports the size, it does not refuse
    s, _, t = http("POST", "/tokenize", {"messages": [{"role": "user", "content": big}]})
    check(s == 200 and t["count"] > MAX_LEN, (s, t.get("count") if isinstance(t, dict) else t))


# completions

@test
def completions_basic_forms():
    r = client.completions.create(model=MODEL, prompt="The capital of France is", max_tokens=6, temperature=0)
    check(r.object == "text_completion" and r.choices[0].text and r.usage.prompt_tokens > 0, r)
    check(r.choices[0].finish_reason in ("length", "stop"), r.choices[0].finish_reason)
    check("Paris" in r.choices[0].text, r.choices[0].text)
    # prompt as token ids
    ids = http("POST", "/tokenize", {"prompt": "The capital of France is"})[2]["tokens"]
    r2 = client.completions.create(model=MODEL, prompt=ids, max_tokens=6, temperature=0)
    check(r2.choices[0].text == r.choices[0].text and r2.usage.prompt_tokens == len(ids), (r2.choices[0].text, r.choices[0].text))
    # a list of prompts: one choice each, in order
    r3 = client.completions.create(model=MODEL, prompt=["The capital of France is", "The capital of Italy is"], max_tokens=6, temperature=0)
    check([c.index for c in r3.choices] == [0, 1] and "Paris" in r3.choices[0].text and "Rome" in r3.choices[1].text, [c.text for c in r3.choices])
    r4 = client.completions.create(model=MODEL, prompt=[ids, ids], max_tokens=4, temperature=0)
    check(len(r4.choices) == 2 and r4.choices[0].text == r4.choices[1].text, r4.choices)
    # n and echo
    r5 = client.completions.create(model=MODEL, prompt="Once upon a", max_tokens=5, n=2, temperature=1.0, echo=True)
    check(len(r5.choices) == 2 and all(c.text.startswith("Once upon a") for c in r5.choices), [c.text for c in r5.choices])
    # stop strings
    r6 = client.completions.create(model=MODEL, prompt="1, 2, 3, 4, 5, 6, 7, 8,", max_tokens=30, temperature=0, stop=["12"])
    check("12" not in r6.choices[0].text and r6.choices[0].finish_reason == "stop", r6.choices[0])


@test
def completions_stream_usage_and_logprobs():
    chunks = list(client.completions.create(model=MODEL, prompt="Count: 1, 2, 3,", max_tokens=10, temperature=0, stream=True,
                                            stream_options={"include_usage": True}, logprobs=2))
    check(chunks[-1].usage and chunks[-1].usage.completion_tokens > 0 and chunks[-1].choices == [], chunks[-1])
    text = "".join(c.choices[0].text for c in chunks if c.choices)
    non = client.completions.create(model=MODEL, prompt="Count: 1, 2, 3,", max_tokens=10, temperature=0, logprobs=2)
    check(text == non.choices[0].text, (text, non.choices[0].text))
    lp = non.choices[0].logprobs
    check(lp and len(lp.tokens) == len(lp.token_logprobs) == len(lp.top_logprobs) == len(lp.text_offset) >= 2, lp)
    check(lp.text_offset[0] == 0 and all(len(t) <= 2 for t in lp.top_logprobs), lp)
    sl = [c.choices[0].logprobs for c in chunks if c.choices and c.choices[0].logprobs]
    check(sl and sum(len(x.tokens) for x in sl) >= 2, "no logprobs in the streamed chunks")
    # no usage chunk unless asked
    chunks = list(client.completions.create(model=MODEL, prompt="Hi", max_tokens=3, temperature=0, stream=True))
    check(all(c.choices for c in chunks), "usage chunk without include_usage")
    error_shape(*http("POST", "/v1/completions", {"model": MODEL, "prompt": "Hi", "max_tokens": 3, "echo": True, "logprobs": 1})[::2])


# cancellation

def quiesced(timeout=60):
    return wait_for(lambda: (lambda m: m["blaze_requests_running"] == 0 and m["blaze_requests_waiting"] == 0)(metrics()),
                    "the scheduler to go idle", timeout)


@test
def cancel_streaming_disconnect():
    quiesced()
    before = metrics()["blaze_requests_cancelled_total"]
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Write a very long story about a dragon."}], "max_tokens": 3000,
            "stream": True, "temperature": 0.7, **NOTHINK}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=60)
    got = 0
    for line in r:
        if line.startswith(b"data:"):
            got += 1
        if got >= 6:
            break
    check(metrics()["blaze_requests_running"] >= 1, "the request is not running")
    r.close()
    t0 = time.time()
    quiesced(20)
    check(time.time() - t0 < 10, f"the scheduler kept generating {time.time() - t0:.1f} s after the disconnect")
    check(metrics()["blaze_requests_cancelled_total"] > before, "the cancellation was not counted by the scheduler")


@test
def cancel_non_streaming_disconnect():
    quiesced()
    before = metrics()["blaze_requests_cancelled_total"]
    host, port = urllib.parse.urlparse(BASE).hostname, urllib.parse.urlparse(BASE).port or 80
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Write a very long story about a dragon."}],
                       "max_tokens": 3000, "temperature": 0.7, **NOTHINK}).encode()
    s = socket.create_connection((host, port))
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
    wait_for(lambda: metrics()["blaze_requests_running"] >= 1, "the request to start")
    time.sleep(1.0)
    s.close()
    t0 = time.time()
    quiesced(20)
    check(time.time() - t0 < 10, f"the scheduler kept generating {time.time() - t0:.1f} s after the disconnect")
    check(metrics()["blaze_requests_cancelled_total"] > before, "the cancellation was not counted by the scheduler")


@test
def cancel_during_prefill():
    quiesced()
    before = metrics()["blaze_requests_cancelled_total"]
    n = args.prefill_tokens or min(12000, MAX_LEN * 3 // 4)
    # varied text (no prefix-cache hit, one token or two per word)
    text = " ".join(f"item{i} is {i * 7 % 13} of {i % 5}." for i in range(n // 9))
    body = {"model": MODEL, "messages": [{"role": "user", "content": text + "\nSummarize."}], "max_tokens": 5, "stream": True, **NOTHINK}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=60)
    wait_for(lambda: metrics()["blaze_requests_running"] >= 1, "the prefill to start")
    time.sleep(1.0)
    r.close()
    quiesced(30)
    gone = time.time() - t0
    # a full prefill of this prompt is measured below; the cancelled one must have stopped well before it
    t1 = time.time()
    s, _, b = raw_chat({"messages": [{"role": "user", "content": text + "\nSummarize in one word."}], "max_tokens": 3})
    full = time.time() - t1
    check(s == 200, b)
    print(f"      prefill cancelled after {gone:.1f} s; the same prompt takes {full:.1f} s ({b['usage']['prompt_tokens']} tokens)")
    check(metrics()["blaze_requests_cancelled_total"] > before, "the cancellation was not counted by the scheduler")
    check(gone < full * 0.8 or full < 4, f"cancelled prefill took {gone:.1f} s of {full:.1f} s")


@test
def concurrency_parallel_requests():
    res = {}

    def one(i):
        r = chat([{"role": "user", "content": f"Reply with only the number {i * 11}."}], max_tokens=10)
        res[i] = r.choices[0].message.content

    th = [threading.Thread(target=one, args=(i,)) for i in range(1, 5)]
    [t.start() for t in th]
    [t.join() for t in th]
    check(all(str(i * 11) in res[i] for i in range(1, 5)), res)
    quiesced()
def main():
    global MODEL, MAX_LEN
    if args.list:
        for t in TESTS:
            print(t.__name__)
        return 0
    try:
        card = http("GET", "/v1/models")[2]["data"][0]
    except Exception as e:
        sys.exit(f"cannot reach {BASE}: {e}")
    MODEL = args.model or card["id"]
    MAX_LEN = card.get("max_model_len") or 16384
    pick = [s for s in args.k.split(",") if s]
    skip = [s for s in args.skip.split(",") if s]
    run = [t for t in TESTS if (not pick or any(s in t.__name__ for s in pick)) and not any(s in t.__name__ for s in skip)]
    print(f"{BASE}  model {MODEL}  context {MAX_LEN}  {len(run)} tests")
    failed = []
    for t in run:
        t0 = time.time()
        try:
            t()
            print(f"  PASS {t.__name__}  ({time.time() - t0:.1f} s)")
        except Exception as e:
            failed.append(t.__name__)
            print(f"  FAIL {t.__name__}  ({time.time() - t0:.1f} s): {type(e).__name__}: {str(e)[:600]}")
            if not isinstance(e, AssertionError):
                traceback.print_exc(limit=4)
    print(f"{len(run) - len(failed)}/{len(run)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
