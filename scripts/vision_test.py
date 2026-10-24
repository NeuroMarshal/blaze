#!/usr/bin/env python3
"""End-to-end test of image input through a running `blaze serve` (OpenAI chat completions with image_url parts).

  python scripts/vision_test.py http://localhost:8102 [--model NAME] [--images DIR]

Pictures come from scripts/make_test_images.py (PIL needed only to draw them; --images DIR reads existing ones). Every
check prints PASS / FAIL with the model's answer; the exit code is the number of failures. Standard library only.
"""
import argparse
import base64
import importlib.util
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def data_url(path):
    mime = "image/jpeg" if path.endswith((".jpg", ".jpeg")) else "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def image_part(path, detail=None):
    img = {"url": data_url(path)}
    if detail:
        img["detail"] = detail
    return {"type": "image_url", "image_url": img}


def user(*parts):
    return {"role": "user", "content": list(parts)}


def text(t):
    return {"type": "text", "text": t}


class Run:
    def __init__(self, base, model):
        self.base, self.model, self.failed = base, model, 0

    def chat(self, messages, **kw):
        body = {"model": self.model, "messages": messages, "max_tokens": kw.pop("max_tokens", 160), "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}, **kw}
        return post(self.base, "/v1/chat/completions", body)

    def check(self, name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
        self.failed += not ok

    def answer(self, name, messages, expect_all=(), expect_any=(), **kw):
        status, r = self.chat(messages, **kw)
        if status != 200:
            self.check(name, False, f"HTTP {status}: {r}")
            return None
        msg = r["choices"][0]["message"].get("content") or ""
        low = msg.lower()
        ok = all(w.lower() in low for w in expect_all) and (not expect_any or any(w.lower() in low for w in expect_any))
        self.check(name, ok, f"prompt_tokens {r['usage']['prompt_tokens']}, cached {r['usage']['prompt_tokens_details']['cached_tokens']}: {msg.strip()[:300]!r}")
        return r

    def error(self, name, messages, code, needle, **kw):
        status, r = self.chat(messages, **kw)
        msg = (r.get("error") or {}).get("message", "")
        self.check(name, status == code and needle in msg, f"HTTP {status}: {msg[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--model", default=None)
    ap.add_argument("--images", default=None, help="directory with words.png sunset.png door.jpg tiny.png")
    ap.add_argument("--cache", action="store_true", help="the server has --prompt-cache-slots: the follow-up turn must hit the cache")
    ap.add_argument("--limit", type=int, default=None, help="the server's --limit-mm-per-prompt image count, to test the 400")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    model = a.model or json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
    d = a.images
    if d is None:
        d = tempfile.mkdtemp(prefix="blaze-vision-")
        spec = importlib.util.spec_from_file_location("mk", os.path.join(HERE, "make_test_images.py"))
        mk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mk)
        mk.words(os.path.join(d, "words.png")), mk.sunset(os.path.join(d, "sunset.png")), mk.cats(os.path.join(d, "door.jpg")), mk.tiny(os.path.join(d, "tiny.png"))
    p = lambda n: os.path.join(d, n)
    t = Run(base, model)

    # text only keeps working
    t.answer("text only", [{"role": "user", "content": "What is the capital of France? One word."}], expect_all=["Paris"], max_tokens=20)

    # the picture of words and shapes (PNG, 640x480)
    r = t.answer("describe words.png", [user(image_part(p("words.png")), text("Describe this picture: which word is written large, and which shapes with which colors are there?"))],
                 expect_all=["BLAZE", "circle", "square", "triangle"], expect_any=["red"])
    t.answer("read text in words.png", [user(image_part(p("words.png")), text("What is written at the bottom of the picture? Answer with the text only."))], expect_all=["HELLO", "42"], max_tokens=30)
    # a JPEG (1024x768)
    t.answer("read door.jpg", [user(image_part(p("door.jpg")), text("What sentence is written at the top, and which numbers are shown at the bottom?"))],
             expect_all=["door", "1", "2", "3"], expect_any=["open"])
    # the image before the question, one after the other, and two images in one request
    t.answer("two images", [user(text("Picture A:"), image_part(p("sunset.png")), text("Picture B:"), image_part(p("words.png")),
                                 text("Which picture shows a sun and which one has the word BLAZE? Answer briefly."))],
             expect_all=["A", "B"], expect_any=["sun"])
    # a multi-turn conversation that repeats the image: the prefix cache must serve the first turn
    first = [user(image_part(p("words.png")), text("What word is written at the top?"))]
    s, r1 = t.chat(first, max_tokens=40)
    ans = (r1["choices"][0]["message"].get("content") or "").strip() if s == 200 else ""
    follow = first + [{"role": "assistant", "content": ans}, {"role": "user", "content": "And what color is the circle?"}]
    r2 = t.answer("follow-up turn", follow, expect_any=["red"], max_tokens=40)
    if r2 and a.cache:
        c = r2["usage"]["prompt_tokens_details"]["cached_tokens"]
        t.check("prompt cache with an image", c > 0, f"cached {c} of {r2['usage']['prompt_tokens']}")
    # a different image with the same text must not reuse the first image's state
    t.answer("other image, same question", [user(image_part(p("sunset.png")), text("What word is written at the top?"))], expect_all=["SUNSET"], max_tokens=30)
    t.answer("original image again", first, expect_all=["BLAZE"], max_tokens=30)

    # settings: detail low and mm_processor_kwargs shrink the image (fewer prompt tokens), high / auto do not
    def tokens(detail=None, kw=None):
        body = {"messages": [user(image_part(p("door.jpg"), detail), text("hi"))], "max_tokens": 1, "temperature": 0}
        if kw is not None:
            body["mm_processor_kwargs"] = kw
        s, r = post(base, "/v1/chat/completions", {"model": model, "chat_template_kwargs": {"enable_thinking": False}, **body})
        return r["usage"]["prompt_tokens"] if s == 200 else (s, r)

    full, auto, high, low = tokens(), tokens("auto"), tokens("high"), tokens("low")
    t.check("detail auto / high = the processor's own limits", full == auto == high, f"default {full}, auto {auto}, high {high}")
    t.check("detail low shrinks the image", isinstance(low, int) and low < full, f"low {low} < default {full}")
    small = tokens(kw={"max_pixels": 100 * 32 * 32})
    t.check("mm_processor_kwargs max_pixels", isinstance(small, int) and small < full and small <= 100 + 60, f"{small} tokens with max_pixels = 100 tokens")
    small2 = tokens(kw={"size": {"longest_edge": 100 * 32 * 32}})
    t.check("mm_processor_kwargs size.longest_edge", small2 == small, f"{small2}")
    big = tokens(kw={"min_pixels": 3000 * 32 * 32, "max_pixels": 4000 * 32 * 32})
    t.check("mm_processor_kwargs min_pixels", isinstance(big, int) and big > full, f"{big} tokens with min_pixels = 3000 tokens")
    t.error("bad mm_processor_kwargs", [user(image_part(p("words.png")), text("hi"))], 400, "max_pixels", mm_processor_kwargs={"max_pixels": -1})

    # errors
    t.error("not an image", [user({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, text("hi"))], 400, "image")
    t.error("bad scheme", [user({"type": "image_url", "image_url": {"url": "ftp://example.com/x.png"}}, text("hi"))], 400, "URL")
    t.error("bad detail", [user(image_part(p("words.png"), "ultra"), text("hi"))], 400, "detail")
    if a.limit is not None:
        many = [user(*[image_part(p("tiny.png")) for _ in range(a.limit + 1)], text("hi"))]
        t.error("limit-mm-per-prompt", many, 400, f"At most {a.limit} image(s) may be provided in one prompt.")
    # an image that cannot fit the context
    t.error("image beyond the context", [user(image_part(p("door.jpg")), text("hi"))], 400, "context length", mm_processor_kwargs={"min_pixels": 10**9, "max_pixels": 2 * 10**9})
    # streaming
    s, _ = post(base, "/v1/chat/completions", {"model": model, "messages": first, "max_tokens": 5, "chat_template_kwargs": {"enable_thinking": False}})
    t.check("request with the image after the others", s == 200, f"HTTP {s}")
    print(f"{'ALL PASSED' if not t.failed else str(t.failed) + ' FAILED'}")
    sys.exit(t.failed)


if __name__ == "__main__":
    main()
