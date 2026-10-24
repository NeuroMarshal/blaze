#!/usr/bin/env python3
"""Test pictures for the vision path: large written words and simple shapes a model can describe unambiguously.

  python scripts/make_test_images.py OUT_DIR
"""
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont


def font(size):
    return ImageFont.load_default(size=size)


def words(path, w=640, h=480):
    """White page: the word BLAZE, a red circle, a blue square, a green triangle, and HELLO 42."""
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((w * 0.05, h * 0.03), "BLAZE", font=font(h // 4), fill="black")
    d.ellipse((w * 0.06, h * 0.40, w * 0.30, h * 0.40 + w * 0.24), fill=(220, 30, 30))
    d.rectangle((w * 0.38, h * 0.42, w * 0.38 + w * 0.22, h * 0.42 + w * 0.22), fill=(30, 60, 220))
    cx, top, s = w * 0.80, h * 0.40, w * 0.12
    d.polygon([(cx, top), (cx - s, top + s * 1.7), (cx + s, top + s * 1.7)], fill=(30, 160, 60))
    d.text((w * 0.05, h * 0.80), "HELLO 42", font=font(h // 8), fill="black")
    im.save(path)


def sunset(path, w=448, h=336):
    """Orange sky gradient with a big yellow sun over a dark blue sea and the word SUNSET."""
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            t = y / h
            px[x, y] = (int(255 - 80 * t), int(140 - 90 * t), int(40 + 40 * t)) if y < h * 0.65 else (10, 30, 90 + int(40 * (y / h)))
    d = ImageDraw.Draw(im)
    d.ellipse((w * 0.35, h * 0.30, w * 0.65, h * 0.30 + w * 0.30), fill=(255, 230, 40))
    d.text((w * 0.22, h * 0.04), "SUNSET", font=font(h // 5), fill="white")
    im.save(path)


def cats(path, w=1024, h=768):
    """JPEG, 4:3: a table of three coloured stripes, the numbers 1 2 3, and the sentence OPEN THE DOOR."""
    im = Image.new("RGB", (w, h), (235, 235, 220))
    d = ImageDraw.Draw(im)
    for i, c in enumerate([(200, 40, 40), (40, 160, 70), (40, 80, 200)]):
        d.rectangle((0, h * 0.30 + i * h * 0.10, w, h * 0.30 + (i + 1) * h * 0.10), fill=c)
    d.text((w * 0.04, h * 0.02), "OPEN THE DOOR", font=font(h // 9), fill=(20, 20, 20))
    for i, n in enumerate("123"):
        d.text((w * (0.15 + 0.30 * i), h * 0.65), n, font=font(h // 4), fill=(20, 20, 20))
    for i in range(12):
        a = i / 12 * 2 * math.pi
        d.line((w * 0.5, h * 0.50, w * 0.5 + 80 * math.cos(a), h * 0.50 + 80 * math.sin(a)), fill=(255, 255, 255), width=3)
    im.save(path, quality=92)


def tiny(path, w=200, h=150):
    """Below min_pixels: the processor upscales it to 320 x 224."""
    im = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.ellipse((20, 20, 100, 100), fill=(230, 30, 30))
    d.text((110, 50), "RED", font=font(40), fill="black")
    im.save(path)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    words(os.path.join(out, "words.png"))
    sunset(os.path.join(out, "sunset.png"))
    cats(os.path.join(out, "door.jpg"))
    tiny(os.path.join(out, "tiny.png"))
