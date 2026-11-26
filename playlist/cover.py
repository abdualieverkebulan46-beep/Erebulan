#!/usr/bin/env python3
"""Draws the static 1920x1080 background: can + laptop + earbuds + "playlist."
All objects are original, unbranded drawings (no real logos)."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

S = 2                      # supersampling
W, H = 1920 * S, 1080 * S
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FONT = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"


def shadow(img, box, radius, blur, alpha=90, off=(0, 18)):
    sh = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(sh)
    x0, y0, x1, y1 = box
    d.rounded_rectangle((x0 + off[0] * S, y0 + off[1] * S, x1 + off[0] * S, y1 + off[1] * S), radius, fill=alpha)
    sh = sh.filter(ImageFilter.GaussianBlur(blur * S))
    img.paste((0, 0, 0), mask=sh)


def hgrad(w, h, stops):
    """horizontal gradient from list of (pos, rgb)"""
    xs = np.linspace(0, 1, w)
    arr = np.zeros((h, w, 3))
    for c in range(3):
        arr[:, :, c] = np.interp(xs, [p for p, _ in stops], [col[c] for _, col in stops])[None, :]
    return Image.fromarray(arr.astype(np.uint8))


def vgrad(w, h, top, bot):
    a = np.linspace(0, 1, h)[:, None, None]
    arr = np.array(top) * (1 - a) + np.array(bot) * a
    return Image.fromarray(np.broadcast_to(arr, (h, w, 3)).astype(np.uint8))


def rounded_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
    return m


def draw_can(img, cx, top):
    w, h = 150 * S, 365 * S
    x0, y0 = cx - w // 2, top
    shadow(img, (x0, y0, x0 + w, y0 + h), 40 * S, 14, 70)
    body = hgrad(w, h, [(0, (150, 152, 158)), (0.18, (238, 240, 243)), (0.32, (255, 255, 255)),
                        (0.6, (226, 228, 232)), (0.88, (170, 172, 178)), (1, (120, 122, 128))])
    d = ImageDraw.Draw(body)
    # label: original geometric design
    lab_top, lab_bot = int(h * 0.16), int(h * 0.82)
    for i in range(5):
        y = lab_top + 50 * S + i * 26 * S
        d.line([(int(w * 0.18), y), (int(w * 0.82), y - 34 * S)], fill=(40, 44, 52), width=7 * S)
    f1 = ImageFont.truetype(FONT, 34 * S)
    f2 = ImageFont.truetype(FONT, 15 * S)
    for txt, f, y, col in (("FOCUS", f1, int(h * 0.62), (30, 32, 38)),
                           ("ZERO SUGAR", f2, int(h * 0.075), (60, 64, 72)),
                           ("CITRUS · CALM", f2, int(h * 0.74), (40, 150, 170))):
        tw = d.textlength(txt, font=f)
        d.text(((w - tw) / 2, y), txt, font=f, fill=col)
    mask = rounded_mask(w, h, 26 * S)
    img.paste(body, (x0, y0), mask)
    dd = ImageDraw.Draw(img)
    # top rim
    dd.rounded_rectangle((x0 + 6 * S, y0 - 6 * S, x0 + w - 6 * S, y0 + 12 * S), 8 * S, fill=(200, 202, 207), outline=(150, 152, 158), width=2 * S)
    dd.rounded_rectangle((x0 + 8 * S, y0 + h - 14 * S, x0 + w - 8 * S, y0 + h + 4 * S), 8 * S, fill=(185, 187, 192))


def draw_laptop(img, box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    shadow(img, box, 22 * S, 22, 110, (0, 22))
    lid = vgrad(w, h, (66, 71, 82), (44, 48, 57))
    # soft diagonal sheen
    sheen = Image.new("L", (w, h), 0)
    ImageDraw.Draw(sheen).polygon([(0, 0), (int(w * 0.45), 0), (int(w * 0.2), h), (0, h)], fill=22)
    sheen = sheen.filter(ImageFilter.GaussianBlur(60 * S))
    lid.paste((255, 255, 255), mask=sheen)
    img.paste(lid, (x0, y0), rounded_mask(w, h, 22 * S))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, 22 * S, outline=(30, 33, 40), width=3 * S)
    # small original mark: a circle-and-dot
    cx, cy, r = (x0 + x1) // 2, (y0 + y1) // 2, 34 * S
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(28, 30, 36), width=7 * S)
    d.ellipse((cx - 9 * S, cy - 9 * S, cx + 9 * S, cy + 9 * S), fill=(28, 30, 36))


def draw_case(img, box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    shadow(img, box, 70 * S, 16, 70, (0, 16))
    body = vgrad(w, h, (255, 255, 255), (232, 233, 236))
    img.paste(body, (x0, y0), rounded_mask(w, h, 70 * S))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, 70 * S, outline=(205, 206, 210), width=2 * S)
    # closed lid seam + soft top highlight
    ly = y0 + int(h * 0.34)
    d.line([(x0 + 6 * S, ly), (x1 - 6 * S, ly)], fill=(196, 197, 202), width=3 * S)
    hl = Image.new("L", img.size, 0)
    ImageDraw.Draw(hl).rounded_rectangle((x0 + 30 * S, y0 + 14 * S, x1 - 30 * S, y0 + 40 * S), 14 * S, fill=60)
    img.paste((255, 255, 255), mask=hl.filter(ImageFilter.GaussianBlur(8 * S)))
    # status light
    sx, sy = (x0 + x1) // 2, y0 + int(h * 0.62)
    d.ellipse((sx - 4 * S, sy - 4 * S, sx + 4 * S, sy + 4 * S), fill=(120, 220, 160))


def main(text="playlist."):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw_can(img, 345 * S, 290 * S)
    draw_laptop(img, (550 * S, 230 * S, 1370 * S, 805 * S))
    draw_case(img, (1460 * S, 450 * S, 1735 * S, 680 * S))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 118 * S)
    tw = d.textlength(text, font=f)
    d.text(((W - tw) / 2, 860 * S), text, font=f, fill=(12, 12, 14))
    img = img.resize((1920, 1080), Image.LANCZOS)
    os.makedirs(OUT, exist_ok=True)
    img.save(os.path.join(OUT, "cover.png"))
    img.convert("RGB").save(os.path.join(OUT, "thumbnail.jpg"), quality=92)


if __name__ == "__main__":
    main()
