#!/usr/bin/env python3
"""Three cover concepts for the series (all objects drawn, no real logos):
  1. clean   - white flat-lay: laptop, earbuds case, energy can     "playlist."
  2. night   - dark desk, open laptop glowing, over-ear headphones   "late night focus."
  3. morning - warm beige flat-lay, laptop, headphones, latte, notes  "morning focus."
"""
import math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from cover import (S, W, H, OUT, FONT, shadow, hgrad, vgrad, rounded_mask,
                   draw_can, draw_laptop, draw_case)


def soft(img, shape_fn, blur, color, alpha=255):
    m = Image.new("L", img.size, 0)
    shape_fn(ImageDraw.Draw(m), alpha)
    img.paste(color, mask=m.filter(ImageFilter.GaussianBlur(blur * S)))


def title(img, text, y, color, size=118):
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, size * S)
    tw = d.textlength(text, font=f)
    d.text(((W - tw) / 2, y * S), text, font=f, fill=color)


def headphones_top(img, cx, cy, scale=1.0, col=(26, 26, 28), accent=(176, 146, 92)):
    """over-ear headphones lying flat, seen from above"""
    s = S * scale
    d = ImageDraw.Draw(img)
    r = 150 * s
    soft(img, lambda dd, a: dd.arc((cx - r, cy - r + 30 * s, cx + r, cy + r + 30 * s), 180, 360, fill=a, width=int(30 * s)), 14, (0, 0, 0), 70)
    d.arc((cx - r, cy - r, cx + r, cy + r), 180, 360, fill=(40, 40, 42), width=int(28 * s))
    d.arc((cx - r + 6 * s, cy - r + 6 * s, cx + r - 6 * s, cy + r - 6 * s), 200, 340, fill=(70, 70, 74), width=int(6 * s))
    for side in (-1, 1):
        x = cx + side * r
        box = (x - 62 * s, cy - 10 * s, x + 62 * s, cy + 150 * s)
        shadow(img, tuple(int(v) for v in box), int(48 * s), 16, 90, (0, 16))
        d.rounded_rectangle(box, int(50 * s), fill=col)
        # textured cup: subtle grain lines
        for k in range(14):
            yy = box[1] + 16 * s + k * 9 * s
            d.line([(box[0] + 14 * s, yy), (box[2] - 14 * s, yy)], fill=(36, 36, 39), width=max(1, int(2 * s)))
        d.rounded_rectangle(box, int(50 * s), outline=(58, 58, 62), width=int(4 * s))
        d.ellipse((x - 8 * s, box[3] - 32 * s, x + 8 * s, box[3] - 16 * s), fill=accent)
        d.rectangle((x - 10 * s, cy - 26 * s, x + 10 * s, cy), fill=accent)


def latte_top(img, cx, cy):
    d = ImageDraw.Draw(img)
    shadow(img, (cx - 130 * S, cy - 130 * S, cx + 130 * S, cy + 130 * S), 130 * S, 16, 60, (8, 14))
    d.ellipse((cx - 130 * S, cy - 130 * S, cx + 130 * S, cy + 130 * S), fill=(250, 248, 244), outline=(225, 220, 212), width=2 * S)
    d.ellipse((cx - 88 * S, cy - 88 * S, cx + 88 * S, cy + 88 * S), fill=(255, 255, 255), outline=(222, 216, 206), width=3 * S)
    d.rounded_rectangle((cx + 82 * S, cy - 18 * S, cx + 128 * S, cy + 18 * S), 16 * S, outline=(222, 216, 206), width=9 * S)
    d.ellipse((cx - 74 * S, cy - 74 * S, cx + 74 * S, cy + 74 * S), fill=(150, 98, 60))
    # latte-art heart
    c = (246, 232, 212)
    d.ellipse((cx - 38 * S, cy - 34 * S, cx + 2 * S, cy + 4 * S), fill=c)
    d.ellipse((cx - 2 * S, cy - 34 * S, cx + 38 * S, cy + 4 * S), fill=c)
    d.polygon([(cx - 36 * S, cy - 6 * S), (cx + 36 * S, cy - 6 * S), (cx, cy + 40 * S)], fill=c)


def notebook_top(img, box):
    x0, y0, x1, y1 = box
    shadow(img, box, 10 * S, 14, 60, (6, 12))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, 10 * S, fill=(252, 251, 248), outline=(220, 214, 204), width=2 * S)
    for k in range(1, 12):
        y = y0 + 34 * S + k * 26 * S
        if y < y1 - 20 * S:
            d.line([(x0 + 40 * S, y), (x1 - 24 * S, y)], fill=(214, 222, 232), width=2 * S)
    for k in range(9):
        y = y0 + 30 * S + k * 38 * S
        if y < y1 - 20 * S:
            d.ellipse((x0 - 8 * S, y, x0 + 12 * S, y + 20 * S), outline=(150, 150, 155), width=4 * S)
    # pen
    px, py = x1 - 40 * S, y0 + 40 * S
    d.line([(px, py), (px - 30 * S, y1 - 30 * S)], fill=(30, 30, 34), width=14 * S)
    d.line([(px - 27 * S, y1 - 60 * S), (px - 30 * S, y1 - 30 * S)], fill=(176, 146, 92), width=14 * S)


def window_light(img, alpha=55):
    """diagonal soft window-frame shadows"""
    m = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(m)
    for k in range(4):
        x = -400 * S + k * 520 * S
        d.polygon([(x, 0), (x + 120 * S, 0), (x + 120 * S + 700 * S, H), (x + 700 * S, H)], fill=alpha)
    img.paste((120, 92, 60), mask=m.filter(ImageFilter.GaussianBlur(40 * S)))


# ------------------------------------------------------------------ concepts
def clean():
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw_can(img, 345 * S, 290 * S)
    draw_laptop(img, (550 * S, 230 * S, 1370 * S, 805 * S))
    draw_case(img, (1460 * S, 450 * S, 1735 * S, 680 * S))
    title(img, "playlist.", 860, (12, 12, 14))
    return img


def morning():
    img = vgrad(W, H, (240, 228, 210), (226, 210, 186))
    window_light(img)
    notebook_top(img, (150 * S, 190 * S, 450 * S, 590 * S))
    latte_top(img, 330 * S, 770 * S)
    draw_laptop(img, (600 * S, 200 * S, 1360 * S, 735 * S))
    headphones_top(img, 1610 * S, 420 * S, 0.95)
    title(img, "morning focus.", 830, (58, 40, 26))
    return img


def night():
    img = vgrad(W, H, (14, 16, 24), (6, 7, 11))
    d = ImageDraw.Draw(img)
    desk_y = 800 * S
    # desk surface
    img.paste(vgrad(W, H - desk_y, (34, 30, 30), (16, 14, 15)), (0, desk_y))
    # screen glow onto wall and desk
    soft(img, lambda dd, a: dd.ellipse((560 * S, 120 * S, 1360 * S, 900 * S), fill=a), 120, (60, 110, 200), 110)
    soft(img, lambda dd, a: dd.ellipse((520 * S, 780 * S, 1400 * S, 980 * S), fill=a), 60, (80, 130, 220), 90)
    d = ImageDraw.Draw(img)
    # laptop front view
    sx0, sy0, sx1, sy1 = 620 * S, 320 * S, 1300 * S, 752 * S
    d.rounded_rectangle((sx0 - 14 * S, sy0 - 14 * S, sx1 + 14 * S, sy1 + 22 * S), 18 * S, fill=(22, 23, 27), outline=(60, 62, 70), width=2 * S)
    scr = vgrad(sx1 - sx0, sy1 - sy0, (24, 30, 48), (14, 18, 30))
    sd = ImageDraw.Draw(scr)
    rng = np.random.default_rng(3)
    palette = [(120, 170, 255), (190, 140, 255), (120, 220, 170), (240, 200, 120), (150, 155, 170)]
    y = 26 * S
    while y < (sy1 - sy0) - 30 * S:
        x = 30 * S + int(rng.integers(0, 4)) * 30 * S
        for _ in range(int(rng.integers(1, 5))):
            w = int(rng.integers(40, 170)) * S
            if x + w > (sx1 - sx0) - 30 * S:
                break
            sd.rounded_rectangle((x, y, x + w, y + 9 * S), 4 * S, fill=palette[int(rng.integers(0, len(palette)))])
            x += w + 14 * S
        y += 24 * S
    img.paste(scr, (sx0, sy0))
    # base
    d.polygon([(sx0 - 90 * S, sy1 + 22 * S), (sx1 + 90 * S, sy1 + 22 * S), (sx1 + 130 * S, desk_y + 6 * S), (sx0 - 130 * S, desk_y + 6 * S)], fill=(52, 54, 60))
    d.rectangle((sx0 - 130 * S, desk_y + 6 * S, sx1 + 130 * S, desk_y + 18 * S), fill=(34, 35, 40))
    d.line([(sx0 - 130 * S, desk_y + 6 * S), (sx1 + 130 * S, desk_y + 6 * S)], fill=(110, 140, 200), width=2 * S)
    # mug (left) with steam
    mx, mt = 360 * S, 600 * S
    d.rounded_rectangle((mx - 75 * S, mt, mx + 75 * S, desk_y + 10 * S), 18 * S, fill=(232, 228, 222))
    d.rounded_rectangle((mx + 62 * S, mt + 40 * S, mx + 120 * S, mt + 140 * S), 30 * S, outline=(232, 228, 222), width=16 * S)
    d.rectangle((mx - 75 * S, mt, mx + 75 * S, mt + 16 * S), fill=(200, 196, 190))
    for k, off in enumerate((-30, 5, 38)):
        pts = [(mx + off * S + 14 * S * math.sin(t / 7 + k), mt - 20 * S - t * 4 * S) for t in range(0, 36)]
        soft(img, lambda dd, a, p=pts: dd.line(p, fill=a, width=6 * S), 5, (200, 200, 210), 90)
    d = ImageDraw.Draw(img)
    # headphones (right) lying on desk, side view
    hx, hy = 1620 * S, desk_y
    d.arc((hx - 170 * S, hy - 300 * S, hx + 170 * S, hy + 40 * S), 185, 355, fill=(48, 48, 52), width=26 * S)
    for side in (-1, 1):
        x = hx + side * 160 * S
        d.rounded_rectangle((x - 58 * S, hy - 150 * S, x + 58 * S, hy + 4 * S), 46 * S, fill=(24, 24, 26), outline=(70, 72, 80), width=3 * S)
        for k in range(10):
            yy = hy - 128 * S + k * 11 * S
            d.line([(x - 38 * S, yy), (x + 38 * S, yy)], fill=(34, 34, 37), width=2 * S)
        d.ellipse((x - 7 * S, hy - 30 * S, x + 7 * S, hy - 16 * S), fill=(176, 146, 92))
    # rim light on headphones from the screen
    soft(img, lambda dd, a: dd.arc((hx - 170 * S, hy - 300 * S, hx + 170 * S, hy + 40 * S), 190, 260, fill=a, width=6 * S), 3, (110, 150, 230), 160)
    title(img, "late night focus.", 880, (236, 238, 245), 104)
    return img


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("clean", clean), ("night", night), ("morning", morning)):
        fn().resize((1920, 1080), Image.LANCZOS).save(os.path.join(OUT, f"cover_{name}.png"))
        print("saved", name)


def dumbbell_top(img, cx, cy, ang=-12):
    layer = Image.new("RGBA", (900 * S, 400 * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    mx, my = 450 * S, 200 * S
    d.rounded_rectangle((mx - 230 * S, my - 16 * S, mx + 230 * S, my + 16 * S), 10 * S, fill=(150, 152, 158))
    for k in range(-200, 200, 14):
        d.line([(mx + k * S, my - 14 * S), (mx + k * S + 6 * S, my + 14 * S)], fill=(120, 122, 128), width=2 * S)
    for side in (-1, 1):
        for j, (w, h) in enumerate(((60, 230), (48, 190))):
            x = mx + side * (250 + j * 58) * S
            d.rounded_rectangle((x - w * S // 2, my - h * S // 2, x + w * S // 2, my + h * S // 2), 14 * S, fill=(28, 28, 30), outline=(60, 60, 64), width=3 * S)
    layer = layer.rotate(ang, resample=Image.BICUBIC, expand=False)
    sh = Image.new("L", layer.size, 0)
    sh.paste(layer.split()[3].point(lambda a: 110 if a else 0))
    sh = sh.filter(ImageFilter.GaussianBlur(16 * S))
    x0, y0 = cx - 450 * S, cy - 200 * S
    img.paste((0, 0, 0), (x0 + 10 * S, y0 + 22 * S), sh)
    img.paste(layer, (x0, y0), layer)


def shaker_top(img, cx, cy):
    d = ImageDraw.Draw(img)
    shadow(img, (cx - 95 * S, cy - 95 * S, cx + 95 * S, cy + 95 * S), 95 * S, 14, 90, (8, 16))
    d.ellipse((cx - 95 * S, cy - 95 * S, cx + 95 * S, cy + 95 * S), fill=(18, 18, 20), outline=(60, 60, 66), width=4 * S)
    d.ellipse((cx - 70 * S, cy - 70 * S, cx + 70 * S, cy + 70 * S), fill=(30, 30, 33))
    d.ellipse((cx - 22 * S, cy - 22 * S, cx + 22 * S, cy + 22 * S), fill=(232, 92, 40))


def gym():
    base = vgrad(W, H, (58, 58, 60), (36, 36, 38)).convert("L")
    noise = Image.effect_noise((W, H), 28).filter(ImageFilter.GaussianBlur(1.2 * S))
    img = Image.blend(base, noise, 0.18).convert("RGB")
    img = Image.blend(img, vgrad(W, H, (52, 50, 50), (30, 30, 32)), 0.55)
    dumbbell_top(img, 900 * S, 420 * S)
    headphones_top(img, 1600 * S, 380 * S, 0.9)
    shaker_top(img, 300 * S, 470 * S)
    title(img, "gym focus.", 820, (244, 244, 246), 124)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((W // 2 - 70 * S, 980 * S, W // 2 + 70 * S, 990 * S), 5 * S, fill=(232, 92, 40))
    return img


if __name__ == "__main__":
    gym().resize((1920, 1080), Image.LANCZOS).save(os.path.join(OUT, "cover_gym.png"))
    print("saved gym")
