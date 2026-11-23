#!/usr/bin/env python3
"""Рендер анимированного сериала «ЭХО»: 5 серий × 90 с, 9:16.

Процедурная графика (PIL) + синтезированная музыка (numpy) -> ffmpeg.
Запуск:  python3 video/render.py            # все серии + полный выпуск
         python3 video/render.py 1 3        # только серии 1 и 3
"""
import math, os, random, subprocess, sys, textwrap, wave
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from episodes import EPISODES

W, H, FPS, SHOT = 540, 960, 24, 10
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FD = "/usr/share/fonts/truetype/dejavu/"
F_SUB = ImageFont.truetype(FD + "DejaVuSans.ttf", 27)
F_NAME = ImageFont.truetype(FD + "DejaVuSans-Bold.ttf", 23)
F_BIG = ImageFont.truetype(FD + "DejaVuSans-Bold.ttf", 110)
F_MID = ImageFont.truetype(FD + "DejaVuSans-Bold.ttf", 34)
F_SM = ImageFont.truetype(FD + "DejaVuSans-Bold.ttf", 22)
F_CAP = ImageFont.truetype(FD + "DejaVuSans-Bold.ttf", 46)
F_HOOK = ImageFont.truetype(FD + "DejaVuSans-Bold.ttf", 38)

BLUE, GOLD, RED = (90, 180, 255), (255, 200, 90), (255, 50, 60)
CHARS = {
    "lena": dict(body=(18, 40, 26), rim=(90, 235, 130)),
    "max": dict(body=(34, 26, 22), rim=(245, 150, 60)),
    "vein": dict(body=(205, 205, 215), rim=(255, 255, 255)),
    "agent": dict(body=(12, 12, 16), rim=(200, 200, 215)),
    "father": dict(body=(30, 24, 12), rim=GOLD),
    "crowd": dict(body=(10, 8, 18), rim=(70, 55, 100)),
}
SPEAKER = {"Лена": (90, 235, 130), "Эхо": BLUE, "Макс": (245, 150, 60),
           "Вейн": (235, 235, 250), "Отец": GOLD}


def tri(t, a, b):
    return max(0.0, min(1.0, (t - a) / (b - a)))


def ease(x):
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- фоны
def grad(top, bot):
    a = np.linspace(0, 1, H)[:, None, None]
    arr = np.array(top) * (1 - a) + np.array(bot) * a
    return Image.fromarray(np.broadcast_to(arr, (H, W, 3)).astype(np.uint8))


SKIES = {
    "night": grad((6, 5, 22), (48, 14, 74)),
    "dawn": grad((60, 50, 120), (255, 165, 105)),
    "dark": grad((5, 5, 10), (20, 18, 30)),
    "server": grad((3, 8, 18), (8, 22, 44)),
    "bar": grad((36, 12, 22), (12, 6, 8)),
    "storm": grad((10, 10, 24), (30, 26, 50)),
}


def make_skyline(seed, base_y, hmin, hmax, col, win_p, width=W * 3):
    rnd = random.Random(seed)
    im = Image.new("RGBA", (width, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = 0
    while x < width:
        bw, bh = rnd.randint(40, 110), rnd.randint(hmin, hmax)
        top = base_y - bh
        d.rectangle([x, top, x + bw, H], fill=col + (255,))
        if rnd.random() < 0.3:
            ax = x + bw // 2
            d.line([(ax, top), (ax, top - rnd.randint(20, 60))], fill=col + (255,), width=3)
            d.ellipse([ax - 3, top - 63, ax + 3, top - 57], fill=(255, 60, 60, 220))
        for wy in range(top + 10, base_y - 5, 14):
            for wx in range(x + 6, x + bw - 6, 10):
                if rnd.random() < win_p:
                    c = rnd.choice([(255, 210, 120), (90, 220, 255), (255, 80, 200), (255, 240, 200)])
                    d.rectangle([wx, wy, wx + 4, wy + 6], fill=c + (rnd.randint(120, 230),))
        if rnd.random() < 0.25:
            c = rnd.choice([(255, 60, 200), (60, 230, 255), (255, 150, 40)])
            sy = top + rnd.randint(20, 80)
            d.rectangle([x + 4, sy, x + 12, sy + rnd.randint(60, 140)], fill=c + (230,))
        x += bw + rnd.randint(0, 8)
    return im


FAR = make_skyline(1, int(H * 0.78), 150, 420, (22, 16, 44), 0.12)
NEAR = make_skyline(2, H, 180, 480, (8, 6, 16), 0.16)


def paste_layer(img, L, off):
    off = int(off) % (L.width - W)
    c = L.crop((off, 0, off + W, H))
    img.paste(c, (0, 0), c)


def _vignette():
    yy, xx = np.mgrid[0:H, 0:W]
    dd = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    a = np.clip((dd - 0.75) * 260, 0, 190).astype(np.uint8)
    v = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    v.putalpha(Image.fromarray(a))
    return v


VIG = _vignette()


def _bokeh(seed, cols):
    rnd = random.Random(seed)
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(26):
        x, y, r = rnd.randint(0, W), rnd.randint(0, H), rnd.randint(20, 70)
        d.ellipse([x - r, y - r, x + r, y + r], fill=rnd.choice(cols) + (rnd.randint(25, 60),))
    return im


BOKEH = _bokeh(7, [(255, 80, 200), (60, 200, 255), (255, 190, 90)])


def rain(d, t, n=240, wind=70, box=None):
    rnd = np.random.default_rng(3)
    x0, v, ln, y0 = rnd.uniform(0, W, n), rnd.uniform(900, 1500, n), rnd.uniform(12, 28, n), rnd.uniform(0, H, n)
    y = (y0 + v * t) % H
    x = (x0 + wind * t * 0.4) % W
    for i in range(n):
        if box and not (box[0] < x[i] < box[2] and box[1] < y[i] < box[3]):
            continue
        d.line([(x[i], y[i]), (x[i] - wind * 0.02 * ln[i] / 10, y[i] - ln[i])], fill=(160, 185, 235, 85), width=1)


# ---------------------------------------------------------------- объекты
@lru_cache(None)
def orb_sprite(r, col, lvl):
    S = int(r * 5)
    c = S / 2
    yy, xx = np.mgrid[0:S, 0:S]
    dd = np.hypot(xx - c, yy - c) / r
    glow = np.exp(-(dd ** 2) * 0.55) * (lvl / 10)
    core = np.clip(1.0 - dd, 0, 1) ** 1.3 * (0.4 + 0.6 * lvl / 10)
    colf = np.array(col) / 255.0
    rgb = colf[None, None, :] * (1 - core[..., None]) + core[..., None]
    a = np.clip(glow * 0.85 + core, 0, 1)
    arr = np.dstack([np.clip(rgb, 0, 1) * 255, a * 255]).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def draw_orb(img, d, cx, cy, r, t, col=BLUE, threads=True, lvl=10):
    if r < 2:
        return
    rr = max(2, int(r * (1 + 0.08 * math.sin(t * 3)) / 2) * 2)
    if threads and r > 12:
        for i in range(7):
            a = t * 0.6 + i * 2 * math.pi / 7
            pts = [(cx + math.cos(a + k * 0.18) * rr * (0.9 + k * 0.35),
                    cy + math.sin(a + k * 0.18) * rr * (0.9 + k * 0.35) + math.sin(t * 2 + k + i) * 4)
                   for k in range(6)]
            d.line(pts, fill=GOLD + (int(140 * lvl / 10),), width=2)
    sp = orb_sprite(rr, col, lvl)
    img.paste(sp, (int(cx - sp.width / 2), int(cy - sp.height / 2)), sp)
    if threads and r >= 10 and lvl >= 4:  # мордочка Эхо
        ec = (20, 40, 90) if col == BLUE else (90, 10, 20)
        for sg in (-1, 1):
            ex_ = cx + sg * 0.32 * rr
            if (t % 3.7) < 0.13:
                d.line([(ex_ - 0.12 * rr, cy - 0.05 * rr), (ex_ + 0.12 * rr, cy - 0.05 * rr)], fill=ec, width=max(2, int(0.07 * rr)))
            else:
                d.ellipse([ex_ - 0.1 * rr, cy - 0.24 * rr, ex_ + 0.1 * rr, cy + 0.12 * rr], fill=ec)
        if talking("echo", t):
            o = abs(math.sin(t * 15)) * 0.16 * rr + 0.04 * rr
            d.ellipse([cx - 0.12 * rr, cy + 0.32 * rr - o / 2, cx + 0.12 * rr, cy + 0.32 * rr + o / 2], fill=ec)
        else:
            d.arc([cx - 0.14 * rr, cy + 0.12 * rr, cx + 0.14 * rr, cy + 0.36 * rr], 20, 160, fill=ec, width=max(2, int(0.05 * rr)))


def limb(d, pts, b, r, w):
    d.line(pts, fill=r, width=w + 4, joint="curve")
    d.line(pts, fill=b, width=w, joint="curve")


OUTL = (28, 20, 44)
SKIN = (255, 218, 180)
LOOK = {
    "lena": dict(hair=(28, 26, 36), top=(52, 150, 86), pants=(44, 44, 66), style="bob", scar=True),
    "max": dict(hair=(226, 112, 40), top=(128, 128, 140), pants=(52, 46, 62), style="spiky", beard=True, glasses=True),
    "vein": dict(hair=(196, 196, 206), top=(242, 242, 246), pants=(226, 226, 232), style="bun"),
    "father": dict(hair=(206, 206, 210), top=(146, 98, 62), pants=(72, 62, 56), style="short", mustache=True),
    "agent": dict(hair=(18, 18, 24), top=(22, 22, 28), pants=(22, 22, 28), style="hood", mask=True),
}
CROWD_TOPS = [(200, 80, 80), (80, 120, 200), (230, 180, 60), (120, 80, 160), (70, 160, 150), (220, 120, 170)]
CROWD_HAIR = [(30, 25, 25), (120, 70, 30), (220, 190, 100), (60, 40, 20)]
SPEAK = dict(char=None, t0=0.0, t1=0.0)
CHAR_OF = {"Лена": "lena", "Макс": "max", "Вейн": "vein", "Отец": "father", "Эхо": "echo"}


def talking(ch, t):
    return SPEAK["char"] == ch and SPEAK["t0"] < t < SPEAK["t1"]


def lighter(c, k=0.25):
    return tuple(int(v + (255 - v) * k) for v in c)


def person(img, d, x, y, s, ch, t=0.0, run=False, kneel=False, reach=False, pack=False, sit=False,
           emo="neutral", talk=None, seed=0):
    """Мультяшный персонаж: большая круглая голова, лицо, причёска, одежда, толстый контур. y — уровень ног."""
    if ch == "crowd":
        r = random.Random(seed)
        L = dict(hair=r.choice(CROWD_HAIR), top=r.choice(CROWD_TOPS), pants=(50, 50, 70), style=r.choice(["short", "bob", "spiky"]))
    else:
        L = LOOK[ch]
    if talk is None:
        talk = talking(ch, t)
    ow = max(2, int(3.2 * s))
    R = 48 * s
    hip_y, sh_y = y - 52 * s, y - 122 * s
    hx, hy = x, y - 172 * s
    if run:
        hy += abs(math.sin(t * 13)) * -4 * s

    def lim(pts, col, w):
        d.line(pts, fill=OUTL, width=int(w + 2 * ow), joint="curve")
        for p_ in (pts[0], pts[-1]):
            rr = (w + 2 * ow) / 2
            d.ellipse([p_[0] - rr, p_[1] - rr, p_[0] + rr, p_[1] + rr], fill=OUTL)
        d.line(pts, fill=col, width=int(w), joint="curve")

    def shoe(fx, fy):
        d.ellipse([fx - 13 * s, fy - 9 * s, fx + 13 * s, fy + 4 * s], fill=(46, 34, 34), outline=OUTL, width=ow)

    def hand(px, py):
        d.ellipse([px - 8 * s, py - 8 * s, px + 8 * s, py + 8 * s], fill=SKIN, outline=OUTL, width=ow)

    if pack:
        draw_orb(img, d, x - 42 * s, y - 108 * s, 14 * s, t, threads=False)
    lw = 15 * s
    if run:
        ph = math.sin(t * 13)
        for sg in (1, -1):
            f = (x + sg * ph * 30 * s, y - 4 * s)
            lim([(x + sg * 8 * s, hip_y), (x + sg * ph * 18 * s, y - 28 * s), f], L["pants"], lw)
            shoe(*f)
    elif kneel:
        lim([(x - 8 * s, hip_y), (x - 12 * s, y - 8 * s), (x - 42 * s, y - 6 * s)], L["pants"], lw)
        lim([(x + 8 * s, hip_y), (x + 34 * s, hip_y + 4 * s), (x + 34 * s, y - 6 * s)], L["pants"], lw)
        shoe(x + 38 * s, y - 4 * s)
    elif sit:
        for sg in (1, -1):
            lim([(x + sg * 8 * s, hip_y), (x + sg * 8 * s + 34 * s, hip_y), (x + sg * 8 * s + 34 * s, y - 6 * s)], L["pants"], lw)
            shoe(x + sg * 8 * s + 38 * s, y - 4 * s)
    else:
        for sg in (1, -1):
            lim([(x + sg * 11 * s, hip_y), (x + sg * 12 * s, y - 8 * s)], L["pants"], lw)
            shoe(x + sg * 14 * s, y - 4 * s)
    # руки (сзади тела не рисуем — они поверх, «дугой», как в мультфильмах)
    d.rounded_rectangle([x - 31 * s, sh_y, x + 31 * s, hip_y + 8 * s], radius=int(18 * s), fill=L["top"], outline=OUTL, width=ow)
    if ch == "lena":
        d.line([(x, sh_y + 6 * s), (x, hip_y + 4 * s)], fill=lighter(L["top"], 0.45), width=max(1, int(2 * s)))
        d.rectangle([x - 31 * s + ow, hip_y - 2 * s, x + 31 * s - ow, hip_y + 4 * s], fill=lighter(L["top"], -0.0))
    elif ch in ("vein", "max", "agent", "father"):
        lap = (255, 255, 255) if ch == "agent" else lighter(L["top"], 0.35)
        d.polygon([(x - 13 * s, sh_y + 2 * s), (x, sh_y + 30 * s), (x + 13 * s, sh_y + 2 * s)], fill=lap, outline=OUTL)
        if ch == "agent":
            d.line([(x, sh_y + 12 * s), (x, sh_y + 30 * s)], fill=(20, 20, 26), width=max(2, int(4 * s)))
    else:
        for sg in (-1, 1):
            d.line([(x + sg * 8 * s, sh_y + 6 * s), (x + sg * 9 * s, sh_y + 30 * s)], fill=(255, 255, 255), width=max(1, int(2.5 * s)))
        d.rounded_rectangle([x - 18 * s, sh_y + 40 * s, x + 18 * s, hip_y], radius=int(5 * s), outline=OUTL, width=max(1, ow - 1))
    aw = 12 * s
    if run:
        ph = math.sin(t * 13)
        for sg in (1, -1):
            e_ = (x + sg * 30 * s - sg * 0 - ph * sg * 22 * s, sh_y + 56 * s)
            lim([(x + sg * 26 * s, sh_y + 10 * s), (x + sg * 40 * s - ph * sg * 10 * s, sh_y + 34 * s), e_], L["top"], aw)
            hand(*e_)
    else:
        for sg in (-1, 1):
            if reach and sg == 1:
                e_ = (x + 88 * s, sh_y + 18 * s)
                lim([(x + 26 * s, sh_y + 10 * s), (x + 58 * s, sh_y + 16 * s), e_], L["top"], aw)
            else:
                e_ = (x + sg * 38 * s, hip_y - 4 * s)
                lim([(x + sg * 26 * s, sh_y + 10 * s), (x + sg * 46 * s, sh_y + 40 * s), e_], L["top"], aw)
            hand(*e_)
    # голова
    st = L["style"]
    if st == "bob":
        d.rounded_rectangle([hx - 1.14 * R, hy - 1.05 * R, hx + 1.14 * R, hy + 0.8 * R], radius=int(0.6 * R), fill=L["hair"], outline=OUTL, width=ow)
    if st == "hood":
        d.ellipse([hx - 1.2 * R, hy - 1.2 * R, hx + 1.2 * R, hy + 1.1 * R], fill=L["hair"], outline=OUTL, width=ow)
    if st == "bun":
        d.ellipse([hx - 0.36 * R, hy - 1.5 * R, hx + 0.36 * R, hy - 0.82 * R], fill=L["hair"], outline=OUTL, width=ow)
    if L.get("mask"):
        d.ellipse([hx - 0.85 * R, hy - 0.85 * R, hx + 0.85 * R, hy + 0.9 * R], fill=(236, 236, 242), outline=OUTL, width=ow)
        for sg in (-1, 1):
            d.polygon([(hx + sg * 0.12 * R, hy + 0.02 * R), (hx + sg * 0.55 * R, hy - 0.1 * R), (hx + sg * 0.5 * R, hy + 0.08 * R)], fill=(10, 10, 14))
        if int(t * 3 + seed) % 2:
            d.ellipse([hx - 0.05 * R, hy + 0.45 * R, hx + 0.05 * R, hy + 0.55 * R], fill=RED)
        return hx, hy, R
    d.ellipse([hx - R, hy - R, hx + R, hy + R], fill=SKIN, outline=OUTL, width=ow)
    if st == "bob":
        d.chord([hx - 1.06 * R, hy - 1.14 * R, hx + 1.06 * R, hy + 0.5 * R], 180, 360, fill=L["hair"], outline=OUTL, width=ow)
    elif st == "spiky":
        for k in range(5):
            a = math.radians(222 + k * 24)
            a1, a2 = a - 0.26, a + 0.26
            tip = (hx + math.cos(a) * 1.4 * R, hy + math.sin(a) * 1.38 * R)
            d.polygon([(hx + math.cos(a1) * 0.85 * R, hy + math.sin(a1) * 0.85 * R), tip,
                       (hx + math.cos(a2) * 0.85 * R, hy + math.sin(a2) * 0.85 * R)], fill=L["hair"], outline=OUTL, width=ow)
        d.chord([hx - 1.04 * R, hy - 1.1 * R, hx + 1.04 * R, hy + 0.3 * R], 180, 360, fill=L["hair"], outline=OUTL, width=ow)
    else:
        d.chord([hx - 1.03 * R, hy - 1.07 * R, hx + 1.03 * R, hy + 0.1 * R], 180, 360, fill=L["hair"], outline=OUTL, width=ow)
    # лицо
    ex, ey = 0.36 * R, hy + 0.1 * R
    blink = ((t + seed * 0.37) % 3.3) < 0.13
    for sg in (-1, 1):
        cx = hx + sg * ex
        if blink:
            d.line([(cx - 0.12 * R, ey), (cx + 0.12 * R, ey)], fill=OUTL, width=max(2, int(0.06 * R)))
        elif emo == "surprised":
            d.ellipse([cx - 0.17 * R, ey - 0.2 * R, cx + 0.17 * R, ey + 0.2 * R], fill=(255, 255, 255), outline=OUTL, width=max(2, int(0.05 * R)))
            d.ellipse([cx - 0.07 * R, ey - 0.07 * R, cx + 0.07 * R, ey + 0.07 * R], fill=OUTL)
        else:
            d.ellipse([cx - 0.1 * R, ey - 0.16 * R, cx + 0.1 * R, ey + 0.16 * R], fill=OUTL)
            d.ellipse([cx - 0.06 * R, ey - 0.11 * R, cx - 0.0 * R, ey - 0.04 * R], fill=(255, 255, 255))
        yb = ey - 0.34 * R
        din, dout = {"angry": (0.1, -0.04), "sad": (-0.1, 0.04), "worried": (-0.1, 0.04), "surprised": (-0.1, -0.1)}.get(emo, (0, 0))
        d.line([(hx + sg * (ex + 0.17 * R), yb + dout * R), (hx + sg * (ex - 0.14 * R), yb + din * R)], fill=OUTL, width=max(2, int(0.08 * R)))
        d.ellipse([cx + sg * 0.08 * R - 0.12 * R, ey + 0.22 * R, cx + sg * 0.08 * R + 0.12 * R, ey + 0.32 * R], fill=(255, 140, 140, 90))
    if L.get("scar"):
        d.line([(hx - ex - 0.14 * R, ey - 0.52 * R), (hx - ex + 0.06 * R, ey - 0.36 * R)], fill=(200, 110, 110), width=max(2, int(0.05 * R)))
    if L.get("glasses"):
        gw = max(2, int(0.05 * R))
        for sg in (-1, 1):
            d.ellipse([hx + sg * ex - 0.24 * R, ey - 0.22 * R, hx + sg * ex + 0.24 * R, ey + 0.22 * R], outline=(60, 60, 74), width=gw)
        d.line([(hx - ex + 0.24 * R, ey - 0.04 * R), (hx + ex - 0.24 * R, ey - 0.04 * R)], fill=(60, 60, 74), width=gw)
    if L.get("beard"):
        d.chord([hx - 0.72 * R, hy + 0.2 * R, hx + 0.72 * R, hy + 1.02 * R], 0, 180, fill=L["hair"], outline=OUTL, width=ow)
    my = hy + 0.56 * R
    mw = max(2, int(0.07 * R))
    if L.get("mustache"):
        for sg in (-1, 1):
            d.chord([hx + (sg - 1) * 0.2 * R, my - 0.2 * R, hx + (sg + 1) * 0.2 * R, my + 0.02 * R], 180, 360, fill=L["hair"], outline=OUTL)
    if talk:
        o = abs(math.sin(t * 15)) * 0.22 * R + 0.05 * R
        d.ellipse([hx - 0.16 * R, my - o / 2, hx + 0.16 * R, my + o / 2], fill=(130, 34, 46), outline=OUTL, width=mw)
    elif emo == "worried":
        pts = [(hx - 0.22 * R + k * 0.44 * R / 12, my + math.sin(k * math.pi / 3) * 0.05 * R) for k in range(13)]
        d.line(pts, fill=OUTL, width=mw)
    elif emo == "sad":
        d.arc([hx - 0.2 * R, my - 0.02 * R, hx + 0.2 * R, my + 0.28 * R], 200, 340, fill=OUTL, width=mw)
    elif emo == "surprised":
        d.ellipse([hx - 0.08 * R, my - 0.1 * R, hx + 0.08 * R, my + 0.12 * R], fill=(130, 34, 46), outline=OUTL, width=mw)
    elif emo in ("happy", "smug"):
        d.chord([hx - 0.22 * R, my - 0.22 * R, hx + 0.22 * R, my + 0.16 * R], 10, 170, fill=(130, 34, 46), outline=OUTL, width=mw) if emo == "happy" \
            else d.arc([hx - 0.24 * R, my - 0.26 * R, hx + 0.12 * R, my + 0.1 * R], 20, 150, fill=OUTL, width=mw)
    else:
        d.arc([hx - 0.15 * R, my - 0.14 * R, hx + 0.15 * R, my + 0.04 * R], 30, 150, fill=OUTL, width=mw)
    return hx, hy, R


def tower(img, d, cx, base, t, s=1.0, col=BLUE):
    top = base - int(560 * s)
    poly = [(cx - 50 * s, base), (cx - 16 * s, top + 40 * s), (cx, top), (cx + 16 * s, top + 40 * s), (cx + 50 * s, base)]
    d.polygon(poly, fill=(14, 14, 30), outline=col + (170,), width=2)
    for k in range(13):
        yy = top + 60 * s + k * 38 * s
        ww = 16 * s + (yy - top) / (base - top) * 34 * s
        d.line([(cx - ww + 4, yy), (cx + ww - 4, yy)], fill=col + (80,), width=1)
    for i, ch in enumerate("НЕКСУС"):
        d.text((cx - 7 * s, top + 120 * s + i * 30 * s), ch, font=F_SM, fill=col + (200,))
    pulse = 0.5 + 0.5 * math.sin(t * 3)
    draw_orb(img, d, cx, top, 14 * s * (1 + 0.4 * pulse), t, col=col, threads=False)


def drones(d, t, n=3):
    for i in range(n):
        x = W * (0.2 + 0.3 * i) + 60 * math.sin(t * 0.8 + i * 2)
        y = 250 + 45 * i + 20 * math.sin(t * 1.3 + i)
        sw = math.sin(t * 1.5 + i) * 130
        d.polygon([(x, y), (x + sw - 70, H * 0.95), (x + sw + 70, H * 0.95)], fill=(255, 40, 50, 38))
        d.line([(x - 32, y - 9), (x + 32, y - 9)], fill=(90, 90, 110), width=3)
        d.ellipse([x - 22, y - 8, x + 22, y + 8], fill=(20, 20, 28), outline=(110, 110, 130))
        if int(t * 4 + i) % 2:
            d.ellipse([x - 4, y - 3, x + 4, y + 5], fill=RED)


def grid(d, t, mode):
    a = 90 if mode == "on" else int(90 * (1 - tri(t, 0.5, 7)))
    if a <= 0:
        return
    rnd = random.Random(int(t * 8))
    vp = (W / 2, 380)
    for k in range(13):
        x = -300 + k * 95
        if mode == "break" and rnd.random() < tri(t, 0, 6):
            continue
        d.line([vp, (x, H)], fill=BLUE + (a,), width=1)
    for k in range(12):
        y = 420 + (k ** 1.6) * 8
        if mode == "break" and rnd.random() < tri(t, 0, 6):
            continue
        d.line([(0, y), (W, y)], fill=BLUE + (a,), width=1)


def threads_city(img, d, t, src=(W / 2, 170)):
    prog = ease(tri(t, 0, 6.5))
    rnd = random.Random(11)
    for i in range(28):
        ex, ey = rnd.uniform(-40, W + 40), rnd.uniform(520, 900)
        cx, cy = (src[0] + ex) / 2 + rnd.uniform(-120, 120), rnd.uniform(60, 260)
        n = max(2, int(30 * prog))
        pts = []
        for k in range(n):
            u = k / 29
            pts.append(((1 - u) ** 2 * src[0] + 2 * (1 - u) * u * cx + u * u * ex,
                        (1 - u) ** 2 * src[1] + 2 * (1 - u) * u * cy + u * u * ey))
        d.line(pts, fill=GOLD + (170,), width=2)
        px, py = pts[-1]
        d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 240, 200, 230))


_rnd = random.Random(21)
NODES = [(_rnd.uniform(60, 480), _rnd.uniform(200, 660)) for _ in range(34)]
EDGES = []
for i, a in enumerate(NODES):
    near = sorted(range(len(NODES)), key=lambda j: math.dist(a, NODES[j]))[1:3]
    EDGES += [(i, j) for j in near]
EDGES.sort(key=lambda e: math.dist(NODES[e[0]], (W / 2, H * 0.75)))


def network(d, t, red=None, col=GOLD):
    n = int(len(EDGES) * ease(tri(t, 0, 6)))
    for i, j in EDGES[:n]:
        d.line([NODES[i], NODES[j]], fill=col + (150,), width=2)
    for k, (x, y) in enumerate(NODES):
        if any(k in e for e in EDGES[:n]):
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col + (230,))
    if red is not None and t > 3:
        x, y = NODES[red]
        p = 0.5 + 0.5 * math.sin(t * 6)
        rr = 8 + 10 * p
        d.ellipse([x - rr * 2, y - rr * 2, x + rr * 2, y + rr * 2], fill=RED + (40,))
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=RED + (160,))
        d.text((x + 22, y - 14), "НЕКСУС", font=F_MID, fill=RED + (240,))


def persons(img, d, t, p):
    for q in p.get("who", []):
        x = (q.get("x", 0.5) + q.get("vx", 0) * t) * W
        person(img, d, x, q.get("y", 860), q.get("s", 1.4), q["c"], t,
               run=q.get("run", False), kneel=q.get("kneel", False), reach=q.get("reach", False),
               pack=q.get("pack", False), sit=q.get("sit", False), emo=q.get("emo", "neutral"),
               seed=int(q.get("x", 0.5) * 100))


def orb_from(img, d, t, p):
    o = p.get("orb")
    if not o:
        return
    g = ease(tri(t, *o.get("grow", (-1, 0))))
    lvl = o.get("lvl", 10)
    if "dim" in o:
        lvl = max(2, int(10 - 8 * tri(t, *o["dim"])))
    x0, y0 = o.get("from", (o["x"], o["y"]))
    x, y = x0 + (o["x"] - x0) * g, y0 + (o["y"] - y0) * g
    draw_orb(img, d, x, y, o["r"] * max(0.15, g), t, col=o.get("col", BLUE), lvl=lvl)


# ---------------------------------------------------------------- сцены
def sc_city(img, d, t, p):
    if p.get("sun"):
        sy = H * 0.66 - t * 14
        for rr, a in ((220, 30), (140, 60), (80, 140), (55, 255)):
            d.ellipse([W / 2 - rr, sy - rr, W / 2 + rr, sy + rr], fill=(255, 210, 140, a))
    pan = p.get("pan", 20)
    paste_layer(img, FAR, t * pan * 0.4 + p.get("off", 0))
    if p.get("tower"):
        tower(img, d, W / 2 + p.get("tx", 0), int(H * 0.82), t, p.get("ts", 1.0))
    paste_layer(img, NEAR, t * pan + 300 + p.get("off", 0))
    if p.get("drones"):
        drones(d, t)
    if p.get("grid"):
        grid(d, t, p["grid"])
    if p.get("threads"):
        threads_city(img, d, t)
    if p.get("spark") and t > 4:
        a = int(255 * tri(t, 4, 7))
        draw_orb(img, d, 400, 210, 6 + 2 * math.sin(t * 4), t, threads=False, lvl=max(2, a // 26))
    persons(img, d, t, p)
    orb_from(img, d, t, p)


def sc_interior(img, d, t, p):
    d.polygon([(0, 720), (W, 720), (W, H), (0, H)], fill=(14, 12, 20))
    wx0, wy0, wx1, wy1 = 320, 170, 500, 430
    d.rectangle([wx0, wy0, wx1, wy1], fill=(18, 24, 60))
    rnd = random.Random(5)
    for _ in range(40):
        x, y = rnd.randint(wx0, wx1 - 5), rnd.randint(wy0 + 60, wy1 - 5)
        d.rectangle([x, y, x + 3, y + 4], fill=rnd.choice([(255, 210, 120), (90, 220, 255), (255, 80, 200)]) + (150,))
    rain(d, t, n=300, box=(wx0, wy0, wx1, wy1))
    if p.get("car"):
        k = ease(tri(t, 0.5, 4.5))
        cx = wx1 + 80 - (wx1 - wx0 + 20) * k
        cy = wy1 - 18
        d.polygon([(cx - 70, cy), (cx - 60, cy - 22), (cx - 25, cy - 36), (cx + 30, cy - 36), (cx + 60, cy - 20), (cx + 70, cy)], fill=(4, 4, 6))
        if t < 6.5:
            d.polygon([(cx - 70, cy - 12), (cx - 170, cy - 40), (cx - 170, cy + 10)], fill=(255, 240, 180, 70))
    d.rectangle([wx0, wy0, wx1, wy1], outline=(40, 40, 60), width=6)
    d.line([((wx0 + wx1) / 2, wy0), ((wx0 + wx1) / 2, wy1)], fill=(40, 40, 60), width=4)
    d.line([(wx0, (wy0 + wy1) / 2), (wx1, (wy0 + wy1) / 2)], fill=(40, 40, 60), width=4)
    if p.get("server"):
        d.rectangle([220, 520, 320, 740], fill=(26, 26, 34), outline=(60, 60, 80), width=2)
        rnd = random.Random(int(t * 5))
        for k in range(10):
            yy = 535 + k * 20
            d.line([(232, yy), (300, yy)], fill=(45, 45, 60), width=2)
            c = (60, 255, 120) if rnd.random() < 0.6 else (255, 170, 40)
            if p.get("alive", True) or t > 3:
                d.ellipse([304, yy - 3, 310, yy + 3], fill=c)
    if p.get("photo"):
        d.rectangle([90, 560, 180, 660], fill=(110, 90, 60), outline=(70, 50, 30), width=5)
        person(img, d, 120, 650, 0.45, "father")
        person(img, d, 150, 650, 0.3, "lena")
    if p.get("screen"):
        fl = 0.55 + 0.45 * random.Random(int(t * 12)).random() if t < 5 else 1.0
        d.ellipse([110, 380, 340, 600], fill=(60, 140, 255, int(40 * fl)))
        d.rectangle([150, 430, 300, 520], fill=(40, 110, 220, int(210 * fl)), outline=(80, 80, 100), width=4)
        for yy in range(434, 518, 6):
            d.line([(154, yy), (296, yy)], fill=(20, 60, 140, 120))
    if p.get("sparks") and 1.0 < t < 6.0:
        rnd = random.Random(int(t * 15))
        for _ in range(14):
            a, ln = rnd.uniform(0, 2 * math.pi), rnd.uniform(10, 45)
            d.line([(300, 560), (300 + math.cos(a) * ln, 560 + math.sin(a) * ln)], fill=(255, 230, 120), width=2)
    persons(img, d, t, p)
    if p.get("flash"):
        q = p["who"][0]
        hx, hy = (q["x"] + q.get("vx", 0) * t) * W + 30, q.get("y", 860) - 150
        d.polygon([(hx, hy), (W + 40, 480), (W + 40, 820)], fill=(255, 240, 200, 34))
        rnd = np.random.default_rng(9)
        for i in range(60):
            x = (rnd.uniform(hx, W) + 8 * math.sin(t + i)) % W
            y = rnd.uniform(470, 820) + 6 * math.cos(t * 0.7 + i)
            d.point((x, y), fill=(255, 250, 220, 160))
    if p.get("map"):
        network(d, t)
    orb_from(img, d, t, p)
    if p.get("dark_at") is not None and t > p["dark_at"]:
        d.rectangle([0, 0, W, H], fill=(0, 0, 0, int(225 * tri(t, p["dark_at"], p["dark_at"] + 0.4))))


def sc_photo(img, d, t, p):
    z = 1 + t * 0.02
    x0, y0, x1, y1 = W / 2 - 150 * z, 300 - 30 * z, W / 2 + 150 * z, 640 + 30 * z
    d.rectangle([x0 - 16, y0 - 16, x1 + 16, y1 + 16], fill=(70, 50, 30))
    d.rectangle([x0, y0, x1, y1], fill=(150, 125, 85))
    d.rectangle([x0, y0, x1, y1 - 120 * z], fill=(170, 150, 110))
    person(img, d, W / 2 - 40 * z, y1 - 20, 1.6 * z, "father")
    person(img, d, W / 2 + 55 * z, y1 - 20, 1.0 * z, "lena")
    d.rectangle([x0, y0, x1, y1], fill=(120, 90, 50, 50))
    d.polygon([(0, 0), (260, 0), (W, H), (W - 200, H)], fill=(255, 240, 200, 22))
    d.rectangle([400, 700, 520, 900], fill=(26, 26, 34), outline=(60, 60, 80), width=2)


def sc_map(img, d, t, p):
    paste_layer(img, FAR, 100 + t * 6)
    d.rectangle([0, 0, W, H], fill=(0, 0, 10, 150))
    network(d, t, red=p.get("red"), col=p.get("col", GOLD))
    orb_from(img, d, t, p)


def sc_close(img, d, t, p):
    img.paste(BOKEH, (int(-10 * t), 0), BOKEH)
    ch = p["c"]
    light = p.get("light", (255, 210, 150))
    d.ellipse([W / 2 - 330, 180, W / 2 + 330, 840], fill=light + (45,))
    hx, hy, R = person(img, d, W / 2, H + 380, 4.2, ch, t, emo=p.get("emo", "neutral"), seed=3)
    ex, ey = 0.36 * R, hy + 0.1 * R
    if p.get("logo") and t > 4:
        a = int(255 * tri(t, 4, 5))
        d.ellipse([hx - ex - 0.23 * R, ey - 0.21 * R, hx - ex + 0.23 * R, ey + 0.21 * R], fill=RED + (a // 2,))
        d.text((hx - ex, ey), "N", font=F_MID, anchor="mm", fill=(255, 255, 255, a))
    if p.get("tears"):
        for k, sg in enumerate((-1, 1)):
            ty = ey + 0.2 * R + ((t * 60 + k * 50) % (0.6 * R))
            x_ = hx + sg * ex
            d.ellipse([x_ - 9, ty - 13, x_ + 9, ty + 13], fill=(140, 200, 255, 220), outline=OUTL, width=2)
    orb_from(img, d, t, p)


def sc_market(img, d, t, p):
    paste_layer(img, FAR, 400 + t * 8)
    rnd = random.Random(4)
    for i, (lab, col) in enumerate([("ЛАПША", (255, 70, 180)), ("ЧИПЫ", (60, 220, 255)), ("РЕМОНТ", (255, 160, 40)), ("ГОЛО", (150, 90, 255))]):
        x = 10 + i * 135
        d.ellipse([x - 20, 380, x + 140, 560], fill=col + (30,))
        d.rectangle([x, 470, x + 120, 510], fill=col + (190,))
        d.text((x + 10, 478), lab, font=F_SM, fill=(20, 10, 20))
        d.rectangle([x + 4, 510, x + 116, 700], fill=(16, 12, 26))
    for k in range(9):
        lx = 30 + k * 60
        d.ellipse([lx - 7, 430 + 10 * math.sin(k), lx + 7, 444 + 10 * math.sin(k)], fill=(255, 190, 90, 220))
    crowd = [(rnd.uniform(0, 1), rnd.uniform(0.6, 1.05), rnd.uniform(-0.04, 0.04)) for _ in range(16)]
    for i, (x, s, vx) in sorted(enumerate(crowd), key=lambda c: c[1][1]):
        person(img, d, ((x + vx * t) % 1.1 - 0.05) * W, 700 + (s - 0.6) * 380, s * 1.1, "crowd", t, seed=i)
    if p.get("agents"):
        for k, ax in enumerate((0.12, 0.88)):
            x = ax * W
            person(img, d, x, 900, 1.5, "agent")
            sy = 600 + 160 * math.sin(t * 1.2 + k * 2)
            d.polygon([(x, 650), (W / 2, sy - 40), (W / 2, sy + 40)], fill=(255, 255, 255, 30))
            d.line([(0, sy), (W, sy)], fill=RED + (120,), width=2)
    persons(img, d, t, p)
    orb_from(img, d, t, p)


def sc_bar(img, d, t, p):
    for k in range(5):
        d.ellipse([40 + k * 110, 120, 100 + k * 110, 180], fill=(255, 170, 90, 60))
        d.line([(70 + k * 110, 0), (70 + k * 110, 135)], fill=(60, 40, 30), width=2)
    fl = 1.0 if random.Random(int(t * 10)).random() > 0.05 else 0.3
    for sw, a in ((10, 40), (5, 80)):
        d.text((150, 230), "ЛАПША", font=F_MID, fill=(255, 60, 170, int(a * fl)), stroke_width=sw, stroke_fill=(255, 60, 170, int(a * fl)))
    d.text((150, 230), "ЛАПША", font=F_MID, fill=(255, 220, 240, int(255 * fl)))
    for y in (320, 400):
        d.line([(30, y), (510, y)], fill=(70, 40, 40), width=4)
    persons(img, d, t, p)
    d.rectangle([0, 700, W, 760], fill=(70, 38, 26), outline=(110, 60, 40), width=3)
    d.rectangle([0, 760, W, H], fill=(30, 16, 12))
    if p.get("holo"):
        k = ease(tri(t, 0.5, 3))
        cx, cy, R = W / 2, 520, 150 * k
        if R > 5:
            d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=BLUE + (170,), width=2)
            for j in range(5):
                rx = R * abs(math.cos(t * 0.8 + j * math.pi / 5))
                d.ellipse([cx - rx, cy - R, cx + rx, cy + R], outline=BLUE + (90,), width=1)
            for j in (-2, -1, 1, 2):
                yy = cy + j * R / 3
                rx = math.sqrt(max(0, R * R - (yy - cy) ** 2))
                d.line([(cx - rx, yy), (cx + rx, yy)], fill=BLUE + (90,), width=1)
            d.text((cx, cy - R - 34), "ПРОТОКОЛ «РАССВЕТ»", font=F_SM, anchor="mm", fill=BLUE + (int(255 * k),))
    orb_from(img, d, t, p)


def sc_roofs(img, d, t, p):
    paste_layer(img, FAR, 600 + t * 15)
    sp, per, gap = 260, 286, 80
    scroll = t * sp
    rnd = random.Random(8)
    deco = [rnd.random() for _ in range(40)]
    for k in range(-1, 14):
        gx = 343 + per * k
        x0, x1 = gx + gap / 2 - scroll, gx + per - gap / 2 - scroll
        if x1 < 0 or x0 > W:
            continue
        d.rectangle([x0, 690, x1, H], fill=(12, 10, 20), outline=(60, 50, 90), width=2)
        if deco[k + 1] < 0.5:
            d.rectangle([x0 + 30, 640, x0 + 80, 690], fill=(16, 14, 26))
            d.ellipse([x0 + 26, 630, x0 + 84, 650], fill=(16, 14, 26))
        else:
            d.rectangle([x1 - 90, 660, x1 - 30, 690], fill=(20, 18, 30))
            d.ellipse([x1 - 72, 668, x1 - 60, 680], outline=(90, 90, 110), width=2)
    jump = abs(math.sin(math.pi * t / 1.1)) * 80
    person(img, d, 200, 690 - jump, 1.1, "lena", t, run=True, pack=True)
    if p.get("threads"):
        for k in range(3):
            ox, oy = 180, 560 - jump
            pts = [(ox + u * 300, oy - math.sin(u * math.pi) * 90 - k * 18 + math.sin(t * 3 + u * 6) * 6) for u in np.linspace(0, 1, 14)]
            d.line(pts, fill=GOLD + (170,), width=2)
        d.polygon([(470, 540), (500, 560), (470, 580)], fill=GOLD + (220,))


def sc_shaft(img, d, t, p):
    for x in (90, 450):
        d.rectangle([x - 8, 0, x + 8, H], fill=(30, 30, 40))
    for k in range(8):
        y = (k * 160 + t * 260) % (H + 160) - 80
        d.rectangle([60, y, 480, y + 14], fill=(26, 26, 36))
        d.ellipse([255, y - 4, 285, y + 18], fill=(255, 170, 60, 120))
    d.rectangle([80, 560, 460, 590], fill=(60, 60, 72), outline=(120, 120, 140), width=2)
    person(img, d, 270, 560, 1.4, "lena", t, pack=True)
    for k, ax in enumerate((150, 380)):
        y = 920 - 40 * tri(t, 0, 10)
        person(img, d, ax, y, 0.6, "agent")
        d.line([(ax, y - 80), (ax + 30 * math.sin(t * 2 + k), 590)], fill=RED + (150,), width=2)
    d.rectangle([0, 0, W, H], fill=(255, 30, 40, int(30 + 25 * math.sin(t * 5))))


def sc_train(img, d, t, p):
    mode = p.get("mode", "tunnel")
    if mode == "tunnel":
        d.rectangle([0, 0, W, H], fill=(6, 6, 10))
        for k in range(10):
            x = (k * 160 - t * 1500) % (W + 200) - 100
            for y in (300, 860):
                d.line([(x, y), (x + 90, y)], fill=(255, 170, 70), width=5)
                d.ellipse([x - 20, y - 20, x + 110, y + 20], fill=(255, 150, 60, 25))
        d.arc([-200, 120, W + 200, 1200], 180, 360, fill=(40, 40, 50), width=8)
    else:
        if p.get("tower"):
            tower(img, d, W / 2 + 120, 760, t, 0.9)
        paste_layer(img, FAR, 800 + t * 60)
        d.rectangle([0, 760, W, H], fill=(10, 14, 34))
        rnd = random.Random(int(t * 6))
        for _ in range(30):
            x, y = rnd.randint(0, W), rnd.randint(770, H)
            d.line([(x, y), (x + rnd.randint(10, 40), y)], fill=(90, 140, 220, 90))
        for k in range(6):
            x = (k * 200 - t * 500) % (W + 300) - 150
            d.line([(x, 120), (x + 260, 640)], fill=(80, 70, 110), width=4)
            d.line([(x, 120), (x - 160, 640)], fill=(80, 70, 110), width=3)
    d.rectangle([0, 650, W, 720], fill=(44, 46, 58), outline=(90, 95, 110), width=3)
    d.rectangle([0, 720, W, 830], fill=(30, 32, 42))
    for k in range(5):
        d.rectangle([20 + k * 110, 735, 100 + k * 110, 800], fill=(255, 220, 140, 200))
    person(img, d, 250, 650, 1.0, "lena", t, kneel=True, pack=True)
    if p.get("drone"):
        x, y = 420 - 60 * tri(t, 0, 4) + 20 * math.sin(t * 2), 280 + 20 * math.sin(t * 3)
        if not (5.0 < t):
            d.polygon([(x, y), (200, 650), (320, 650)], fill=(255, 40, 50, 45))
        fall = 0 if t < 5.3 else (t - 5.3) ** 2 * 300
        d.line([(x - 32, y - 9 + fall), (x + 32, y - 9 + fall)], fill=(90, 90, 110), width=3)
        d.ellipse([x - 22, y - 8 + fall, x + 22, y + 8 + fall], fill=(20, 20, 28), outline=(110, 110, 130))
        if 4.8 < t < 5.6:
            draw_orb(img, d, 225, 540, 60 * (1 - abs(t - 5.2) / 0.4), t, threads=False)
            d.rectangle([0, 0, W, H], fill=(200, 230, 255, int(200 * (1 - abs(t - 5.2) / 0.4))))


def sc_server(img, d, t, p):
    vp = (W / 2, 430)
    for k in range(-8, 9):
        d.line([vp, (W / 2 + k * 90, H)], fill=BLUE + (40,), width=1)
    rnd = random.Random(int(t * 6))
    for i in range(6, -1, -1):
        s = 0.8 ** i
        for sg in (-1, 1):
            cx = W / 2 + sg * (55 + 240 * s)
            w2, h2 = 70 * s, 360 * s
            y0, y1 = vp[1] - h2 * 0.9, vp[1] + h2
            d.rectangle([cx - w2, y0, cx + w2, y1], fill=(14, 18, 30), outline=(40, 60, 100), width=max(1, int(2 * s)))
            for k in range(int(14 * s) + 3):
                yy = y0 + 10 * s + k * (y1 - y0 - 20 * s) / (int(14 * s) + 3)
                c = (60, 255, 140) if rnd.random() < 0.7 else BLUE
                if p.get("alarm"):
                    c = RED if rnd.random() < 0.6 else c
                d.rectangle([cx - w2 * 0.7, yy, cx - w2 * 0.7 + max(2, 6 * s), yy + max(2, 4 * s)], fill=c)
    if p.get("door"):
        d.rectangle([W / 2 - 50, 300, W / 2 + 50, 520], fill=(255, 240, 210))
        d.polygon([(W / 2 - 50, 520), (W / 2 + 50, 520), (W / 2 + 220, H), (W / 2 - 220, H)], fill=(255, 240, 210, 50))
        person(img, d, W / 2, 520, 1.15, "max")
    if p.get("screen"):
        d.rectangle([110, 200, 430, 430], fill=(30, 22, 10), outline=GOLD, width=3)
        person(img, d, 270, 520, 1.6, "father", t)
        d.rectangle([110, 430, 430, 560], fill=(3, 8, 18))
        d.rectangle([110, 200, 430, 430], outline=GOLD, width=3)
        for yy in range(204, 428, 5):
            d.line([(114, yy), (426, yy)], fill=(0, 0, 0, 70))
        d.text((122, 208), "REC ● ЗАПИСЬ", font=F_SM, fill=RED)
    if p.get("links"):
        o = p["orb"]
        for k in range(8):
            tx, ty = (60 if k % 2 else W - 60), 260 + k * 60
            prog = ease(tri(t, 0.5 + k * 0.3, 3 + k * 0.3))
            ex, ey = o["x"] + (tx - o["x"]) * prog, o["y"] + (ty - o["y"]) * prog
            d.line([(o["x"], o["y"]), (ex, ey)], fill=GOLD + (170,), width=2)
            u = (t * 0.8 + k * 0.13) % 1 * prog
            d.ellipse([o["x"] + (tx - o["x"]) * u - 3, o["y"] + (ty - o["y"]) * u - 3,
                       o["x"] + (tx - o["x"]) * u + 3, o["y"] + (ty - o["y"]) * u + 3], fill=(255, 255, 230))
    persons(img, d, t, p)
    orb_from(img, d, t, p)
    if p.get("shields"):
        k = 0.82 * ease(tri(t, 1, 8))
        for x0, x1 in ((0, k * W / 2), (W - k * W / 2, W)):
            d.rectangle([x0, 0, x1, H], fill=(70, 74, 84), outline=(130, 135, 150), width=4)
            for yy in range(0, H, 60):
                d.line([(x0, yy), (x1, yy)], fill=(50, 52, 60), width=2)
        if k > 0.05:
            for yy in range(0, H, 40):
                d.polygon([(k * W / 2 - 14, yy), (k * W / 2, yy + 14), (k * W / 2, yy + 34), (k * W / 2 - 14, yy + 20)], fill=(250, 200, 40))
    if p.get("alarm"):
        d.rectangle([0, 0, W, H], fill=(255, 20, 30, int(40 + 35 * math.sin(t * 6))))


def sc_antenna(img, d, t, p):
    paste_layer(img, FAR, 1200 + t * 10)
    if 3.0 < t < 3.25 or 7.4 < t < 7.55:
        d.rectangle([0, 0, W, H], fill=(220, 220, 255, 120))
    cx, top, base = W / 2, 140, 790
    if p.get("beam"):
        k = ease(tri(t, 0.5, 3))
        bw = 30 + 20 * math.sin(t * 8)
        d.polygon([(cx - 8, top), (cx - bw * k * 2, 0), (cx + bw * k * 2, 0), (cx + 8, top)], fill=(200, 230, 255, int(200 * k)))
        d.polygon([(cx - 4, top), (cx - bw * k, 0), (cx + bw * k, 0), (cx + 4, top)], fill=(255, 255, 255, int(230 * k)))
        draw_orb(img, d, cx, top, 40 * k + 5, t)
    for sg in (-1, 1):
        d.line([(cx + sg * 70, base), (cx, top)], fill=(70, 70, 90), width=5)
    for k in range(1, 12):
        y = base - k * (base - top) / 12
        w2 = 70 * (y - top) / (base - top)
        d.line([(cx - w2, y), (cx + w2, y)], fill=(60, 60, 80), width=2)
        d.line([(cx - w2, y), (cx + w2 * 0.9, y + (base - top) / 12)], fill=(50, 50, 70), width=1)
    if int(t * 2) % 2:
        d.ellipse([cx - 6, top - 6, cx + 6, top + 6], fill=RED)
    d.rectangle([0, base, W, H], fill=(22, 22, 30), outline=(70, 70, 90), width=3)
    persons(img, d, t, p)
    orb_from(img, d, t, p)


def sc_hands(img, d, t, p):
    img.paste(BOKEH, (0, int(-6 * t)), BOKEH)
    draw_orb(img, d, W / 2, 460, 120, t)
    for sg in (-1, 1):
        d.ellipse([W / 2 + sg * 160 - 110, 520, W / 2 + sg * 160 + 110, 700], fill=(18, 40, 26), outline=(90, 235, 130), width=4)
        d.ellipse([W / 2 + sg * 60 - 70, 540, W / 2 + sg * 60 + 70, 620], fill=(24, 50, 32), outline=(90, 235, 130), width=3)


SCENES = dict(city=sc_city, interior=sc_interior, photo=sc_photo, map=sc_map, close=sc_close,
              market=sc_market, bar=sc_bar, roofs=sc_roofs, shaft=sc_shaft, train=sc_train,
              server=sc_server, antenna=sc_antenna, hands=sc_hands)
DEFAULT_SKY = dict(city="night", interior="dark", photo="dark", map="dark", close="dark", market="night",
                   bar="bar", roofs="night", shaft="dark", train="night", server="server", antenna="storm", hands="dark")
OUTDOOR = {"city", "market", "roofs", "antenna", "train"}


# ---------------------------------------------------------------- «киношная» обработка
GRAIN = np.random.default_rng(42).normal(0, 1, (8, H, W)).astype(np.float32)


def cinematic(img, t, gt, shot_i):
    # 1) медленный наезд камеры + лёгкая тряска «с рук»
    z = 1.0 + 0.06 * (t / SHOT) + (0.03 if shot_i % 2 else 0.0)
    sx = 4 * math.sin(gt * 1.7) + 2 * math.sin(gt * 4.3)
    sy = 3 * math.cos(gt * 1.3) + 2 * math.sin(gt * 3.7)
    cw, ch = W / z, H / z
    x0, y0 = (W - cw) / 2 + sx, (H - ch) / 2 + sy
    img = img.transform((W, H), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255
    # 2) свечение ярких источников (bloom)
    small = img.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6))
    b = np.asarray(small.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255
    a = 1 - (1 - a) * (1 - np.clip(b - 0.25, 0, 1) * 0.9)
    # 3) цветокоррекция teal & orange + мягкий контраст
    lum = a.mean(axis=2, keepdims=True)
    a = a + (np.array([0.05, 0.02, -0.04]) * lum + np.array([-0.02, 0.02, 0.05]) * (1 - lum))
    a = np.clip(a, 0, 1)
    a = a * a * (3 - 2 * a) * 0.35 + a * 0.65
    a = 0.035 + a * 0.95  # приподнятые «плёночные» тени
    # 4) плёночное зерно
    a = a + GRAIN[int(gt * FPS) % 8][..., None] * 0.012
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


# ---------------------------------------------------------------- оверлеи
def glow_text(d, xy, txt, font, col, a=1.0, anchor="mm"):
    big = font.size > 60
    for sw, al in (((12, 35), (6, 70)) if big else ((6, 30), (3, 60))):
        d.text(xy, txt, font=font, anchor=anchor, fill=col + (int(al * a),), stroke_width=sw, stroke_fill=col + (int(al * a),))
    d.text(xy, txt, font=font, anchor=anchor, fill=(255, 255, 255, int(255 * a)))


CAP_COL = {"Лена": (255, 255, 255), "Эхо": (120, 215, 255), "Макс": (255, 190, 80), "Вейн": (255, 175, 200),
           "Отец": (255, 225, 70), None: (255, 225, 70)}
DUR = {}  # (серия, кадр) -> длительность реплики, заполняется при синтезе голоса


def chunks(txt):
    words, out, cur = txt.split(), [], []
    for w in words:
        if cur and (len(" ".join(cur + [w])) > 15 or len(cur) >= 3):
            out.append(" ".join(cur))
            cur = []
        cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def stroke_text(d, xy, txt, font, col, sw=6, a=1.0):
    d.text(xy, txt, font=font, anchor="mm", fill=col + (int(255 * a),), stroke_width=sw, stroke_fill=(12, 10, 20, int(255 * a)))


def caption(d, spk, txt, t, dur):
    """Субтитры в стиле Shorts: крупно, заглавными, с обводкой, по 2–3 слова синхронно с речью."""
    import re
    txt = re.sub(r"\(.*?\)\s*", "", txt)
    parts = chunks(txt.upper())
    t0 = 0.9
    if t < t0 - 0.25 or t > 9.3:
        return
    total = sum(len(c) + 2 for c in parts)
    acc, idx = 0.0, len(parts) - 1
    for i, c in enumerate(parts):
        acc += (len(c) + 2) / total
        if t < t0 + dur * acc:
            idx = i
            break
    lines = textwrap.wrap(parts[idx], 13)
    pop = 1.0
    y = H * 0.78 - (len(lines) - 1) * 27
    for ln in lines:
        stroke_text(d, (W / 2, y), ln, F_CAP, CAP_COL.get(spk, (255, 255, 255)), 6, pop)
        y += 54


def hook(d, txt, a):
    lines = textwrap.wrap(txt, 18)
    y = 120 - (len(lines) - 1) * 22
    for ln in lines:
        stroke_text(d, (W / 2, y), ln, F_HOOK, (255, 214, 40), 5, a)
        y += 46


def render_frame(ep_i, shot_i, shot, t, gt):
    kind, p = shot["kind"], shot.get("p", {})
    dur = DUR.get((ep_i, shot_i), len(shot.get("txt", "")) * 0.075)
    SPEAK.update(char=CHAR_OF.get(shot.get("spk")), t0=0.9, t1=0.9 + dur)
    img = SKIES[p.get("sky", DEFAULT_SKY[kind])].copy()
    d = ImageDraw.Draw(img, "RGBA")
    SCENES[kind](img, d, t, p)
    if p.get("rain", kind in OUTDOOR):
        rain(d, gt)
    img = cinematic(img, t, gt, shot_i)
    img.paste(VIG, (0, 0), VIG)
    d = ImageDraw.Draw(img, "RGBA")
    ep = EPISODES[ep_i]
    if shot_i == 0 and t < 3.6:
        a = tri(t, 0.2, 0.9) * (1 - tri(t, 2.8, 3.6))
        d.rectangle([0, 0, W, H], fill=(0, 0, 0, int(120 * a)))
        glow_text(d, (W / 2, H / 2 - 60), "ЭХО", F_BIG, BLUE, a)
        glow_text(d, (W / 2, H / 2 + 50), f"СЕРИЯ {ep_i + 1} · {ep['title'].upper()}", F_MID, BLUE, a)
    if ep.get("hook") and not (shot_i == 0 and t < 3.6) and not (shot.get("title") and t > 6.0):
        hook(d, ep["hook"], tri(t, 3.6, 4.0) if shot_i == 0 else 1.0)
    if shot.get("txt"):
        caption(d, shot.get("spk"), shot["txt"], t, dur)
    if shot.get("title") and t > 6.0:
        a = tri(t, 6.0, 7.0)
        d.rectangle([0, 0, W, H], fill=(0, 0, 0, int(190 * a)))
        top, bottom = shot["title"]
        glow_text(d, (W / 2, H / 2 - 40), top, F_MID, BLUE, a)
        glow_text(d, (W / 2, H / 2 + 20), bottom, F_MID, BLUE, a)
    fade = min(1.0, t / 0.3, (SHOT - t) / 0.3)
    if fade < 1.0:
        d.rectangle([0, 0, W, H], fill=(0, 0, 0, int(255 * (1 - max(0, fade)))))
    return img


# ---------------------------------------------------------------- музыка
def tone(f, t):
    return np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.003 * t) + 0.5 * np.sin(2 * np.pi * f * 0.998 * t)


def music(ep_i, path, dur=90, sr=22050):
    t = np.arange(int(dur * sr)) / sr
    ep = EPISODES[ep_i]
    prog = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (261.63, 329.63, 392.0), (196.0, 246.94, 293.66)]
    out = np.zeros_like(t)
    seg = 5.0
    for i in range(int(dur / seg)):
        s, e = int(i * seg * sr), min(len(t), int((i + 1) * seg * sr + 0.8 * sr))
        tt = t[s:e] - i * seg
        env = np.minimum(1, tt / 0.8) * np.minimum(1, np.maximum(0, (seg + 0.8 - tt) / 0.8))
        for f in prog[i % 4]:
            out[s:e] += 0.12 * env * tone(f / 2, t[s:e])
        out[s:e] += 0.18 * env * np.sin(2 * np.pi * prog[i % 4][0] / 4 * t[s:e])
    out *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.1 * t)
    pulse = ep.get("pulse")
    if pulse:
        a, b, bpm = pulse
        beat = 60 / bpm
        for k in range(int((b - a) / beat)):
            s = int((a + k * beat) * sr)
            n = int(0.25 * sr)
            tt = np.arange(min(n, len(t) - s)) / sr
            out[s:s + len(tt)] += 0.5 * np.exp(-tt * 18) * np.sin(2 * np.pi * (55 + 60 * np.exp(-tt * 30)) * tt)
    for at in (0.6, dur - 8.0):
        for k, f in enumerate((440.0, 523.25, 659.25)):
            s = int((at + k * 0.55) * sr)
            tt = np.arange(int(1.6 * sr)) / sr
            seg_ = 0.22 * np.exp(-tt * 2.5) * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt))
            out[s:s + len(tt)] += seg_[: len(out) - s]
    rnd = np.random.default_rng(ep_i)
    noise = np.convolve(rnd.normal(0, 1, len(t)), np.ones(6) / 6, mode="same")
    out += 0.035 * noise
    out = out / np.max(np.abs(out)) * 0.8
    voice, active = voice_track(ep_i, len(out), sr)
    duck = np.convolve(active, np.ones(int(0.3 * sr)) / int(0.3 * sr), mode="same")
    out = out * (1 - 0.6 * duck) * 0.75 + voice
    out = np.clip(out / max(1.0, np.max(np.abs(out)) / 0.95), -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((out * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------- озвучка (Silero)
VOICES = {  # голос Silero v4 (нейросеть), ffmpeg-фильтр
    "Лена": ("xenia", "volume=1.0"),
    "Эхо": ("aidar", "aecho=0.8:0.6:60|120:0.3|0.15"),
    "Макс": ("eugene", "volume=1.0"),
    "Вейн": ("baya", "asetrate=48000*0.95,aresample=48000"),
    "Отец": ("aidar", "asetrate=48000*0.9,aresample=48000,highpass=f=250,lowpass=f=3600,aecho=0.8:0.5:40:0.2"),
    None: ("kseniya", "volume=1.0"),
}
SILERO = os.environ.get("SILERO_MODEL", "/root/silero/v4_ru.pt")  # https://models.silero.ai/models/tts/ru/v4_ru.pt
_tts = None


def silero():
    global _tts
    if _tts is None:
        import torch
        torch.set_num_threads(1)
        _tts = torch.package.PackageImporter(SILERO).load_pickle("tts_models", "model")
        _tts.to("cpu")
    return _tts


def say(spk, txt, sr, max_len):
    import re, tempfile
    whisper = "шёпотом" in txt
    txt = re.sub(r"\(.*?\)\s*", "", txt).strip().replace("«", "").replace("»", "")
    voice, flt = VOICES.get(spk, VOICES[None])
    if whisper:
        flt += ",highpass=f=400,volume=0.6"
    with tempfile.TemporaryDirectory() as td:
        raw, fx = os.path.join(td, "raw.wav"), os.path.join(td, "fx.wav")
        silero().save_wav(text=txt, speaker=voice, sample_rate=48000, audio_path=raw, put_accent=True, put_yo=True)
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", raw]))
        if dur > max_len:
            flt += f",atempo={min(2.0, dur / max_len):.3f}"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-af", flt, "-ar", str(sr), "-ac", "1", fx], check=True)
        with wave.open(fx) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return a / max(1e-6, np.max(np.abs(a))) * 0.9


def voice_track(ep_i, n, sr):
    out, active = np.zeros(n), np.zeros(n)
    for si, shot in enumerate(EPISODES[ep_i]["shots"]):
        if not shot.get("txt"):
            continue
        start = 0.9
        a = say(shot.get("spk"), shot["txt"], sr, SHOT - start - 1.0)
        DUR[(ep_i, si)] = len(a) / sr
        s = int((si * SHOT + start) * sr)
        e = min(n, s + len(a))
        out[s:e] += a[: e - s]
        active[s:e] = 1
    return out, active


# ---------------------------------------------------------------- сборка
def render_episode(ep_i):
    ep = EPISODES[ep_i]
    wav = os.path.join(OUT, f"_music{ep_i + 1}.wav")
    mp4 = os.path.join(OUT, f"echo_ep{ep_i + 1}.mp4")
    music(ep_i, wav)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "23",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", mp4]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for si, shot in enumerate(ep["shots"]):
        for fi in range(SHOT * FPS):
            t = fi / FPS
            ff.stdin.write(render_frame(ep_i, si, shot, t, si * SHOT + t).tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    return mp4


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    eps = [int(a) - 1 for a in sys.argv[1:]] or list(range(len(EPISODES)))
    with Pool(min(4, len(eps))) as pool:
        files = pool.map(render_episode, eps)
    print("\n".join(files))
    if len(eps) == len(EPISODES):
        lst = os.path.join(OUT, "_list.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{os.path.basename(x)}'\n" for x in files)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", os.path.join(OUT, "echo_full_series.mp4")], check=True)
        os.remove(lst)
        print(os.path.join(OUT, "echo_full_series.mp4"))
