#!/usr/bin/env python3
"""Procedural chill-house generator for the "playlist." focus-music channel.

Every track is synthesized from scratch with numpy (no samples), so the
music is 100% original and safe from YouTube Content ID claims.

Usage:  python3 playlist/music.py                 # playlist #1 (locked_in), all tracks
        python3 playlist/music.py late_night      # playlist #2
        python3 playlist/music.py late_night 3 7  # only tracks 3 and 7
"""
import os, sys, wave
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 44100
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "tracks")

# ------------------------------------------------------------------ helpers
def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def env_adsr(n, a=0.005, r=0.05):
    e = np.ones(n)
    na, nr = max(1, int(a * SR)), max(1, int(r * SR))
    e[:na] = np.linspace(0, 1, na)
    e[-nr:] *= np.linspace(1, 0, nr)
    return e


def t_(n):
    return np.arange(n) / SR


# ------------------------------------------------------------------ instruments
@lru_cache(None)
def kick(tone=0):
    n = int(0.45 * SR); t = t_(n)
    f = 45 + (115 + tone * 10) * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 7.5)
    click = hp(np.random.default_rng(1).standard_normal(n), 2500) * np.exp(-t * 260) * 0.25
    return np.tanh((body + click) * 1.6) * 0.9


@lru_cache(None)
def clap(seed=0):
    rng = np.random.default_rng(10 + seed)
    n = int(0.35 * SR); t = t_(n)
    noise = bp(rng.standard_normal(n), 900, 5200)
    e = np.zeros(n)
    for d in (0, 0.011, 0.022):
        k = int(d * SR)
        e[k:] += np.exp(-t[: n - k] * (90 if d < 0.02 else 18))
    return noise * e * 0.35


@lru_cache(None)
def hat(open_=False, seed=0):
    rng = np.random.default_rng(20 + seed + open_)
    n = int((0.28 if open_ else 0.06) * SR); t = t_(n)
    x = hp(rng.standard_normal(n), 7000, 4)
    return x * np.exp(-t * (11 if open_ else 70)) * (0.22 if open_ else 0.16)


@lru_cache(None)
def shaker(seed=0):
    rng = np.random.default_rng(30 + seed)
    n = int(0.07 * SR); t = t_(n)
    x = bp(rng.standard_normal(n), 4000, 11000)
    e = np.minimum(t / 0.015, 1) * np.exp(-t * 45)
    return x * e * 0.07


@lru_cache(None)
def bass(note, steps, step_len):
    n = int(steps * step_len * SR * 0.92); t = t_(n)
    f = midi(note)
    x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t) + 0.08 * np.sin(6 * np.pi * f * t)
    x = np.tanh(x * 1.3)
    return x * env_adsr(n, 0.006, 0.04) * np.exp(-t * 1.2) * 0.5


@lru_cache(None)
def rhodes(chord, dur, bright=1.0):
    n = int(dur * SR); t = t_(n)
    out = np.zeros(n)
    for note in chord:
        f = midi(note)
        x = (np.sin(2 * np.pi * f * t) * np.exp(-t * 1.6)
             + 0.35 * bright * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 4)
             + 0.12 * bright * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t * 7)
             + 0.05 * bright * np.sin(2 * np.pi * 7.02 * f * t) * np.exp(-t * 25))
        out += x
    trem = 1 + 0.18 * np.sin(2 * np.pi * 4.2 * t)
    return out * trem * env_adsr(n, 0.004, 0.08) * (0.16 / np.sqrt(len(chord)))


@lru_cache(None)
def pad(chord, dur, cutoff=1400):
    n = int(dur * SR); t = t_(n)
    out = np.zeros(n)
    for note in chord:
        for det in (-0.09, 0.0, 0.11):
            f = midi(note + 12) * (1 + det / 12)
            ph = np.random.default_rng(int(note * 7 + det * 100)).uniform(0, 6.28)
            for k in range(1, 7):
                out += np.sin(2 * np.pi * k * f * t + ph * k) / k
    out = lp(out, cutoff)
    e = np.minimum(t / 0.6, 1) * np.minimum((dur - t) / 0.6, 1)
    return out * e * (0.03 / np.sqrt(len(chord)))


@lru_cache(None)
def pluck(note, kind=0):
    n = int(0.9 * SR); t = t_(n)
    f = midi(note)
    if kind == 0:   # soft bell pluck
        x = np.sin(2 * np.pi * f * t + 1.2 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 9))
    else:           # mellow marimba-ish
        x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t * 20)
    return x * np.exp(-t * 5.5) * env_adsr(n, 0.003, 0.05) * 0.15


# ------------------------------------------------------------------ musical data
# chords are semitone offsets from the key root (voiced around root+12),
# bass roots are offsets from the root in octave 2.
PROGS = {
    "minor4": ([(0, 3, 7, 10, 14), (-4, 0, 3, 7, 11), (3, 7, 10, 14), (-2, 2, 5, 9)], [0, -4, 3, -2]),
    "vamp":   ([(0, 3, 7, 10, 14), (5, 8, 12, 15, 19)], [0, 5]),
    "major4": ([(0, 4, 7, 11, 14), (-3, 0, 4, 7, 11), (2, 5, 9, 12, 16), (-5, 0, 2, 7, 11)], [0, -3, 2, -5]),
    "descend": ([(0, 3, 7, 10, 14), (-2, 2, 5, 9, 12), (-4, 0, 3, 7, 10), (-5, -2, 2, 5, 9)], [0, -2, -4, -5]),
    "minor_ii": ([(0, 3, 7, 10, 14), (-4, 0, 3, 7, 10), (5, 8, 12, 15, 19), (7, 10, 14, 17)], [0, -4, 5, 7]),
    "lowvamp": ([(0, 3, 7, 10, 14), (-2, 2, 5, 9, 12)], [0, -2]),
    "dorian": ([(0, 3, 7, 10, 14), (2, 5, 9, 12, 17), (0, 3, 7, 10, 14), (5, 9, 12, 16, 19)], [0, 2, 0, 5]),
}
STABS = {
    "off":   [(2, 2), (6, 2), (10, 3)],
    "push":  [(0, 5), (6, 2), (10, 4)],
    "skip":  [(3, 2), (7, 2), (11, 2), (14, 2)],
    "hold":  [(0, 15)],
    "late":  [(2, 3), (7, 3), (13, 3)],
    "sparse": [(0, 3), (10, 5)],
    "lazy":  [(3, 4), (11, 4)],
}
BASSES = {
    "off":   [(2, 2, 0), (6, 2, 0), (10, 2, 0), (14, 2, 12)],
    "walk":  [(0, 3, 0), (3, 2, 0), (8, 3, 0), (11, 2, 7), (14, 2, 12)],
    "long":  [(2, 4, 0), (10, 4, 0)],
    "bounce": [(0, 2, 0), (3, 1, 12), (6, 2, 0), (10, 2, 0), (13, 2, 7)],
}
PENTA_MIN = [0, 3, 5, 7, 10]
PENTA_MAJ = [0, 2, 4, 7, 9]

TRACKS = [
    dict(title="Locked In",          key=45, bpm=122, prog="minor4",  stab="off",  bass="off",    bars_per_chord=1, lead=0, seed=1),
    dict(title="Deep Work Mode",     key=43, bpm=120, prog="vamp",    stab="push", bass="walk",   bars_per_chord=2, lead=1, seed=2),
    dict(title="No Days Off",        key=48, bpm=124, prog="major4",  stab="skip", bass="bounce", bars_per_chord=1, lead=0, seed=3),
    dict(title="Quiet Ambition",     key=41, bpm=118, prog="descend", stab="hold", bass="long",   bars_per_chord=2, lead=1, seed=4),
    dict(title="Discipline Over Mood", key=46, bpm=122, prog="dorian", stab="late", bass="off",   bars_per_chord=1, lead=0, seed=5),
    dict(title="Late Night Grind",   key=44, bpm=120, prog="minor4",  stab="push", bass="bounce", bars_per_chord=2, lead=1, seed=6),
    dict(title="One More Rep",       key=47, bpm=124, prog="vamp",    stab="off",  bass="walk",   bars_per_chord=1, lead=0, seed=7),
    dict(title="Focus Is A Skill",   key=42, bpm=119, prog="major4",  stab="hold", bass="long",   bars_per_chord=2, lead=1, seed=8),
    dict(title="Silent Progress",    key=45, bpm=121, prog="descend", stab="skip", bass="off",    bars_per_chord=1, lead=0, seed=9),
    dict(title="Earn Your Weekend",  key=43, bpm=123, prog="dorian",  stab="late", bass="bounce", bars_per_chord=2, lead=1, seed=10),
]

# playlist #2 — "late night focus": slower, darker, more space
TRACKS_NIGHT = [
    dict(title="After Hours",         key=45, bpm=114, prog="lowvamp",  stab="sparse", bass="long",   bars_per_chord=2, lead=1, seed=21, dark=True),
    dict(title="City Lights Off",     key=42, bpm=116, prog="minor_ii", stab="lazy",   bass="off",    bars_per_chord=1, lead=0, seed=22, dark=True),
    dict(title="Midnight Draft",      key=47, bpm=112, prog="descend",  stab="sparse", bass="walk",   bars_per_chord=2, lead=1, seed=23, dark=True),
    dict(title="3AM Clarity",         key=44, bpm=115, prog="vamp",     stab="hold",   bass="long",   bars_per_chord=2, lead=0, seed=24, dark=True),
    dict(title="Nobody's Watching",   key=41, bpm=117, prog="minor_ii", stab="off",    bass="bounce", bars_per_chord=1, lead=1, seed=25, dark=True),
    dict(title="Blue Screen Glow",    key=46, bpm=113, prog="lowvamp",  stab="lazy",   bass="off",    bars_per_chord=2, lead=0, seed=26, dark=True),
    dict(title="Quiet Hours",         key=43, bpm=114, prog="dorian",   stab="sparse", bass="long",   bars_per_chord=2, lead=1, seed=27, dark=True),
    dict(title="Night Shift Mind",    key=48, bpm=116, prog="minor4",   stab="late",   bass="walk",   bars_per_chord=1, lead=0, seed=28, dark=True),
    dict(title="Still Awake",         key=44, bpm=112, prog="descend",  stab="hold",   bass="long",   bars_per_chord=2, lead=1, seed=29, dark=True),
    dict(title="Before Sunrise",      key=45, bpm=115, prog="minor_ii", stab="lazy",   bass="bounce", bars_per_chord=2, lead=0, seed=30, dark=True),
]
PLAYLISTS = {"locked_in": (TRACKS, OUT), "late_night": (TRACKS_NIGHT, os.path.join(os.path.dirname(OUT), "late_night", "tracks"))}

# bus gains (balanced so chords sit ~8-10 dB under the low end, like real house)
MIX = dict(kick=0.7, bass=0.6, clap=1.0, hatL=1.1, hatR=1.1, stabs=4.0, pad=4.0, lead=2.5, keys=4.5, fx=1.0)

# arrangement (bar ranges) — 184 bars ≈ 6 min at ~122 bpm
SECTIONS = [("intro_a", 8), ("intro_b", 8), ("groove", 40), ("full", 32),
            ("break", 16), ("build", 8), ("full2", 32), ("groove2", 24), ("outro", 16)]
LAYERS = {
    "kick":   {"intro_b", "groove", "full", "build", "full2", "groove2", "outro"},
    "clap":   {"groove", "full", "full2", "groove2"},
    "chat":   {"intro_a", "intro_b", "groove", "full", "build", "full2", "groove2", "outro"},
    "ohat":   {"full", "full2", "groove2"},
    "shaker": {"groove", "full", "break", "full2", "groove2", "outro"},
    "bass":   {"intro_b", "groove", "full", "full2", "groove2"},
    "stabs":  {"groove", "full", "build", "full2", "groove2", "outro"},
    "pad":    {"intro_a", "intro_b", "full", "break", "build", "full2", "outro"},
    "lead":   {"full", "break", "full2"},
    "keys":   {"break"},
}


def section_map():
    out = []
    for name, n in SECTIONS:
        out += [name] * n
    return out


# ------------------------------------------------------------------ render
def place(buf, x, pos, g=1.0):
    if pos >= len(buf):
        return
    end = min(len(buf), pos + len(x))
    buf[pos:end] += x[: end - pos] * g


def reverb_ir(seconds, seed):
    n = int(seconds * SR); t = t_(n)
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(n) * np.exp(-t * 6.5 / seconds)
    ir = lp(ir, 6000)
    ir[: int(0.012 * SR)] = 0
    return ir / np.sqrt(np.sum(ir ** 2))


def render(job):
    plname, idx = job
    tracks, outdir = PLAYLISTS[plname]
    cfg = tracks[idx]
    dark = cfg.get("dark", False)
    rng = np.random.default_rng(cfg["seed"])
    bpm = cfg["bpm"]
    beat = 60.0 / bpm
    step = beat / 4
    bar = beat * 4
    secs = section_map()
    nbars = len(secs)
    N = int((nbars * bar + 3) * SR)
    bus = {k: np.zeros(N) for k in ["kick", "clap", "hatL", "hatR", "bass", "stabs", "pad", "lead", "keys", "fx"]}
    chords, roots = PROGS[cfg["prog"]]
    key = cfg["key"]
    bpc = cfg["bars_per_chord"]
    swing = (0.18 + 0.08 * rng.random()) if dark else (0.12 + 0.08 * rng.random())
    tone = int(rng.integers(0, 3))
    bright = (0.45 + 0.3 * rng.random()) if dark else (0.7 + 0.5 * rng.random())

    scale = PENTA_MAJ if cfg["prog"] == "major4" else PENTA_MIN
    motif = []
    for s in range(32):                       # 2-bar motif, 16th grid
        if rng.random() < (0.32 if s % 2 == 0 else 0.12):
            deg = int(rng.integers(0, len(scale) * 2))
            motif.append((s, key + 36 + scale[deg % len(scale)] + 12 * (deg // len(scale))))
    if len(motif) < 4:
        motif += [(0, key + 36), (6, key + 39), (12, key + 43)]

    def st(b, s):  # sample index of bar b, 16th step s (with swing)
        tt = b * bar + s * step + (swing * step if s % 2 else 0)
        return int(tt * SR)

    for b, sec in enumerate(secs):
        on = lambda layer: sec in LAYERS[layer]
        ci = (b // bpc) % len(chords)
        chord = tuple(key + 12 + c for c in chords[ci])
        root = key - 12 + roots[ci]
        if on("kick"):
            beats = [0, 4, 8, 12]
            if sec == "build" and b % 8 >= 6:
                beats = list(range(0, 16, 2))
            for s in beats:
                place(bus["kick"], kick(tone), st(b, s))
        if on("clap"):
            for s in (4, 12):
                place(bus["clap"], clap(tone), st(b, s), 0.9)
        if sec == "build" and b % 8 >= 4:
            for s in range(0, 16, 2 if b % 8 < 7 else 1):
                place(bus["clap"], clap(tone), st(b, s), 0.25 + 0.5 * s / 16)
        if on("chat"):
            for s in range(16):
                g = 0.55 if s % 4 == 2 else 0.3
                place(bus["hatL" if s % 2 else "hatR"], hat(False, tone), st(b, s), g * (0.6 + 0.4 * rng.random()))
        if on("ohat"):
            for s in (2, 6, 10, 14):
                place(bus["hatR"], hat(True, tone), st(b, s), 0.8)
        if on("shaker"):
            for s in range(16):
                place(bus["hatL"], shaker(tone), st(b, s), 0.7 if s % 2 else 1.0)
        if on("bass"):
            for s, ln, off in BASSES[cfg["bass"]]:
                place(bus["bass"], bass(root + off, ln, step), st(b, s))
        if on("stabs"):
            for s, ln in STABS[cfg["stab"]]:
                place(bus["stabs"], rhodes(chord, round(ln * step, 3), round(bright, 2)), st(b, s))
        if on("pad") and b % bpc == 0:
            place(bus["pad"], pad(chord, round(bar * bpc + 0.4, 3), 900 if dark else 1400), int(b * bar * SR))
        if on("keys") and b % bpc == 0:
            place(bus["keys"], rhodes(chord, round(bar * bpc, 3), round(bright, 2)), int(b * bar * SR), 1.4)
        if on("lead"):
            half = b % 2
            rep = (b // 2) % 4
            for s, note in motif:
                if s // 16 != half:
                    continue
                if rep == 3 and rng.random() < 0.3:
                    continue
                g = 0.55 if sec == "break" else 0.8
                place(bus["lead"], pluck(note, cfg["lead"]), st(b, s % 16), g)

    # riser in build sections
    b0 = secs.index("build")
    n0, n1 = int(b0 * bar * SR), int((b0 + 8) * bar * SR)
    noise = hp(rng.standard_normal(n1 - n0), 1500)
    ramp = np.linspace(0, 1, n1 - n0) ** 2
    bus["fx"][n0:n1] += noise * ramp * 0.06

    # sidechain (duck stuff under the kick)
    t = (np.arange(N) / SR) % beat
    duck = 1 - 0.55 * np.exp(-t / 0.11) * np.minimum(t / 0.004, 1)
    kick_on = np.zeros(N)
    for b, sec in enumerate(secs):
        if sec in LAYERS["kick"]:
            kick_on[int(b * bar * SR): int((b + 1) * bar * SR)] = 1
    kick_on = np.convolve(kick_on, np.ones(2000) / 2000, mode="same")
    side = 1 - (1 - duck) * kick_on
    for k in ("bass", "stabs", "pad", "lead", "keys"):
        bus[k] *= side

    bus["lead"] = hp(bus["lead"], 300)
    # dotted-8th ping-pong delay for the lead
    d = int(step * 3 * SR)
    leadL, leadR = bus["lead"].copy(), bus["lead"].copy()
    echo = bus["lead"].copy()
    for i in range(1, 5):
        echo = np.concatenate([np.zeros(d), echo[:-d]]) * 0.45
        echo = lp(echo, 4000, 1)
        (leadL if i % 2 else leadR)[:] += echo

    # stereo mix
    for k, g in MIX.items():
        bus[k] = bus[k] * g
    leadL, leadR = leadL * MIX["lead"], leadR * MIX["lead"]
    L = (bus["kick"] + bus["bass"] + bus["clap"] + bus["hatL"] + bus["hatR"] * 0.55
         + bus["stabs"] + bus["pad"] * 0.9 + leadL + bus["keys"] + bus["fx"])
    R = (bus["kick"] + bus["bass"] + bus["clap"] + bus["hatL"] * 0.55 + bus["hatR"]
         + bus["stabs"] * 0.9 + bus["pad"] + leadR + bus["keys"] * 0.9 + bus["fx"])

    send = bus["stabs"] * 0.5 + bus["pad"] * 0.8 + bus["lead"] * 0.6 + bus["clap"] * 0.35 + bus["keys"] * 0.6
    send = hp(send, 250)
    L += fftconvolve(send, reverb_ir(3.4 if dark else 2.4, 101))[:N] * (0.45 if dark else 0.35)
    R += fftconvolve(send, reverb_ir(3.4 if dark else 2.4, 202))[:N] * (0.45 if dark else 0.35)

    # vinyl texture
    crackle = np.zeros(N)
    pos = rng.integers(0, N, size=int(N / SR * 9))
    crackle[pos] = rng.uniform(-1, 1, len(pos)) * (0.18 if dark else 0.12)
    crackle = bp(crackle, 1000, 8000) + lp(rng.standard_normal(N), 3000) * 0.0025
    L += crackle; R += np.roll(crackle, 300)

    # filter automation: intro opens, break is muffled, outro closes
    w = np.ones(N)
    def ramp_bars(b_from, b_to, v0, v1):
        a, z = int(b_from * bar * SR), int(b_to * bar * SR)
        w[a:z] = np.linspace(v0, v1, z - a)
    ramp_bars(0, 16, 0.0, 1.0)
    bb = secs.index("break")
    ramp_bars(bb, bb + 2, 1.0, 0.55)
    ramp_bars(bb + 2, bb + 16, 0.55, 0.7)
    ramp_bars(bb + 16, bb + 24, 0.7, 1.0)
    ramp_bars(nbars - 16, nbars, 1.0, 0.0)
    w[int(nbars * bar * SR):] = 0
    w = w ** 1.5
    out = []
    for ch in (L, R):
        muff = lp(ch, 450)
        ch = muff * (1 - w) + ch * w
        ch = hp(ch, 28)
        out.append(ch)
    st_ = np.stack(out)

    # master: loudness to ~-14 LUFS-ish, soft clip, fades
    rms = np.sqrt(np.mean(st_ ** 2))
    st_ *= 0.16 / rms
    st_ = np.tanh(st_ * 1.05) / np.tanh(1.05)
    st_ *= 0.95 / np.max(np.abs(st_))
    fi, fo = int(1.5 * SR), int(4 * SR)
    st_[:, :fi] *= np.linspace(0, 1, fi)
    end = int((nbars * bar + 1.0) * SR)
    st_ = st_[:, :end]
    st_[:, -fo:] *= np.linspace(1, 0, fo)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{idx + 1:02d}.wav")
    pcm = (st_.T * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes(pcm.tobytes())
    return idx, cfg["title"], end / SR


if __name__ == "__main__":
    args = sys.argv[1:]
    pl = args.pop(0) if args and args[0] in PLAYLISTS else "locked_in"
    tracks = PLAYLISTS[pl][0]
    ids = [int(a) - 1 for a in args] or list(range(len(tracks)))
    with Pool(min(4, len(ids))) as p:
        for i, title, dur in p.imap_unordered(render, [(pl, i) for i in ids]):
            print(f"{i + 1:02d} {title:<24} {int(dur // 60)}:{int(dur % 60):02d}", flush=True)
