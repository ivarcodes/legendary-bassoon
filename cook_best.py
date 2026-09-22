"""Cook 3 signature tracks + one 50-minute seamless multi-mood mix."""

import os
import time
import numpy as np
import soundfile as sf

from engine.core import SR
from engine.generate import generate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "chef")
os.makedirs(OUT, exist_ok=True)

# My 3 picks: maximum contrast, strongest character
BEST3 = [
    # (label, mood, key, bpm lock or None, seed, ambience, duration)
    ("01_midnight_jazz", "jazzy", "D", None, 777001, "room", 300),
    ("02_rainy_window", "rainy", "E", None, 777002, "rain", 300),
    ("03_neon_dark", "dark", "A", None, 777003, "none", 300),
]

# 50-min mix: curated order for energy flow (10 x 5 min segments)
MIX_ORDER = [
    ("cozy", "A", "rain"),
    ("chill", "E", "room"),
    ("jazzy", "D", "none"),
    ("rainy", "G", "rain"),
    ("dreamy", "C", "rain"),
    ("sunset", "F", "room"),
    ("nostalgic", "A", "none"),
    ("melancholic", "D", "rain"),
    ("dark", "E", "none"),
    ("spacey", "C", "room"),
    ("uplift", "G", "none"),
    ("chill", "B", "rain"),
    ("cozy", "F", "room"),
    ("jazzy", "A", "none"),
    ("rainy", "D", "rain"),
    ("dreamy", "E", "room"),
    ("sunset", "B", "none"),
    ("nostalgic", "C", "rain"),
    ("dark", "G", "none"),
    ("hype", "A", "none"),
]
SEG_SEC = 150.0  # 2.5 min x 20 = 50 min
XF_SEC = 6.0     # equal-power crossfade


def _eq_power_fade(n):
    """Equal-power crossfade curves (cos/sin)."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return np.cos(t * np.pi / 2.0), np.sin(t * np.pi / 2.0)


def _rms_norm(x, target_rms=0.12):
    r = np.sqrt(np.mean(np.square(x)) + 1e-12)
    return x * (target_rms / r)


def _master_check(stereo, label):
    peak = float(np.max(np.abs(stereo)))
    rms = float(np.sqrt(np.mean(np.square(stereo))))
    # harshness proxy: energy above 8k relative
    mono = stereo.mean(axis=1)
    S = np.abs(np.fft.rfft(mono[::4]))
    f = np.fft.rfftfreq(len(mono[::4]), 4.0 / SR)
    hi = float(np.sum(S[f > 8000]) / (np.sum(S) + 1e-12))
    print(
        f"  {label}: peak={peak:.3f} rms={rms:.4f} "
        f"({20*np.log10(rms+1e-12):.1f} dBFS) hi_ratio={hi:.3f} "
        f"finite={np.isfinite(stereo).all()}",
        flush=True,
    )
    return peak, rms, hi


def _soft_ceiling(stereo, ceiling=0.89):
    """Gentle peak ceiling without crushing (research: ~-1 dBTP)."""
    peak = np.max(np.abs(stereo))
    if peak > ceiling:
        stereo = stereo * (ceiling / peak)
    # tiny safety soft-clip only on overs
    over = np.abs(stereo) > ceiling
    if np.any(over):
        stereo[over] = np.sign(stereo[over]) * (
            ceiling + (1.0 - ceiling) * np.tanh(
                (np.abs(stereo[over]) - ceiling) / (1.0 - ceiling)
            )
        )
    return stereo


def generate_best3():
    print("=== BEST 3 ===", flush=True)
    paths = []
    for label, mood, key, bpm, seed, amb, dur in BEST3:
        path = os.path.join(OUT, f"{label}.wav")
        t0 = time.time()
        left, right, _ = generate(
            duration=dur, bpm=bpm, mood=mood, key=key,
            ambience_style=amb, seed=seed,
        )
        st = np.column_stack((left, right))
        # Stage: normalize RMS then ceiling
        st = _rms_norm(st, target_rms=0.13)
        st = _soft_ceiling(st, ceiling=0.89)
        _master_check(st, label)
        sf.write(path, (st * 32767).astype(np.int16), SR, subtype="PCM_16")
        print(f"  wrote {path}  ({time.time()-t0:.1f}s)", flush=True)
        paths.append(path)
    return paths


def generate_mix50():
    print("=== 50 MIN MIX ===", flush=True)
    xf = int(XF_SEC * SR)
    # Render first segment full; subsequent overlap by xf
    total_sec = len(MIX_ORDER) * SEG_SEC - (len(MIX_ORDER) - 1) * XF_SEC
    print(f"  segments={len(MIX_ORDER)} seg={SEG_SEC}s xf={XF_SEC}s "
          f"total={total_sec/60:.1f} min", flush=True)

    # Pre-allocate in chunks via list then concat (memory ~50min stereo float64 ~1GB - use float32)
    pieces = []  # list of (start_sample, audio float32 stereo)

    for i, (mood, key, amb) in enumerate(MIX_ORDER):
        # Vary seeds hard
        seed = 90000 + i * 1337
        t0 = time.time()
        # Render slightly longer than segment so crossfade has material
        render_dur = SEG_SEC + (XF_SEC if i < len(MIX_ORDER) - 1 else 0.0)
        left, right, _ = generate(
            duration=render_dur, bpm=None, mood=mood, key=key,
            ambience_style=amb, seed=seed,
        )
        st = np.column_stack((left, right)).astype(np.float32)
        # Per-segment RMS normalize so loudness is consistent across mix
        st = _rms_norm(st, target_rms=0.12).astype(np.float32)

        start = int(round(i * (SEG_SEC - XF_SEC) * SR))
        pieces.append((start, st))
        print(
            f"  [{i+1:02d}/{len(MIX_ORDER)}] {mood:12s} key={key} "
            f"{time.time()-t0:.1f}s",
            flush=True,
        )

    # Assemble with equal-power crossfades
    total_n = pieces[-1][0] + len(pieces[-1][1])
    mix = np.zeros((total_n, 2), dtype=np.float64)

    for i, (start, st) in enumerate(pieces):
        n = len(st)
        end = start + n
        # Apply edge fades for seamless join
        audio = st.astype(np.float64).copy()
        if i > 0:
            # fade-in over xf (equal power) — previous segment already has fade-out region overlapping
            fi = min(xf, n)
            g_in, _ = _eq_power_fade(fi)
            # g_in goes 1->0 with cos; we want 0->1: use sin for incoming
            _, g_in = _eq_power_fade(fi)
            audio[:fi] *= g_in[:, None]
        if i < len(pieces) - 1:
            fo = min(xf, n)
            g_out, _ = _eq_power_fade(fo)
            audio[-fo:] *= g_out[:, None]

        # Overlap-add (regions that only one clip owns are full gain already)
        region = mix[start:end]
        # If overlap with previous, both already faded — simple add works for eq-power
        mix[start:end] = region + audio[:len(region)]

    # Global master: gentle RMS + ceiling, no noise boost
    mix = _rms_norm(mix, target_rms=0.12)
    mix = _soft_ceiling(mix, ceiling=0.89)

    # Final fades
    fade_in_n = int(4 * SR)
    fade_out_n = int(8 * SR)
    mix[:fade_in_n] *= np.linspace(0, 1, fade_in_n)[:, None] ** 0.7
    mix[-fade_out_n:] *= np.linspace(1, 0, fade_out_n)[:, None] ** 0.7

    peak, rms, hi = _master_check(mix, "MIX_50")
    path = os.path.join(OUT, "00_MIX_50MIN.wav")
    # Write PCM_16
    pcm = np.clip(mix, -1.0, 1.0)
    sf.write(path, (pcm * 32767).astype(np.int16), SR, subtype="PCM_16")
    print(f"  wrote {path}  size={os.path.getsize(path)/1e6:.1f} MB", flush=True)
    return path


if __name__ == "__main__":
    t0 = time.time()
    generate_best3()
    generate_mix50()
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
