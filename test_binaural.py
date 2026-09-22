"""Verify binaural layer: L/R tone separation = beat, presets, parser, render."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from engine.core import SR
from engine.binaural import PRESETS, apply_binaural
from chat.parser import parse_user_input, format_chat_response

ok = True


def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        ok = False


def tone_peak(ch, sr, center, halfwin=8.0):
    """Strongest spectral peak near `center` Hz (binaural carrier tone)."""
    n = len(ch)
    S = np.abs(np.fft.rfft(ch))
    F = np.fft.rfftfreq(n, 1.0 / sr)
    m = (F >= center - halfwin) & (F <= center + halfwin)
    if not m.any():
        return float("nan")
    return float(F[m][np.argmax(S[m])])


def detect_beat(ch_l, ch_r, carrier, expected_beat):
    """Beat = separation of the L/R carrier tones (drift allows +/-8 Hz)."""
    f_l = tone_peak(ch_l, SR, carrier - 0.5 * expected_beat)
    f_r = tone_peak(ch_r, SR, carrier + 0.5 * expected_beat)
    return f_l, f_r, f_r - f_l


print("=== 1. Presets: L/R tone separation = beat ===")
n = SR * 8
for name, (beat, carrier, label) in PRESETS.items():
    if beat <= 0:
        continue
    l, r = apply_binaural(np.zeros(n), np.zeros(n), preset=name, seed=1)
    f_l, f_r, det = detect_beat(l, r, carrier, beat)
    err = abs(det - beat)
    check(err < 1.0,
          f"{name:6s} beat={beat:4.1f}Hz  L={f_l:.2f} R={f_r:.2f} "
          f"-> {det:.2f}Hz (err={err:.2f})  [{label}]")

print("=== 2. Safety / soothing levels ===")
l, r = apply_binaural(np.zeros(n), np.zeros(n), preset="focus",
                      level_db=-20.0, seed=2)
peak = max(np.abs(l).max(), np.abs(r).max())
check(peak <= 10 ** (-18 / 20) * 1.2, f"focus layer peak {peak:.4f} <= ~-18 dBFS")
check(np.isfinite(l).all() and np.isfinite(r).all(), "no NaN/Inf")
corr = np.corrcoef(l, r)[0, 1]
check(corr < 0.99, f"L/R not mono (corr={corr:.3f})")
check(abs(l[0]) < 1e-6 and abs(l[-1]) < 1e-6, "fade edges start/end at 0")

print("=== 3. Chat parser binaural detection ===")
cases = [
    ("make deep work focus music with binaural beats", "focus"),
    ("something to help me study for my exam", "study"),
    ("calm me down before bed, sleep music", "sleep"),
    ("meditation flow track", "flow"),
    ("sharp gamma attention please", "peak"),
    ("just a chill jazz beat", "off"),
]
for text, want in cases:
    p = parse_user_input(text)
    check(p.get("binaural") == want,
          f"'{text[:42]}' -> {p.get('binaural')} (want {want})")
resp = format_chat_response(parse_user_input("focus binaural track"))
check("Binaural: focus" in resp, "chat response mentions binaural")

print("=== 4. Full render with binaural (integration) ===")
from engine.generate import generate
left, right, sr = generate(
    duration=15.0, bpm=75, mood="chill", key="C",
    ambience_style="rain", seed=424242,
    binaural="focus", binaural_db=-20.0,
)
stereo = np.column_stack((left, right))
peak = float(np.max(np.abs(stereo)))
rms = float(np.sqrt(np.mean(stereo ** 2)))
check(np.isfinite(stereo).all(), "full render finite")
check(peak <= 1.0, f"peak {peak:.3f} <= 1.0 (no clip)")
check(0.01 < rms < 0.5, f"rms {rms:.4f} sane")
check(abs(len(left) / sr - 15.0) < 0.1, f"duration {len(left)/sr:.2f}s ~ 15s")

# Beat must survive the full mix: L/R carrier tones 15 Hz apart at ~200 Hz
mid_l = left[3 * SR:-3 * SR]
mid_r = right[3 * SR:-3 * SR]
f_l, f_r, det = detect_beat(mid_l, mid_r, carrier=200.0, expected_beat=15.0)
check(abs(det - 15.0) < 1.0,
      f"rendered beat {det:.2f} Hz ~ 15 (L={f_l:.1f} R={f_r:.1f})")

# Control: binaural off should NOT have a 15 Hz-separated tone pair at 200 Hz
l0, r0, _ = generate(
    duration=8.0, bpm=75, mood="chill", key="C",
    ambience_style="rain", seed=424242, binaural="off",
)
d0 = l0[2 * SR:-2 * SR] - r0[2 * SR:-2 * SR]
d1 = mid_l - mid_r
def band_energy(x, lo, hi):
    S = np.abs(np.fft.rfft(x)) ** 2
    F = np.fft.rfftfreq(len(x), 1.0 / SR)
    m = (F >= lo) & (F < hi)
    return float(S[m].sum())
# Energy in the L/R tone neighborhoods relative to total
e_on = band_energy(d1, 185, 215)
e_off = band_energy(d0, 185, 215)
tot_on = band_energy(d1, 1, 20000)
tot_off = band_energy(d0, 1, 20000)
r_on, r_off = e_on / tot_on, e_off / tot_off
check(r_on > r_off * 1.5,
      f"tone-pair energy ratio on={r_on:.4f} > off={r_off:.4f} x1.5")

print()
print("ALL PASS" if ok else "SOME FAILURES")
sys.exit(0 if ok else 1)
