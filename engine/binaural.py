"""Binaural beat layer for focus / calm overlays.

Research notes (2021-2026 studies):
- Beat rate = |L - R| Hz; true stereo isolation (headphones) required.
- Useful bands: delta 0.5-4 (sleep), theta 4-8 (calm + focus; 6 Hz
  best supported for both), alpha 8-13 (relaxed study / relaxed
  alertness), beta 13-30 (active focus), gamma 40 (mixed evidence,
  can feel harsher).
- Carrier 100-400 Hz preferred (studies often 208-340 Hz); low
  carrier improved attention vs high carrier in controlled work.
- Mix quietly: ~-18 to -24 dB under music so it is felt not heard;
  pink-noise masking improves comfort. Avoid crushing the master
  after adding BB (phase matters). Safe listen: moderate volume,
  15-60 min sessions; not while driving; epilepsy caution.
"""

import numpy as np
from .core import SR

# preset name -> (beat_hz, carrier_hz, label)
# Carriers kept 180-220 Hz (warm, non-piercing); beat bands per research:
#   delta 0.5-4 sleep | theta 4-8 calm/creativity | alpha 8-13 relaxed
#   focus | beta 13-30 active focus (15 Hz best-evidenced for working
#   memory: Beauchene 2016/2017; Lane 1998) | gamma 40 sharp attention
#   (Reedijk 2013, Melnichuk 2025).
PRESETS = {
    "off":    (0.0, 0.0,  "No binaural beats"),
    "sleep":  (2.0, 200.0, "Delta 2 Hz — wind down / sleep onset"),
    "calm":   (4.0, 200.0, "Theta 4 Hz — deep calm"),
    "flow":   (6.0, 200.0, "Theta 6 Hz — calm focus (best-studied)"),
    "study":  (10.0, 220.0, "Alpha 10 Hz — relaxed alert / study"),
    "clear":  (12.0, 180.0, "Alpha 12 Hz — light focus"),
    "focus":  (15.0, 200.0, "Beta 15 Hz — deep work / active focus"),
    "peak":   (40.0, 200.0, "Gamma 40 Hz — sharp attention (short sessions)"),
}


def make_binaural(
    n_samples,
    beat_hz=6.0,
    carrier_hz=200.0,
    level_db=-20.0,
    fade_sec=4.0,
    pink_mask=0.35,
    seed=None,
):
    """Return (left, right) float64 for a binaural layer of n_samples.

    level_db: level relative to a full-scale sine (approx. how far under
    the music it sits when music is near 0.7 peak). Keep -18..-24.
    pink_mask: 0..1 amount of soft pink noise under the carrier so pure
    tones feel less piercing over long sessions.
    """
    if n_samples <= 0 or beat_hz <= 0 or carrier_hz <= 0:
        return np.zeros(n_samples), np.zeros(n_samples)

    rng = np.random.RandomState(seed if seed is not None else 42)
    t = np.arange(n_samples, dtype=np.float64) / float(SR)

    # Symmetric carriers so perceived pitch stays near carrier_hz
    f_l = carrier_hz - 0.5 * beat_hz
    f_r = carrier_hz + 0.5 * beat_hz

    # Shared micro-drift (same signal both channels): keeps the pure
    # sine from sounding sterile while |f_r - f_l| stays exactly
    # beat_hz — independent per-channel drift smears the beat by
    # ~drift amplitude and breaks entrainment.
    drift = 0.015 * carrier_hz
    d = drift * np.sin(2.0 * np.pi * 0.07 * t + rng.uniform(0, 2 * np.pi))

    left = np.sin(2.0 * np.pi * np.cumsum(f_l + d) / SR)
    right = np.sin(2.0 * np.pi * np.cumsum(f_r + d) / SR)

    # Soft pink-ish noise mask (helps palatability)
    if pink_mask > 0:
        n_l = _pink_noise(n_samples, rng)
        n_r = _pink_noise(n_samples, rng)
        left = left + pink_mask * n_l
        right = right + pink_mask * n_r

    # Amplitude: sine peak -> level_db, then fade edges so it never clicks
    amp = 10.0 ** (level_db / 20.0)
    left *= amp
    right *= amp

    fade_n = int(max(0.5, fade_sec) * SR)
    fade_n = min(fade_n, n_samples // 2) if n_samples >= 4 else 0
    if fade_n > 1:
        ramp = np.linspace(0.0, 1.0, fade_n) ** 1.5
        left[:fade_n] *= ramp
        right[:fade_n] *= ramp
        left[-fade_n:] *= ramp[::-1]
        right[-fade_n:] *= ramp[::-1]

    # Ensure true stereo separation (no accidental mono sum of carriers)
    # by removing any shared DC / near-identical energy is NOT done here —
    # that would break binaural. Only strip DC offset from pink noise.
    left -= np.mean(left)
    right -= np.mean(right)
    return left, right


def _pink_noise(n, rng):
    """1/f-ish noise via filtered white (cheap, smooth enough under a tone)."""
    from scipy.signal import butter, sosfilt
    white = rng.randn(n)
    # gentle low-pass tilt toward pink/brown character
    sos = butter(2, [80.0 / (SR / 2), 6000.0 / (SR / 2)], btype="band", output="sos")
    pink = sosfilt(sos, white)
    peak = np.max(np.abs(pink)) + 1e-12
    return pink / peak


def apply_binaural(left, right, preset="off", beat_hz=None, carrier_hz=None,
                   level_db=-20.0, seed=None, pink_mask=0.35):
    """Mix binaural layer into existing stereo pair (non-destructive sum)."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    n = len(left)

    if beat_hz is None or carrier_hz is None:
        b, c, _ = PRESETS.get(preset, PRESETS["off"])
        if beat_hz is None:
            beat_hz = b
        if carrier_hz is None:
            carrier_hz = c

    if not beat_hz or beat_hz <= 0:
        return left, right

    bb_l, bb_r = make_binaural(
        n,
        beat_hz=float(beat_hz),
        carrier_hz=float(carrier_hz),
        level_db=level_db,
        fade_sec=4.0,
        pink_mask=pink_mask,
        seed=seed,
    )
    return left + bb_l, right + bb_r
