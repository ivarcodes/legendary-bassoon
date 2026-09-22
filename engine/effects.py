"""Effects - mastering chain with sidechain, tape emulation, stereo reverb."""

import numpy as np
from scipy.signal import fftconvolve, sosfilt, butter, lfilter
from .core import SR, soft_clip, tape_saturate, sidechain_compress, wow_flutter, stereo_width, mid_side_encode, mid_side_decode

_LOWPASS_SOS = butter(4, 14000 / (SR / 2), btype="low", output="sos")
_DRUM_LOW_SOS = butter(2, 10000 / (SR / 2), btype="low", output="sos")
_DRUM_HIGH_SOS = butter(2, 40 / (SR / 2), btype="high", output="sos")
_SIDECHAIN_HP = butter(2, 100 / (SR / 2), btype="high", output="sos")

_reverb_ir_l = None
_reverb_ir_r = None


def _get_reverb_irs():
    global _reverb_ir_l, _reverb_ir_r
    if _reverb_ir_l is not None:
        return _reverb_ir_l, _reverb_ir_r
    decay = 0.35
    ir_len = SR // 5
    impulse = np.zeros(ir_len)
    impulse[0] = 1.0
    for delays, label in [([28.7, 36.1, 40.3, 43.2, 48.9], 'l'), ([30.2, 37.8, 41.8, 44.5, 49.7], 'r')]:
        co = np.zeros(ir_len)
        for dm in delays:
            dn = int(SR * dm / 1000.0)
            b = np.zeros(dn + 1)
            b[0] = 1.0
            a = np.zeros(dn + 1)
            a[0] = 1.0
            a[dn] = -decay
            co += lfilter(b, a, impulse) * 0.2
        ap_d = [5.0, 1.7, 0.5] if label == 'l' else [5.3, 1.9, 0.6]
        ap = co
        for dm in ap_d:
            dn = int(SR * dm / 1000.0)
            b = np.zeros(dn + 1)
            b[0] = -0.5
            b[dn] = 0.5
            a = np.zeros(dn + 1)
            a[0] = 1.0
            a[dn] = -0.5
            ap = lfilter(b, a, ap)
        if label == 'l':
            _reverb_ir_l = ap
        else:
            _reverb_ir_r = ap
    return _reverb_ir_l, _reverb_ir_r


def _warm_eq(sig):
    f0, Q, gain = 300.0, 1.0, 1.3
    w0 = 2.0 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2.0 * Q)
    b = np.array([1.0 + alpha * gain, -2.0 * np.cos(w0), 1.0 - alpha * gain])
    a = np.array([1.0 + alpha / gain, -2.0 * np.cos(w0), 1.0 - alpha / gain])
    return lfilter(b / a[0], a / a[0], sig)


def _bus_compress(sig, threshold=0.5, ratio=2.0, attack_ms=3.0, release_ms=80.0):
    abs_sig = np.abs(sig)
    atk = np.exp(-1.0 / (attack_ms / 1000.0 * SR))
    rel = np.exp(-1.0 / (release_ms / 1000.0 * SR))
    env = np.zeros_like(sig)
    env[0] = abs_sig[0]
    for i in range(1, len(sig)):
        if abs_sig[i] > env[i-1]:
            env[i] = atk * env[i-1] + (1 - atk) * abs_sig[i]
        else:
            env[i] = rel * env[i-1] + (1 - rel) * abs_sig[i]
    gain = np.ones_like(sig)
    over = env > threshold
    gain[over] = threshold + (env[over] - threshold) / ratio
    gain[over] /= env[over]
    return sig * gain


def _soft_limiter(sig, ceiling=0.92):
    peak = np.max(np.abs(sig))
    if peak > ceiling:
        sig = sig * (ceiling / peak)
    return soft_clip(sig * 1.3) / 1.3


def process_drum_bus(left, right):
    """Drum bus: dark filter + saturation + compression."""
    left = sosfilt(_DRUM_LOW_SOS, left)
    right = sosfilt(_DRUM_LOW_SOS, right)
    left = tape_saturate(left, drive=0.35)
    right = tape_saturate(right, drive=0.35)
    left = _bus_compress(left, threshold=0.4, ratio=2.5, attack_ms=1.0, release_ms=50.0)
    right = _bus_compress(right, threshold=0.4, ratio=2.5, attack_ms=1.0, release_ms=50.0)
    left = sosfilt(_DRUM_HIGH_SOS, left)
    right = sosfilt(_DRUM_HIGH_SOS, right)
    return left, right


def apply_effects_stereo(left, right, bpm=75, drums_for_sidechain=None, seed=None):
    """Mastering chain with sidechain, tape wow/flutter, stereo reverb."""

    if drums_for_sidechain is not None:
        kick_trigger = sosfilt(_SIDECHAIN_HP, np.abs(drums_for_sidechain))
        kick_trigger = envelope_follower = np.abs(kick_trigger)

        from .core import envelope_follower as ef
        kick_env = ef(kick_trigger, attack_ms=1.0, release_ms=40.0)
        kick_env = kick_env / (np.max(kick_env) + 1e-9)

        sidechain_amount = 0.35
        left = sidechain_compress(left, kick_env * sidechain_amount, threshold=0.2, ratio=3.0, attack_ms=0.5, release_ms=30.0)
        right = sidechain_compress(right, kick_env * sidechain_amount, threshold=0.2, ratio=3.0, attack_ms=0.5, release_ms=30.0)

    left = _warm_eq(left)
    right = _warm_eq(right)

    left = tape_saturate(left, drive=0.12)
    right = tape_saturate(right, drive=0.12)

    ir_l, ir_r = _get_reverb_irs()
    wet_l = fftconvolve(left, ir_l, mode="same")
    wet_r = fftconvolve(right, ir_r, mode="same")
    left = left * 0.82 + wet_l * 0.18
    right = right * 0.82 + wet_r * 0.18

    n = len(left)
    t = np.arange(n, dtype=np.float64) / SR
    mod = 1.0 + np.sin(2.0 * np.pi * 0.6 * t) * 0.0015
    mod += np.sin(2.0 * np.pi * 3.8 * t) * 0.0005
    idx = np.clip(np.cumsum(mod), 0, n - 1)
    left = np.interp(np.arange(n), idx, left)
    right = np.interp(np.arange(n), idx, right)

    fx_rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    wow_rate = fx_rng.uniform(0.3, 0.55)
    flutter_rate = fx_rng.uniform(3.8, 5.5)
    left = wow_flutter(left, wow_rate=wow_rate, wow_depth=fx_rng.uniform(0.0005, 0.0012),
                       flutter_rate=flutter_rate, flutter_depth=fx_rng.uniform(0.0001, 0.0003), rng=fx_rng)
    right = wow_flutter(right, wow_rate=fx_rng.uniform(0.3, 0.55), wow_depth=fx_rng.uniform(0.0005, 0.0012),
                        flutter_rate=fx_rng.uniform(3.8, 5.5), flutter_depth=fx_rng.uniform(0.0001, 0.0003), rng=fx_rng)

    left = _bus_compress(left, threshold=0.3, ratio=2.0, attack_ms=5.0, release_ms=120.0) * 2.5
    right = _bus_compress(right, threshold=0.3, ratio=2.0, attack_ms=5.0, release_ms=120.0) * 2.5

    width = fx_rng.uniform(1.1, 1.45)
    left, right = stereo_width(left, right, width=width)

    left = sosfilt(_LOWPASS_SOS, left)
    right = sosfilt(_LOWPASS_SOS, right)

    left = _soft_limiter(left, ceiling=0.92)
    right = _soft_limiter(right, ceiling=0.92)

    return left, right


def envelope_follower(sig, attack_ms=2.0, release_ms=100.0):
    n = len(sig)
    atk = np.exp(-1.0 / (attack_ms / 1000.0 * SR))
    rel = np.exp(-1.0 / (release_ms / 1000.0 * SR))
    env = np.zeros(n)
    abs_sig = np.abs(sig)
    env[0] = abs_sig[0]
    for i in range(1, n):
        if abs_sig[i] > env[i-1]:
            env[i] = atk * env[i-1] + (1 - atk) * abs_sig[i]
        else:
            env[i] = rel * env[i-1] + (1 - rel) * abs_sig[i]
    return env