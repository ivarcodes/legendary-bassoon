"""Core DSP utilities - all shared processing functions."""

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, lfilter, hilbert

SR = 44100
_filter_cache = {}


def seconds_to_samples(s):
    return int(s * SR)


def midi_to_freq(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def lowpass(sig, cutoff):
    if cutoff >= SR / 2:
        return sig.copy()
    key = ("lp", int(cutoff))
    if key not in _filter_cache:
        _filter_cache[key] = butter(4, cutoff / (SR / 2), btype="low", output="sos")
    return sosfiltfilt(_filter_cache[key], sig)


def highpass(sig, cutoff):
    if cutoff <= 0:
        return sig.copy()
    key = ("hp", int(cutoff))
    if key not in _filter_cache:
        _filter_cache[key] = butter(2, cutoff / (SR / 2), btype="high", output="sos")
    return sosfiltfilt(_filter_cache[key], sig)


def bandpass(sig, lo, hi):
    key = ("bp", int(lo), int(hi))
    if key not in _filter_cache:
        _filter_cache[key] = butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
    return sosfiltfilt(_filter_cache[key], sig)


def soft_clip(sig):
    return np.tanh(sig)


def bitcrush(sig, bits=12, downsample=2):
    """Reduce bit depth and downsample for lo-fi grit."""
    levels = 2.0 ** bits
    crushed = np.round(sig * levels) / levels
    if downsample > 1:
        crushed = crushed[::downsample]
        crushed = np.interp(np.arange(len(sig)), np.arange(0, len(sig), downsample), crushed)
    return crushed


def tape_saturate(sig, drive=0.3):
    """Asymmetric tape-like saturation."""
    x = sig * (1.0 + drive)
    pos = np.tanh(x)
    neg = np.tanh(x * 0.8) * 0.9
    return np.where(x >= 0, pos, neg)


def pan_stereo(sig, pan):
    angle = (pan + 1.0) * np.pi / 4.0
    return sig * np.cos(angle), sig * np.sin(angle)


def stereo_delay_width(sig, delay_n):
    left = sig.copy()
    right = np.zeros(len(sig), dtype=np.float64)
    if delay_n < len(sig):
        right[delay_n:] = sig[:-delay_n]
    return left, right


def pink_noise(n, rng=None):
    """1/f noise via Voss-McCartney."""
    if rng is None:
        rng = np.random.RandomState()
    white = rng.randn(n)
    pink = np.cumsum(white)
    pink = pink / np.max(np.abs(pink)) * 0.5
    return pink


def adsr(duration, attack=0.01, decay=0.1, sustain=0.7, release=0.1):
    n = seconds_to_samples(duration)
    env = np.zeros(n, dtype=np.float64)
    a = min(int(attack * SR), n)
    d = min(int(decay * SR), max(0, n - a))
    r = min(int(release * SR), max(0, n - a - d))
    s = max(0, n - a - d - r)
    p = 0
    if a > 0:
        env[p:p+a] = np.linspace(0, 1, a)
        p += a
    if d > 0:
        env[p:p+d] = np.linspace(1, sustain, d)
        p += d
    if s > 0:
        env[p:p+s] = sustain
        p += s
    if r > 0:
        env[p:p+r] = np.linspace(sustain, 0, r)
    return env


def fade_in(sig, sec):
    n = seconds_to_samples(sec)
    if n >= len(sig):
        return sig.copy()
    out = sig.copy()
    out[:n] *= np.linspace(0, 1, n)
    return out


def fade_out(sig, sec):
    n = seconds_to_samples(sec)
    if n >= len(sig):
        return sig.copy()
    out = sig.copy()
    out[-n:] *= np.linspace(1, 0, n)
    return out


def envelope_follower(sig, attack_ms=2.0, release_ms=100.0):
    """Fast envelope follower for sidechain."""
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


def sidechain_compress(signal, trigger, threshold=0.3, ratio=4.0, attack_ms=1.0, release_ms=50.0, makeup=1.0):
    """Sidechain compression: signal ducked by trigger envelope."""
    trigger_env = envelope_follower(trigger, attack_ms, release_ms)
    gain = np.ones_like(signal)
    over = trigger_env > threshold
    gain[over] = 1.0 - (trigger_env[over] - threshold) * (1.0 - 1.0/ratio)
    gain = np.clip(gain, 1.0/ratio, 1.0)
    return signal * gain * makeup


def wow_flutter(sig, wow_rate=0.5, wow_depth=0.001, flutter_rate=5.0, flutter_depth=0.0003, rng=None):
    """Tape wow (slow) and flutter (fast) modulation."""
    if rng is None:
        rng = np.random.RandomState()
    n = len(sig)
    t = np.arange(n, dtype=np.float64) / SR

    wow = wow_depth * np.sin(2.0 * np.pi * wow_rate * t + rng.uniform(0, 2*np.pi))
    flutter = flutter_depth * np.sin(2.0 * np.pi * flutter_rate * t + rng.uniform(0, 2*np.pi))
    mod = 1.0 + wow + flutter

    idx = np.cumsum(mod)
    idx = np.clip(idx, 0, n - 1)
    return np.interp(np.arange(n), idx, sig)


def mid_side_encode(left, right):
    """Convert L/R to Mid/Side."""
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    return mid, side


def mid_side_decode(mid, side):
    """Convert Mid/Side to L/R."""
    left = mid + side
    right = mid - side
    return left, right


def stereo_width(left, right, width=1.0):
    """Adjust stereo width via mid/side."""
    mid, side = mid_side_encode(left, right)
    side *= width
    return mid_side_decode(mid, side)