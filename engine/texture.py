"""Texture - realistic vinyl crackle, tape hiss, optional rain ambience."""

import numpy as np
from scipy.signal import butter, sosfilt, lfilter
from .core import SR, seconds_to_samples, pink_noise, highpass, lowpass, bandpass, tape_saturate, wow_flutter


_VINYL_POP_SOS = butter(2, np.array([200, 8000]) / (SR / 2), btype="band", output="sos")
_VINYL_CRACKLE_SOS = butter(2, np.array([1000, 12000]) / (SR / 2), btype="band", output="sos")
_TAPE_HISS_SOS = butter(2, np.array([2000, 14000]) / (SR / 2), btype="band", output="sos")


def _generate_vinyl_pops(duration, rng, density=8.0):
    """Poisson-distributed vinyl pops with realistic shapes."""
    n = seconds_to_samples(duration)
    clicks = np.zeros(n, dtype=np.float64)

    n_clicks = int(duration * density)
    if n_clicks == 0:
        return clicks

    positions = np.cumsum(rng.exponential(n / n_clicks, n_clicks)).astype(int)
    positions = positions[positions < n - 100]

    for pos in positions:
        pop_type = rng.choice(['pop', 'crackle', 'tick'], p=[0.3, 0.5, 0.2])

        if pop_type == 'pop':
            length = rng.randint(30, 120)
            env = np.exp(-np.arange(length) * rng.uniform(80, 150))
            click = rng.randn(length) * env * rng.uniform(0.08, 0.25)
            click = sosfilt(_VINYL_POP_SOS, click)
        elif pop_type == 'crackle':
            length = rng.randint(15, 60)
            env = np.exp(-np.arange(length) * rng.uniform(200, 400))
            click = rng.randn(length) * env * rng.uniform(0.04, 0.15)
            click = sosfilt(_VINYL_CRACKLE_SOS, click)
        else:
            length = rng.randint(5, 20)
            env = np.exp(-np.arange(length) * rng.uniform(500, 1000))
            click = rng.randn(length) * env * rng.uniform(0.02, 0.08)

        end = min(pos + length, n)
        clicks[pos:end] += click[:end - pos]

    return clicks


def _generate_continuous_hiss(duration, rng, style='vinyl'):
    """Continuous background hiss - vinyl surface noise or tape hiss."""
    n = seconds_to_samples(duration)

    if style == 'vinyl':
        noise = pink_noise(n, rng) * 0.015
        noise = sosfilt(_VINYL_CRACKLE_SOS, noise)
        noise = highpass(noise, 800)

        mod_len = SR * 3
        for i in range(0, n, mod_len):
            end = min(i + mod_len, n)
            noise[i:end] *= rng.uniform(0.7, 1.3)

    elif style == 'tape':
        noise = rng.randn(n) * 0.01
        noise = sosfilt(_TAPE_HISS_SOS, noise)
        noise = highpass(noise, 1500)

        mod = 1.0 + 0.1 * np.sin(2.0 * np.pi * 0.3 * np.arange(n) / SR)
        noise *= mod

    else:
        noise = pink_noise(n, rng) * 0.008
        noise = bandpass(noise, 2000, 10000)

    return noise


def _generate_turntable_rumble(duration, rng):
    """Low-frequency turntable rumble."""
    n = seconds_to_samples(duration)
    t = np.arange(n, dtype=np.float64) / SR

    rumble = np.zeros(n)
    for freq in [25, 35, 50, 70]:
        amp = rng.uniform(0.0005, 0.002)
        phase = rng.uniform(0, 2 * np.pi)
        rumble += amp * np.sin(2.0 * np.pi * freq * t + phase)

    rumble += pink_noise(n, rng) * 0.0003
    return lowpass(rumble, 100)


def generate_vinyl(duration, volume=0.10, seed=None, style='worn'):
    """Vinyl crackle: pops + continuous hiss + rumble. Target ~-14 to -18 dBFS."""
    rng = np.random.RandomState(seed)
    gen_len = seconds_to_samples(4.0)

    pops = _generate_vinyl_pops(4.0, rng, density=rng.uniform(6, 12))
    hiss = _generate_continuous_hiss(4.0, rng, style='vinyl')
    rumble = _generate_turntable_rumble(4.0, rng)

    if style == 'clean':
        pops *= 0.3
        hiss *= 0.4
        rumble *= 0.2
    elif style == 'worn':
        pops *= 1.0
        hiss *= 1.0
        rumble *= 1.0
    elif style == 'heavy':
        pops *= 1.8
        hiss *= 1.5
        rumble *= 2.0

    texture = (pops + hiss + rumble) * volume

    texture = wow_flutter(texture, wow_rate=0.35, wow_depth=0.0005,
                          flutter_rate=4.0, flutter_depth=0.0001, rng=rng)

    target = seconds_to_samples(duration)
    reps = int(np.ceil(target / gen_len))
    return np.tile(texture, reps)[:target]


def generate_ambience(duration, style="rain", volume=0.025, seed=None):
    """Synthesized ambience: rain, room tone, or vinyl lead-in."""
    rng = np.random.RandomState(seed)
    gen_len = seconds_to_samples(5.0)

    noise = pink_noise(gen_len, rng)

    if style == "rain":
        bed = bandpass(noise, 800, 7000)
        mod_len = SR * 2
        for i in range(0, gen_len, mod_len):
            end = min(i + mod_len, gen_len)
            bed[i:end] *= rng.uniform(0.5, 1.0)
        bed = lowpass(bed, 5000)
        bed = tape_saturate(bed, drive=0.1)

    elif style == "room":
        bed = lowpass(noise, 300)
        bed = highpass(bed, 40)
        bed *= 0.3

    elif style == "vinyl_lead":
        bed = _generate_continuous_hiss(5.0, rng, style='vinyl') * 2.0
        bed = lowpass(bed, 8000)

    elif style == "tape":
        bed = _generate_continuous_hiss(5.0, rng, style='tape') * 1.5

    else:
        bed = noise * 0.5

    bed *= volume

    target = seconds_to_samples(duration)
    reps = int(np.ceil(target / gen_len))
    return np.tile(bed, reps)[:target]