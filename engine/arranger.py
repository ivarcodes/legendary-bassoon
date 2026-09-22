"""Arranger - dynamic structure, sidechain integration, macro/micro automation."""

import numpy as np
from scipy.signal import butter, sosfilt
from .core import SR, seconds_to_samples, fade_in, fade_out, envelope_follower

_SIDECHAIN_HP = butter(2, 100 / (SR / 2), btype="high", output="sos")


def _smooth_env(env, window=0.1):
    """Smooth envelope with moving average."""
    kern_n = seconds_to_samples(window)
    if kern_n < 2:
        return env
    kernel = np.ones(kern_n) / kern_n
    return np.convolve(env, kernel, mode='same')


def _build_section_envs(total_bars, bar_len, target, rng, structure='standard'):
    """Build gain envelopes per section with dynamic shapes."""
    g = {n: np.zeros(target) for n in ['drums', 'chords', 'bass', 'melody', 'texture', 'ambience']}

    section_map = {
        'intro': (0, 0.15),
        'verse': (0.15, 0.35),
        'chorus': (0.35, 0.55),
        'breakdown': (0.55, 0.7),
        'verse2': (0.7, 0.85),
        'outro': (0.85, 1.0),
    } if structure == 'standard' else {
        'intro': (0, 0.2),
        'groove': (0.2, 0.6),
        'breakdown': (0.6, 0.75),
        'groove2': (0.75, 0.9),
        'outro': (0.9, 1.0),
    }

    for section, (start_frac, end_frac) in section_map.items():
        s_bar = int(start_frac * total_bars)
        e_bar = int(end_frac * total_bars)
        s = s_bar * bar_len
        e = min(e_bar * bar_len, target)
        if s >= target:
            continue

        if section == 'intro':
            g['chords'][s:e] = np.linspace(0.3, 1.0, e-s)
            g['texture'][s:e] = 1.0
            g['ambience'][s:e] = 1.0
            g['drums'][s:e] = np.linspace(0, 0.6, e-s) if e_bar - s_bar > 4 else 0.0
            g['bass'][s:e] = np.linspace(0, 0.8, e-s)
            g['melody'][s:e] = np.linspace(0, 0.5, e-s)

        elif section in ('verse', 'groove'):
            g['chords'][s:e] = 1.0
            g['drums'][s:e] = 1.0
            g['bass'][s:e] = 1.0
            # Sparse melody - only half the verse, rests let chords breathe
            half = (e - s) // 2
            if half > 0:
                g['melody'][s:s + half] = 1.0
                g['melody'][s + half:e] = 0.15
            g['texture'][s:e] = 1.0
            g['ambience'][s:e] = 1.0

        elif section in ('chorus', 'groove2'):
            g['chords'][s:e] = 1.0
            g['drums'][s:e] = 1.0
            g['bass'][s:e] = 1.0
            g['melody'][s:e] = 1.0
            g['texture'][s:e] = 1.0
            g['ambience'][s:e] = 1.0

        elif section == 'breakdown':
            # Drums drop out first, then bass - chords + melody carry
            drop = max(1, (e - s) // 3)
            g['drums'][s:s + drop] = np.linspace(1.0, 0.0, drop)
            g['drums'][s + drop:e] = 0.0
            g['bass'][s:e] = np.linspace(1.0, 0.3, e-s)
            g['chords'][s:e] = 1.0
            g['melody'][s:e] = np.linspace(1.0, 0.7, e-s)
            g['texture'][s:e] = 1.0
            g['ambience'][s:e] = 1.0

        elif section == 'outro':
            g['chords'][s:e] = np.linspace(1.0, 0.0, e-s)
            g['drums'][s:e] = np.linspace(1.0, 0.0, e-s)
            g['bass'][s:e] = np.linspace(1.0, 0.0, e-s)
            g['melody'][s:e] = np.linspace(0.5, 0.0, e-s)
            g['texture'][s:e] = 1.0
            g['ambience'][s:e] = np.linspace(1.0, 0.5, e-s)

    return g


def _add_macro_automation(g, target, bar_len, total_bars, rng):
    """Slow macro-level automation for movement."""
    t = np.arange(target, dtype=np.float64) / SR

    for layer in ['drums', 'chords', 'bass', 'melody']:
        rate = rng.uniform(0.02, 0.08)
        depth = rng.uniform(0.001, 0.004)
        phase = rng.uniform(0, 2*np.pi)
        lfo = 1.0 + depth * np.sin(2.0 * np.pi * rate * t + phase)
        g[layer] *= lfo

    chord_brightness = 1.0 + 0.005 * np.sin(2.0 * np.pi * 0.04 * t + 1.2)
    g['chords'] *= chord_brightness


def _add_micro_drops(g, total_bars, bar_len, target, rng):
    """Rare element drops for variation."""
    if total_bars < 16:
        return

    num_drops = max(1, total_bars // 24)
    drop_bars = set()
    for _ in range(num_drops):
        bar = rng.randint(8, total_bars - 8)
        drop_bars.add(bar)

    for bar in drop_bars:
        s = bar * bar_len
        e = min(s + bar_len, target)
        if s >= target:
            continue
        # Drop melody most often (frees space), rarely drums for surprise
        drop_layer = rng.choice(['melody', 'melody', 'drums', 'bass'])
        g[drop_layer][s:e] = 0.0


def _add_fill_automation(g, total_bars, bar_len, target, rng):
    """Drum fills every 8/16 bars."""
    if total_bars < 16:
        return

    for bar in range(8, total_bars - 4, 16):
        if rng.random() < 0.7:
            s = bar * bar_len
            e = min(s + bar_len, target)
            if s >= target:
                continue
            fill_env = np.linspace(1.0, 1.4, e-s)
            g['drums'][s:e] *= fill_env


def arrange_stereo(drums_l, drums_r, chords_l, chords_r, bass_l, bass_r,
                   melody, texture, ambience, duration, bpm, seed=None):
    """Mix layers with dynamic envelopes, sidechain prep, macro/micro automation."""
    rng = np.random.RandomState(seed)
    target = seconds_to_samples(duration)
    bar_dur = 4.0 * (60.0 / bpm)
    one_bar = seconds_to_samples(bar_dur)
    total_bars = int(duration / bar_dur)

    def pad(x):
        if len(x) < target:
            return np.pad(x, (0, target - len(x)))
        return x[:target]

    drums_l, drums_r = pad(drums_l), pad(drums_r)
    chords_l, chords_r = pad(chords_l), pad(chords_r)
    bass_l, bass_r = pad(bass_l), pad(bass_r)
    melody = pad(melody)
    texture = pad(texture)
    ambience = pad(ambience) if ambience is not None else np.zeros(target)

    g = _build_section_envs(total_bars, one_bar, target, rng,
                            structure=rng.choice(['standard', 'looped']))

    _add_macro_automation(g, target, one_bar, total_bars, rng)
    _add_micro_drops(g, total_bars, one_bar, target, rng)
    _add_fill_automation(g, total_bars, one_bar, target, rng)

    for name in g:
        g[name] = _smooth_env(g[name], window=0.08)
        g[name] = np.clip(g[name], 0, 1.2)

    drums_mono = (drums_l + drums_r) * 0.5
    kick_trigger = sosfilt(_SIDECHAIN_HP, np.abs(drums_mono))
    kick_env = envelope_follower(kick_trigger, attack_ms=1.0, release_ms=40.0)
    kick_env = kick_env / (np.max(kick_env) + 1e-9)

    sidechain_amount = 0.25
    sidechain_gain = 1.0 - kick_env * sidechain_amount
    sidechain_gain = np.clip(sidechain_gain, 0.5, 1.0)
    g['chords'] *= sidechain_gain
    g['bass'] *= sidechain_gain
    g['melody'] *= sidechain_gain ** 0.5

    left = (drums_l * g['drums'] * 0.75 +
            chords_l * g['chords'] * 0.55 +
            bass_l * g['bass'] * 0.40 +
            melody * g['melody'] * 0.45 +
            texture * g['texture'] * 0.04 +
            ambience * g['ambience'] * 0.03)

    right = (drums_r * g['drums'] * 0.75 +
             chords_r * g['chords'] * 0.55 +
             bass_r * g['bass'] * 0.40 +
             melody * g['melody'] * 0.45 +
             texture * g['texture'] * 0.04 +
             ambience * g['ambience'] * 0.03)

    left = fade_in(left, 3.0)
    right = fade_in(right, 3.0)

    fade_bars = min(16, max(8, total_bars // 4))
    fade_n = int(fade_bars * bar_dur * SR)
    if fade_n < target:
        fade = np.linspace(1, 0, fade_n) ** 0.7
        left[-fade_n:] *= fade
        right[-fade_n:] *= fade

    return left, right, drums_l, drums_r