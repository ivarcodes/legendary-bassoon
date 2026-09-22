"""Bass - Deep sub bass with subtle electric color, sidechain-ready, warm attack."""

import numpy as np
from scipy.signal import butter, sosfilt
from .core import SR, seconds_to_samples, soft_clip, highpass, tape_saturate, midi_to_freq

_SUB_SOS = butter(4, 80 / (SR / 2), btype="low", output="sos")
_COLOR_LP = butter(2, 500 / (SR / 2), btype="low", output="sos")


def _sub_bass_note(freq, duration, velocity, rng, glide_from=None):
    """Pure sine sub-bass - the foundation of lofi low end."""
    n = seconds_to_samples(duration)
    t = np.arange(n, dtype=np.float64) / SR

    # Soft attack - no click, rounded
    attack_n = min(int(rng.uniform(0.015, 0.035) * SR), n)
    env = np.ones(n)
    if attack_n > 0:
        env[:attack_n] = np.linspace(0, 1, attack_n) ** 1.2

    # Glide from previous note (portamento)
    if glide_from is not None and glide_from > 0:
        glide_n = min(int(0.08 * SR), n)
        freqs = np.full(n, freq)
        if glide_n > 0:
            freqs[:glide_n] = np.linspace(glide_from, freq, glide_n)
        phase = 2.0 * np.pi * np.cumsum(freqs) / SR
    else:
        phase = 2.0 * np.pi * freq * t

    sub = np.sin(phase) * env * velocity * 0.9
    sub = sosfilt(_SUB_SOS, sub)
    return sub


def _electric_bass_note(freq, duration, velocity, rng):
    """Subtle electric bass color - fingerstyle, warm, very soft attack."""
    n = seconds_to_samples(duration)
    t = np.arange(n, dtype=np.float64) / SR

    # Very soft attack - fingerstyle
    attack_ms = rng.uniform(25, 45)
    attack_n = min(int(attack_ms / 1000.0 * SR), n)
    decay_rate = rng.uniform(1.5, 3.0)
    sustain_level = rng.uniform(0.5, 0.7)
    release_ms = rng.uniform(150, 250)
    release_n = min(int(release_ms / 1000.0 * SR), n)
    sustain_n = max(0, n - attack_n - release_n)

    env = np.zeros(n)
    if attack_n > 0:
        env[:attack_n] = np.linspace(0, 1, attack_n) ** 2.0  # Very slow attack
    if sustain_n > 0:
        env[attack_n:attack_n+sustain_n] = sustain_level * np.exp(-np.arange(sustain_n) / SR * decay_rate)
    if release_n > 0:
        rel_start = attack_n + sustain_n
        env[rel_start:rel_start+release_n] = np.linspace(env[rel_start-1] if rel_start > 0 else sustain_level, 0, release_n) ** 1.5

    fundamental = np.sin(2.0 * np.pi * freq * t) * env * velocity * 0.3

    # Minimal harmonics - just warmth
    harmonic_ratios = [2.0, 3.0]
    harmonic_gains = [0.15, 0.05]
    harmonics = np.zeros(n)
    for ratio, gain in zip(harmonic_ratios, harmonic_gains):
        h_env = env ** 1.2
        harmonics += gain * np.sin(2.0 * np.pi * freq * ratio * t) * h_env * velocity * 0.3

    # Tiny finger noise
    finger_noise = rng.randn(n) * np.exp(-t * 400) * velocity * 0.003

    note = fundamental + harmonics + finger_noise
    note = sosfilt(_COLOR_LP, note)
    note = tape_saturate(note, drive=0.1)
    note = soft_clip(note * 1.05) / 1.05
    return note


def generate_bass_for_chords(chord_roots_midi, bpm, duration, chord_dur=None,
                             delay_ms=0.0, seed=None, sidechain_env=None):
    """
    Generate sub bass + subtle electric color.
    
    Args:
        chord_roots_midi: root notes of each chord
        chord_dur: duration of each chord in seconds (defaults to 2 bars)
    
    Returns: (left, right, sub_out) - sub is separate for sidechain
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    beat_dur = 60.0 / bpm
    bar_dur = 4.0 * beat_dur
    if chord_dur is None:
        chord_dur = 2.0 * bar_dur  # 2 bars per chord - matches chords.py
    
    n_chords = len(chord_roots_midi)
    cycle_dur = n_chords * chord_dur
    cycle_len = seconds_to_samples(cycle_dur)

    # Sub bass: two octaves down from chord root (deep lofi sub)
    sub_freqs = np.array([midi_to_freq(m - 24) for m in chord_roots_midi])
    # Electric color: two octaves down
    color_freqs = np.array([midi_to_freq(m - 36) for m in chord_roots_midi])

    note_dur = chord_dur * rng.uniform(0.9, 0.96)

    sub_notes = []
    color_notes = []

    for i, (sub_freq, color_freq) in enumerate(zip(sub_freqs, color_freqs)):
        vel = rng.uniform(0.5, 0.75)
        glide_from = sub_freqs[i-1] if i > 0 else None
        sn = _sub_bass_note(sub_freq, note_dur, vel, rng, glide_from=glide_from)
        cn = _electric_bass_note(color_freq, note_dur, vel, rng)
        sub_notes.append(sn)
        color_notes.append(cn)

    # Build cycle
    cycle_sub = np.zeros(cycle_len, dtype=np.float64)
    cycle_color = np.zeros(cycle_len, dtype=np.float64)
    step = seconds_to_samples(chord_dur)

    for i in range(n_chords):
        start = i * step
        end = min(start + len(sub_notes[i]), cycle_len)
        cycle_sub[start:end] += sub_notes[i][:end - start]
        cycle_color[start:end] += color_notes[i][:end - start]

    # Stereo width only on color layer (sub stays mono)
    if delay_ms > 0:
        delay_n = max(1, int(delay_ms / 1000.0 * SR))
        color_left = cycle_color.copy()
        color_right = np.zeros(cycle_len, dtype=np.float64)
        color_right[delay_n:] = cycle_color[:-delay_n] if delay_n < cycle_len else 0
    else:
        color_left = cycle_color
        color_right = cycle_color

    # High-pass sub at 20Hz, color at 30Hz
    cycle_sub = highpass(cycle_sub, 20)
    color_left = highpass(color_left, 30)
    color_right = highpass(color_right, 30)

    # Tile to target duration
    target = seconds_to_samples(duration)
    repeats = int(np.ceil(target / cycle_len))

    sub_out = np.tile(cycle_sub, repeats)[:target]
    color_l = np.tile(color_left, repeats)[:target]
    color_r = np.tile(color_right, repeats)[:target]

    # Combine: sub is dominant, color is subtle
    left = sub_out * 0.85 + color_l * 0.15
    right = sub_out * 0.85 + color_r * 0.15

    return left, right, sub_out


def generate_bass_sidechain_trigger(bpm, duration, seed=None):
    """Generate kick-triggered sidechain envelope for bass ducking."""
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    target = seconds_to_samples(duration)
    beat_dur = 60.0 / bpm
    bar_dur = 4.0 * beat_dur
    bars = int(duration / bar_dur)

    env = np.zeros(target)
    for bar in range(bars):
        base = bar * seconds_to_samples(bar_dur)
        # Kick on beats 1 and 3
        for beat in [0.0, 2.0]:
            pos = int(base + beat * beat_dur * SR)
            if pos < target:
                # Fast attack, medium release envelope
                attack_n = int(0.002 * SR)
                release_n = int(0.12 * SR)
                end = min(pos + attack_n + release_n, target)
                if pos + attack_n < end:
                    env[pos:pos+attack_n] = np.linspace(0, 1, attack_n)
                if pos + attack_n < end:
                    env[pos+attack_n:end] = np.linspace(1, 0, end - pos - attack_n) ** 0.5

    return env