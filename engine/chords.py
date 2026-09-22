"""Rhodes/Wurlitzer chords - authentic electric piano with extended jazz voicings."""

import numpy as np
from .core import SR, seconds_to_samples, soft_clip, highpass, bitcrush, tape_saturate, midi_to_freq

# All progressions in minor keys (lofi is predominantly minor)
# Each chord is [root, 3rd, 5th, 7th, 9th, 13th] - extended voicings
# Rootless voicings also supported (3rd, 5th, 7th, 9th, 13th)

PROGRESSIONS = {
    # Classic study lofi - im7, bVImaj7, iv7, V7
    "chill": [
        [57, 60, 64, 67, 71, 76],   # Am9 (A C E G B)
        [65, 69, 72, 76, 79, 84],   # Fmaj9 (F A C E G)
        [62, 65, 69, 72, 76, 81],   # Dm9 (D F A C E)
        [67, 71, 74, 77, 81, 86],   # G13 (G B D F A E)
    ],
    # Jazzy - ii-V-i with extensions
    "jazzy": [
        [62, 65, 69, 72, 76, 81],   # Dm9
        [67, 71, 74, 77, 81, 86],   # G13
        [57, 60, 64, 67, 71, 76],   # Am9
        [69, 72, 76, 79, 83, 88],   # E7b9 (V7 of Am)
    ],
    # Dreamy - bVImaj7, bVIImaj7, im9, IVmaj7
    "dreamy": [
        [65, 69, 72, 76, 79, 84],   # Fmaj9
        [67, 71, 74, 77, 81, 86],   # Gmaj9 (bVII in A minor context)
        [57, 60, 64, 67, 71, 76],   # Am9
        [69, 73, 76, 79, 84, 88],   # Dmaj9 (IV in A minor - bright)
    ],
    # Melancholic - im7, bIIImaj7, iv7, bVImaj7
    "melancholic": [
        [57, 60, 64, 67, 71],       # Am7 (rootless: C E G B)
        [62, 65, 69, 72, 76],       # Cmaj9 (bIII)
        [62, 65, 69, 72, 76],       # Dm7 (iv)
        [65, 69, 72, 76, 79],       # Fmaj7 (bVI)
    ],
    # Nostalgic - im9, bVImaj9, bVIImaj9, V7
    "nostalgic": [
        [57, 60, 64, 67, 71, 76],   # Am9
        [65, 69, 72, 76, 79, 84],   # Fmaj9
        [67, 71, 74, 77, 81, 86],   # Gmaj9
        [67, 71, 74, 77, 80, 85],   # G7b13
    ],
    # Dark/late night - im7, bVI7, iv7, V7alt
    "dark": [
        [52, 55, 59, 62, 66],       # Em7
        [60, 64, 67, 70, 74],       # C7 (bVI7)
        [57, 60, 64, 67, 71],       # Am7
        [59, 62, 66, 69, 72],       # B7alt
    ],
}

# Rootless voicings for smoother voice leading (3rd, 5th, 7th, 9th, 13th)
ROOTLESS = {
    "chill": [
        [60, 64, 67, 71, 76],   # Am9 rootless (C E G B)
        [69, 72, 76, 79, 84],   # Fmaj9 rootless (A C E G)
        [65, 69, 72, 76, 81],   # Dm9 rootless (F A C E)
        [71, 74, 77, 81, 86],   # G13 rootless (B D F A)
    ],
}

_fundamental_ratios = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0])
_fundamental_gains = np.array([1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02])


def _rhodes_tine_model(freq, duration, velocity, rng, detune_cents=0):
    """Physical model of a Rhodes tine - asymmetrical decay, inharmonicity."""
    n = seconds_to_samples(duration)
    t = np.arange(n, dtype=np.float64) / SR

    f = freq * (2.0 ** (detune_cents / 1200.0))

    inharm = 1.0 + rng.uniform(0.0002, 0.0008)
    ratios = _fundamental_ratios * inharm

    gains = _fundamental_gains * rng.uniform(0.85, 1.15, len(_fundamental_gains))
    gains[0] *= velocity

    sig = np.zeros(n, dtype=np.float64)
    for i, (ratio, gain) in enumerate(zip(ratios, gains)):
        decay_rate = 1.5 + ratio * 0.8
        sig += gain * np.sin(2.0 * np.pi * f * ratio * t) * np.exp(-t * decay_rate)

    pickup_pos = rng.uniform(0.15, 0.35)
    pickup_comb = np.sin(np.pi * pickup_pos * ratios)
    sig *= pickup_comb.sum() / len(ratios)

    return sig


def _rhodes_stereo_chord(midi_notes, duration, velocities, rng, pan=0.15):
    """Rhodes chord with stereo spread, key-off noise, stereo tremolo."""
    n = seconds_to_samples(duration)
    t = np.arange(n, dtype=np.float64) / SR

    freqs = np.array([midi_to_freq(m) for m in midi_notes])
    vels = velocities

    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    for i, (freq, vel) in enumerate(zip(freqs, vels)):
        detune = rng.uniform(-3, 3)
        note_l = _rhodes_tine_model(freq, duration, vel, rng, detune)
        note_r = _rhodes_tine_model(freq, duration, vel, rng, detune + rng.uniform(-1.5, 1.5))

        note_l = highpass(note_l, 80)
        note_r = highpass(note_r, 80)

        note_l = bitcrush(note_l, bits=12, downsample=1)
        note_r = bitcrush(note_r, bits=12, downsample=1)

        spread = 0.4 + i * 0.08
        angle = (pan + spread) * np.pi / 4.0
        left += note_l * np.cos(angle) * vel
        right += note_r * np.sin(angle) * vel

    # Stereo tremolo - subtle, slightly different L/R rates
    trem_rate = rng.uniform(4.2, 5.2)
    trem_depth = rng.uniform(0.08, 0.15)
    tmod = 1.0 - trem_depth * (1.0 - np.sin(2.0 * np.pi * trem_rate * t)) / 2.0
    left *= tmod
    right *= tmod

    trem_rate_l = trem_rate * rng.uniform(0.985, 1.015)
    trem_rate_r = trem_rate * rng.uniform(0.985, 1.015)
    tmod_l = 1.0 - trem_depth * 0.25 * (1.0 - np.sin(2.0 * np.pi * trem_rate_l * t)) / 2.0
    tmod_r = 1.0 - trem_depth * 0.25 * (1.0 - np.sin(2.0 * np.pi * trem_rate_r * t)) / 2.0
    left *= tmod_l
    right *= tmod_r

    # Soft attack
    atk = min(int(rng.uniform(0.004, 0.01) * SR), n)
    if atk > 0:
        left[:atk] *= np.linspace(0, 1, atk) ** 0.5
        right[:atk] *= np.linspace(0, 1, atk) ** 0.5

    # Key-off noise (mechanical release)
    key_off_gain = rng.uniform(0.01, 0.025)
    key_off = rng.randn(n) * key_off_gain * np.exp(-t * 40.0)
    left += key_off
    right += key_off

    # Tape saturation for warmth
    left = tape_saturate(left, drive=0.12)
    right = tape_saturate(right, drive=0.12)

    left = soft_clip(left)
    right = soft_clip(right)

    return left * 0.3, right * 0.3


def _make_chord(midi_notes, dur, velocity=0.45, rng=None, pan=0.15):
    if rng is None:
        rng = np.random.RandomState()
    # Velocity variation per note for human feel
    velocities = velocity * rng.uniform(0.85, 1.15, len(midi_notes))
    return _rhodes_stereo_chord(midi_notes, dur, velocities, rng, pan=pan)


def generate_loop(duration, style="chill", bpm=80, seed=None, pan=0.15,
                  use_rootless=False, bars_per_chord=2):
    """
    Generate chord loop with authentic lofi progressions.
    
    Returns: (left, right, prog, chord_dur_seconds)
        prog: list of chord MIDI note lists
        chord_dur: duration of each chord in seconds
    """
    rng = np.random.RandomState(seed)
    beat_dur = 60.0 / bpm
    bar_dur = 4.0 * beat_dur
    chord_dur = bars_per_chord * bar_dur  # Slower harmonic rhythm (2 bars per chord)

    prog = PROGRESSIONS.get(style, PROGRESSIONS["chill"])
    if use_rootless and style in ROOTLESS and rng.random() < 0.5:
        prog = ROOTLESS[style]

    loop_len = seconds_to_samples(len(prog) * chord_dur)
    ll = np.zeros(loop_len, dtype=np.float64)
    lr = np.zeros(loop_len, dtype=np.float64)

    pos = 0
    for chord_idx, midi_notes in enumerate(prog):
        # Slight timing variation per chord
        actual_dur = chord_dur * rng.uniform(0.92, 0.98)
        cl, cr = _make_chord(midi_notes, actual_dur, rng=rng, pan=pan + rng.uniform(-0.02, 0.02))
        end = min(pos + len(cl), loop_len)
        ll[pos:end] += cl[:end - pos]
        lr[pos:end] += cr[:end - pos]

        # Occasional ghost chord (quiet, higher inversion) - on beat 3 of second bar
        if bars_per_chord >= 2 and rng.random() < 0.3:
            ghost_notes = [n + 12 for n in midi_notes[1:4]]
            gcl, gcr = _make_chord(ghost_notes, beat_dur * 0.5, velocity=0.12, rng=rng, pan=-pan * 0.5)
            ghost_pos = min(pos + int((bar_dur + 2 * beat_dur) * SR), loop_len - len(gcl))
            if ghost_pos > pos:
                ll[ghost_pos:ghost_pos+len(gcl)] += gcl
                lr[ghost_pos:ghost_pos+len(gcr)] += gcr

        pos += seconds_to_samples(chord_dur)

    # Crossfade loop boundaries
    cf = int(0.02 * SR)
    if cf > 0 and loop_len > cf * 2:
        w = np.linspace(0, 1, cf) ** 0.5
        ll[:cf] *= w
        ll[-cf:] *= w[::-1]
        lr[:cf] *= w
        lr[-cf:] *= w[::-1]

    # High-pass to remove mud, leave room for bass
    ll = highpass(ll, 120)
    lr = highpass(lr, 120)

    target = seconds_to_samples(duration)
    reps = int(np.ceil(target / loop_len))
    return (np.tile(ll, reps)[:target], np.tile(lr, reps)[:target],
            prog, chord_dur)