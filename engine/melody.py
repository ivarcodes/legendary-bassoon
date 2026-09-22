"""Melody - chord-aware FM/epiano lead with motif, chord tones on strong beats."""

import numpy as np
from .core import SR, seconds_to_samples, midi_to_freq, adsr, highpass, lowpass, tape_saturate, soft_clip

# Chord tone intervals (relative to chord root) for strong-beat landings
_CHORD_TONES = [0, 4, 7, 11, 14, 17]  # root, 3rd, 5th, 7th, 9th, 13th (major)
_CHORD_TONES_MINOR = [0, 3, 7, 10, 14, 17]  # minor 3rd, minor 7th

# Passing tones for weak beats (scale steps)
_PASSING = [2, 5, 9, 12]  # 9th, 11th, 13th extensions


def _fm_epiano_note(freq, duration, velocity, rng, brightness=0.5):
    n = seconds_to_samples(duration)
    t = np.arange(n, dtype=np.float64) / SR

    carrier_freq = freq
    modulator_freq = freq * 2.001
    mod_index = brightness * 8.0 * velocity

    mod_env = adsr(duration, attack=0.005, decay=0.15, sustain=0.3, release=0.4)
    mod_signal = np.sin(2.0 * np.pi * modulator_freq * t) * mod_env * mod_index

    carrier_phase = 2.0 * np.pi * carrier_freq * t + mod_signal
    carrier = np.sin(carrier_phase)

    carrier_env = adsr(duration, attack=0.003, decay=0.25, sustain=0.4, release=0.5)
    carrier *= carrier_env * velocity

    harm_2 = 0.15 * np.sin(2.0 * np.pi * carrier_freq * 2 * t) * carrier_env * velocity
    harm_3 = 0.08 * np.sin(2.0 * np.pi * carrier_freq * 3 * t) * carrier_env * velocity

    note = carrier + harm_2 + harm_3
    note = tape_saturate(note, drive=0.2)
    note = soft_clip(note * 1.2) / 1.2
    return note


def _add_vibrato(note, rate=5.5, depth=0.006, delay=0.15):
    n = len(note)
    t = np.arange(n, dtype=np.float64) / SR
    vib_env = np.zeros(n)
    delay_n = int(delay * SR)
    if delay_n < n:
        vib_env[delay_n:] = 1.0 - np.exp(-(t[delay_n:] - delay) * 3.0)
    mod = 1.0 + vib_env * depth * np.sin(2.0 * np.pi * rate * t)
    idx = np.cumsum(mod)
    idx = np.clip(idx, 0, n - 1)
    return np.interp(np.arange(n), idx, note)


def generate_melody(duration, bpm, root_midi=60, prog=None, chord_dur=None,
                    seed=None, density="normal"):
    """
    Generate melody that follows the chord progression.
    
    Args:
        prog: list of chord MIDI note lists (e.g., [[57,60,64,67], ...])
        chord_dur: duration of each chord in seconds (defaults to 1 bar)
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    beat_dur = 60.0 / bpm
    bar_dur = 4.0 * beat_dur
    
    if prog is None:
        # Fallback: single chord (A minor)
        prog = [[57, 60, 64, 67]]
    if chord_dur is None:
        chord_dur = bar_dur  # 1 bar per chord by default

    n_chords = len(prog)
    loop_dur = n_chords * chord_dur
    loop_len = seconds_to_samples(loop_dur)
    total_loops = int(np.ceil(duration / loop_dur))

    target = seconds_to_samples(duration)
    out = np.zeros(target, dtype=np.float64)

    # Build a motif (repeating rhythmic/h melodic pattern)
    motif_len = rng.choice([3, 4, 5])
    motif_intervals = rng.choice([-2, -1, 0, 1, 2, 3], size=motif_len)
    motif_durations = rng.choice([0.5, 1.0, 1.5], size=motif_len, p=[0.3, 0.5, 0.2])

    density_mult = {"sparse": 0.6, "normal": 1.0, "dense": 1.4}.get(density, 1.0)
    phrase_chance = min(0.85, 0.55 * density_mult)

    for loop in range(total_loops):
        if rng.random() > phrase_chance and loop < total_loops - 1:
            continue

        # Generate 1-2 phrases per loop
        num_phrases = rng.choice([1, 2], p=[0.75, 0.25])
        
        for phrase_idx in range(num_phrases):
            # Beat offset within loop
            beat_offset = rng.choice([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
            
            # Number of notes in phrase
            num_notes = rng.choice([3, 4, 5, 6], p=[0.3, 0.4, 0.2, 0.1])
            
            # Start on a chord tone of first chord
            first_chord = prog[0]
            chord_root = first_chord[0]
            is_minor = chord_root % 12 in [0, 2, 3, 5, 7, 8, 10]  # rough check
            tones = _CHORD_TONES_MINOR if is_minor else _CHORD_TONES
            
            # Starting note: chord tone in upper register
            start_tone = rng.choice(tones[:4])  # root, 3rd, 5th, or 7th
            current_midi = chord_root + 12 + start_tone + rng.choice([0, 12])
            
            phrase_notes = []
            for ni in range(num_notes):
                # Determine which chord we're in based on beat position
                beat_pos = beat_offset + sum(nd / beat_dur for _, nd, _, _ in phrase_notes)
                chord_idx = int((beat_pos * beat_dur) / chord_dur) % n_chords
                active_chord = prog[chord_idx]
                active_root = active_chord[0]
                
                is_strong_beat = (beat_pos % 1.0) < 0.1 or (beat_pos % 2.0) < 0.1
                
                if is_strong_beat and rng.random() < 0.7:
                    # Land on chord tone of active chord
                    chord_tone = rng.choice(_CHORD_TONES[:5])
                    # Find nearest chord tone to current position
                    target_pitch_class = (active_root + chord_tone) % 12
                    while current_midi % 12 != target_pitch_class:
                        if current_midi % 12 < target_pitch_class:
                            current_midi += 1
                        else:
                            current_midi -= 1
                        # Keep in reasonable range
                        if current_midi < root_midi + 12:
                            current_midi += 12
                        elif current_midi > root_midi + 36:
                            current_midi -= 12
                else:
                    # Passing tone - step within scale
                    step = rng.choice([-2, -1, 1, 2])
                    current_midi = max(root_midi + 12, min(root_midi + 36, current_midi + step))
                
                # Duration
                if ni == 0:
                    dur_mult = rng.choice([0.5, 1.0, 1.5])
                else:
                    dur_mult = rng.choice([0.25, 0.5, 0.75, 1.0], p=[0.1, 0.4, 0.3, 0.2])
                dur = dur_mult * beat_dur
                
                vel = rng.uniform(0.4, 0.75)
                brightness = rng.uniform(0.3, 0.7)
                
                phrase_notes.append((current_midi, dur, vel, brightness))

            # Render phrase
            for i, (midi, dur, vel, bright) in enumerate(phrase_notes):
                freq = midi_to_freq(midi)
                note_sig = _fm_epiano_note(freq, dur, vel, rng, bright)
                note_sig = _add_vibrato(note_sig, rate=rng.uniform(4.5, 6.5),
                                       depth=rng.uniform(0.004, 0.009))
                note_sig = highpass(note_sig, 200)
                note_sig = lowpass(note_sig, 4500)

                n_comb = len(note_sig)
                note_pos = int(loop * loop_len + (beat_offset * beat_dur) * SR)
                end = min(note_pos + n_comb, target)
                if note_pos < target:
                    out[note_pos:end] += note_sig[:end - note_pos]

                beat_offset += dur / beat_dur

            # Rest between phrases
            if rng.random() < 0.3:
                beat_offset += rng.uniform(0.5, 1.0)

    out = highpass(out, 180)
    out = lowpass(out, 5500)
    out = tape_saturate(out, drive=0.1)
    out = soft_clip(out * 1.1) / 1.1

    return out
