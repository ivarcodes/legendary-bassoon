"""Generate lofi track - compose MIDI, render SoundFont stems, master."""

import numpy as np
import os
import tempfile
import soundfile as sf

from .core import SR, seconds_to_samples, fade_in
from .compose import compose_midi
from .sfrender import render_stems, render_midi
from .texture import generate_vinyl, generate_ambience
from .effects import apply_effects_stereo
from .binaural import apply_binaural


def _stem_env(duration, bpm, name, rng, structure=None):
    """Per-stem arrangement envelope (randomized structure per track)."""
    target = seconds_to_samples(duration)
    env = np.ones(target, dtype=np.float64)
    bpm = bpm if (bpm and bpm > 0) else 75
    bar = 4.0 * 60.0 / bpm
    total_bars = max(1, int(duration / bar))
    st = structure or {}
    intro = st.get("intro_frac", 0.10)
    brk = st.get("break_start", 0.55)
    brk_len = st.get("break_len", 0.10)
    outro = st.get("outro_start", 0.90)

    def ramp(a_f, b_f, v0, v1):
        a = int(a_f * target)
        b = min(int(b_f * target), target)
        if b > a:
            env[a:b] = np.linspace(v0, v1, b - a)

    if name == "drums":
        ramp(0.0, intro, 0.0, 1.0)
        ramp(brk, brk + brk_len, 1.0, 0.0)
        ramp(brk + brk_len, min(1.0, brk + brk_len + 0.08), 0.0, 1.0)
        ramp(outro, 1.0, 1.0, 0.0)
    elif name == "bass":
        ramp(0.0, intro * 0.8, 0.0, 1.0)
        ramp(brk, brk + brk_len, 1.0, 0.25)
        ramp(brk + brk_len, min(1.0, brk + brk_len + 0.08), 0.25, 1.0)
        ramp(outro, 1.0, 1.0, 0.0)
    elif name == "melody":
        env[:] = 0.0
        ramp(intro, min(1.0, intro + 0.08), 0.0, 1.0)
        ramp(brk, min(1.0, brk + brk_len + 0.05), 1.0, 0.7)
        ramp(outro, 1.0, 1.0, 0.0)
        # micro drops
        n_drops = int(st.get("drop_bars", 1))
        for _ in range(max(0, n_drops)):
            if total_bars >= 12:
                b0 = int(rng.randint(6, max(7, total_bars - 4)))
                s = int(b0 * bar * SR)
                e = min(s + int(bar * SR), target)
                if s < target:
                    env[s:e] = 0.0
    elif name in ("rhodes", "pad", "vibes", "strings"):
        ramp(0.0, intro * 0.6, 0.4, 1.0)
        ramp(outro + 0.02, 1.0, 1.0, 0.0)

    # smooth
    kern = max(2, int(0.06 * SR))
    kernel = np.ones(kern) / kern
    env = np.convolve(env, kernel, mode="same")
    return np.clip(env, 0.0, 1.2)


def _render_all(duration, bpm, mood, key, ambience_style, seed, progress_callback,
                binaural="off", binaural_db=-20.0):
    # Always use a unique seed so no two tracks share DNA
    if seed is None:
        seed = int(np.random.RandomState().randint(0, 2**31 - 1))
    rng = np.random.RandomState(seed)
    # Derive child seeds so texture/effects never collide either
    seed_comp = int(rng.randint(0, 2**31 - 1))
    seed_tex = int(rng.randint(0, 2**31 - 1))
    seed_amb = int(rng.randint(0, 2**31 - 1))
    seed_fx = int(rng.randint(0, 2**31 - 1))

    if progress_callback:
        progress_callback(0.05, "Composing MIDI...")
    result = compose_midi(duration, bpm=bpm, mood=mood, key=key, seed=seed_comp)
    if len(result) == 3:
        pm, prog, dna = result
    else:
        pm, prog = result
        dna = {}
    structure = dna.get("structure") if isinstance(dna, dict) else None
    # Actual tempo used (for envelopes / sidechain timing)
    actual_bpm = bpm
    try:
        tempi, _ = pm.get_tempo_changes()
        if tempi is not None and len(tempi) > 0:
            candidate = float(tempi[0])
            if candidate > 0:
                actual_bpm = candidate
    except Exception:
        pass
    if not actual_bpm or actual_bpm <= 0:
        from .compose import MOOD_DNA
        actual_bpm = MOOD_DNA.get(mood, {}).get("default_bpm", 75)
    if not actual_bpm or actual_bpm <= 0:
        actual_bpm = 75

    if progress_callback:
        progress_callback(0.2, "Rendering stems...")

    # Render each instrument separately for mix control
    try:
        stems = render_stems(pm, duration, gain_db=-6.0, sample_rate=SR)
    except Exception:
        # Fallback: single mixed render
        if progress_callback:
            progress_callback(0.3, "Rendering mix...")
        stereo = render_midi(pm, duration, gain_db=-4.0, sample_rate=SR)
        stems = {"rhodes": stereo}

    if progress_callback:
        progress_callback(0.55, "Mixing...")

    target = seconds_to_samples(duration)
    left = np.zeros(target)
    right = np.zeros(target)

    # Mood-dependent mix balance - further differentiates character
    mood_mix = {
        "chill":     {"drums": 1.1, "bass": 1.0, "rhodes": 0.9, "melody": 0.7, "pad": 0.4, "vibes": 0.55, "strings": 0.3},
        "jazzy":     {"drums": 0.9, "bass": 1.1, "rhodes": 1.0, "melody": 0.85, "pad": 0.2, "vibes": 0.5, "strings": 0.2},
        "dreamy":    {"drums": 0.6, "bass": 0.8, "rhodes": 0.8, "melody": 0.75, "pad": 0.7, "vibes": 0.6, "strings": 0.55},
        "melancholic":{"drums": 0.95, "bass": 1.0, "rhodes": 0.95, "melody": 0.65, "pad": 0.5, "vibes": 0.3, "strings": 0.55},
        "nostalgic": {"drums": 1.15, "bass": 1.05, "rhodes": 0.9, "melody": 0.75, "pad": 0.4, "vibes": 0.5, "strings": 0.25},
        "dark":      {"drums": 1.2, "bass": 1.15, "rhodes": 0.7, "melody": 0.55, "pad": 0.65, "vibes": 0.2, "strings": 0.2},
        "uplift":    {"drums": 1.1, "bass": 1.0, "rhodes": 0.85, "melody": 0.9, "pad": 0.5, "vibes": 0.6, "strings": 0.5},
        "rainy":     {"drums": 0.75, "bass": 0.9, "rhodes": 0.95, "melody": 0.7, "pad": 0.55, "vibes": 0.55, "strings": 0.3},
        "sunset":    {"drums": 1.0, "bass": 1.0, "rhodes": 0.95, "melody": 0.8, "pad": 0.45, "vibes": 0.5, "strings": 0.45},
        "hype":      {"drums": 1.3, "bass": 1.2, "rhodes": 0.75, "melody": 0.95, "pad": 0.15, "vibes": 0.5, "strings": 0.15},
        "spacey":    {"drums": 0.4, "bass": 0.85, "rhodes": 0.7, "melody": 0.6, "pad": 0.85, "vibes": 0.65, "strings": 0.6},
        "cozy":      {"drums": 1.05, "bass": 1.0, "rhodes": 1.0, "melody": 0.75, "pad": 0.4, "vibes": 0.5, "strings": 0.25},
    }
    gains = dict(mood_mix.get(mood, {
        "drums": 1.15, "bass": 1.0, "rhodes": 0.85,
        "melody": 0.75, "pad": 0.45, "vibes": 0.55, "strings": 0.40,
    }))
    # Per-track mix jitter so balance never repeats
    for k in gains:
        gains[k] = float(np.clip(gains[k] * rng.uniform(0.85, 1.15), 0.1, 1.5))

    drums_l = np.zeros(target)
    drums_r = np.zeros(target)

    for name, stereo in stems.items():
        # pad/trim
        sl = stereo[:, 0]
        sr_ = stereo[:, 1]
        if len(sl) < target:
            sl = np.pad(sl, (0, target - len(sl)))
            sr_ = np.pad(sr_, (0, target - len(sr_)))
        else:
            sl = sl[:target]
            sr_ = sr_[:target]

        env = _stem_env(duration, actual_bpm, name, rng, structure=structure)
        g = gains.get(name, 0.7)
        left += sl * env * g
        right += sr_ * env * g

        if name == "drums":
            drums_l = sl * env * g
            drums_r = sr_ * env * g

    # Kick sidechain from drum stem
    if np.any(drums_l):
        from .effects import envelope_follower
        from scipy.signal import butter, sosfilt
        hp = butter(2, 100 / (SR / 2), btype="high", output="sos")
        mono = (drums_l + drums_r) * 0.5
        trig = sosfilt(hp, np.abs(mono))
        ke = envelope_follower(trig, attack_ms=1.0, release_ms=45.0)
        ke = ke / (np.max(ke) + 1e-9)
        duck = 1.0 - float(rng.uniform(0.18, 0.38)) * ke
        # duck non-drum content: approximate by ducking full then adding drums back
        left = (left - drums_l) * duck + drums_l
        right = (right - drums_r) * duck + drums_r

    if progress_callback:
        progress_callback(0.7, "Texture...")

    texture = generate_vinyl(duration, seed=seed_tex)
    if ambience_style and ambience_style not in ("none", ""):
        amb = generate_ambience(duration, style=ambience_style, seed=seed_amb)
    else:
        amb = np.zeros(target)

    tex_gain = float(rng.uniform(0.04, 0.11))
    amb_gain = float(rng.uniform(0.015, 0.05))
    left += texture * tex_gain + amb * amb_gain
    right += texture * tex_gain + amb * amb_gain

    if progress_callback:
        progress_callback(0.8, "Mastering...")

    from .effects import apply_effects_stereo as _fx
    left, right = _fx(
        left, right,
        bpm=actual_bpm if actual_bpm and actual_bpm > 0 else 75,
        seed=seed_fx,
    )

    # Binaural overlay AFTER mastering so tones keep clean L/R phase
    if binaural and binaural not in ("off", ""):
        if progress_callback:
            progress_callback(0.85, f"Binaural ({binaural})...")
        left, right = apply_binaural(
            left, right, preset=binaural,
            level_db=binaural_db, seed=int(rng.randint(0, 2**31 - 1)),
        )

    left = fade_in(left, float(rng.uniform(1.5, 3.5)))
    right = fade_in(right, float(rng.uniform(1.5, 3.5)))
    fade_n = min(int(8 * 4 * (60.0 / (actual_bpm if actual_bpm > 0 else 75)) * SR), target // 2)
    if fade_n > 64:
        f = np.linspace(1, 0, fade_n) ** float(rng.uniform(0.6, 0.9))
        left[-fade_n:] *= f
        right[-fade_n:] *= f

    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak > 0:
        left = left / peak * 0.72
        right = right / peak * 0.72

    return left, right


def generate(duration=120.0, bpm=None, mood="chill", key=None,
             ambience_style="rain", progress_callback=None, seed=None,
             binaural="off", binaural_db=-20.0):
    """bpm=None uses a randomized mood tempo. seed=None = fully unique track.

    binaural: 'off' | 'sleep' | 'calm' | 'flow' | 'study' | 'clear'
              | 'focus' | 'peak'
    (headphone-only layer; keep binaural_db around -18..-24).
    """
    left, right = _render_all(duration, bpm, mood, key, ambience_style,
                              seed, progress_callback,
                              binaural=binaural, binaural_db=binaural_db)
    if progress_callback:
        progress_callback(1.0, "Done!")
    return left, right, SR


def generate_to_wav(filepath, duration=120.0, bpm=None, mood="chill", key=None,
                    ambience_style="rain", progress_callback=None, seed=None,
                    binaural="off", binaural_db=-20.0):
    """bpm=None uses a randomized mood tempo. seed=None = fully unique track."""
    left, right, sr = generate(
        duration=duration, bpm=bpm, mood=mood, key=key,
        ambience_style=ambience_style, progress_callback=progress_callback,
        seed=seed, binaural=binaural, binaural_db=binaural_db,
    )

    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Audio render produced NaN or infinite samples")

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    output_dir = os.path.dirname(os.path.abspath(filepath)) or "."
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(prefix=".lofi_render_", suffix=".wav", dir=output_dir)
        os.close(fd)
        pcm = (np.clip(np.column_stack((left, right)), -1.0, 1.0) * 32767.0).astype(np.int16)
        sf.write(temp_path, pcm, sr, subtype="PCM_16")
        os.replace(temp_path, filepath)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    return filepath
