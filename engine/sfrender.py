"""Render PrettyMIDI to stereo float audio via tinysoundfont + SoundFont."""

import os
import numpy as np
import tempfile
import tinysoundfont
import pretty_midi

from .core import SR

_SF_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "soundfonts", "GeneralUser-GS.sf2"),
    os.path.join(os.path.dirname(__file__), "..", "soundfonts", "FluidR3_GM.sf2"),
    r"D:\openS\lofi-generator\soundfonts\GeneralUser-GS.sf2",
]


def find_soundfont():
    for p in _SF_CANDIDATES:
        ap = os.path.abspath(p)
        if os.path.isfile(ap) and os.path.getsize(ap) > 100_000:
            return ap
    # Search nearby
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".sf2", ".sf3")):
                return os.path.join(dirpath, f)
    raise FileNotFoundError(
        "No SoundFont (.sf2) found. Place GeneralUser-GS.sf2 in soundfonts/"
    )


def render_midi(pm, duration, gain_db=-2.0, sample_rate=SR):
    """Render PrettyMIDI offline to (N, 2) float32 stereo."""
    return _render_pm(pm, duration, gain_db, sample_rate)


def render_stems(pm, duration, stem_names=None, gain_db=-2.0, sample_rate=SR):
    """Render each instrument as a separate stem. Returns dict name -> (N,2)."""
    stems = {}
    try:
        tempi, _ = pm.get_tempo_changes()
        tempo = float(tempi[0]) if len(tempi) and float(tempi[0]) > 0 else 120.0
    except Exception:
        tempo = 120.0
    if not tempo or tempo <= 0:
        tempo = 120.0

    for inst in pm.instruments:
        name = inst.name or f"inst_{inst.program}"
        if stem_names and name not in stem_names:
            continue
        sub = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        sub.instruments.append(inst)
        stems[name] = _render_pm(sub, duration, gain_db, sample_rate)
    return stems


def _render_pm(pm, duration, gain_db, sample_rate):
    import pretty_midi  # noqa: F401
    sf_path = find_soundfont()

    fd, midi_path = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    try:
        pm.write(midi_path)

        synth = tinysoundfont.Synth(gain=gain_db)
        sfid = synth.sfload(sf_path)
        if sfid < 0:
            raise RuntimeError(f"Failed to load soundfont: {sf_path}")

        # Auto-select drum kit on channel 9 if drums present
        seq = tinysoundfont.Sequencer(synth)
        seq.midi_load(midi_path)

        total_samples = int(duration * sample_rate) + sample_rate
        chunk = 2048
        buffers = []
        elapsed = 0.0
        max_time = duration + 1.0

        while elapsed < max_time:
            remaining = max_time - elapsed
            step_sec = min(chunk / sample_rate, remaining)
            step_samples = max(1, int(step_sec * sample_rate))

            seq.process(step_sec)
            view = synth.generate(step_samples)
            arr = np.frombuffer(view, dtype=np.float32).copy()
            buffers.append(arr)

            elapsed += step_sec
            if seq.is_empty() and elapsed > duration + 0.9:
                break

        if not buffers:
            raise RuntimeError("Synth produced no audio")

        audio = np.concatenate(buffers)
        n_frames = len(audio) // 2
        stereo = audio.reshape(n_frames, 2)

        want = int(duration * sample_rate)
        if n_frames >= want:
            stereo = stereo[:want]
        else:
            pad = np.zeros((want - n_frames, 2), dtype=np.float32)
            stereo = np.concatenate([stereo, pad], axis=0)

        return stereo.astype(np.float64)
    finally:
        try:
            os.remove(midi_path)
        except OSError:
            pass
