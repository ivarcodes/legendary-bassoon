"""Vocal chop / melodic hook - disabled (user said too noisy)."""

import numpy as np


def generate_hook(duration, bpm, root_midi=60, style="chill"):
    """Disabled - returns silence."""
    return np.zeros(int(duration * 44100), dtype=np.float64)
