"""Compose full lofi tracks as multi-track MIDI with mood-specific DNA."""

import numpy as np
import pretty_midi

# Progressions: lists of (chord_name, [midi notes])
PROGRESSIONS = {
    "chill": [
        ("Am9", [57, 60, 64, 67, 71]),
        ("Fmaj9", [53, 57, 60, 64, 67]),
        ("Dm9", [50, 53, 57, 60, 64]),
        ("G13", [55, 59, 62, 65, 69]),
    ],
    "jazzy": [
        ("Dm9", [50, 53, 57, 60, 64]),
        ("G13", [55, 59, 62, 65, 69]),
        ("Am9", [57, 60, 64, 67, 71]),
        ("E7b9", [52, 56, 59, 62, 65]),
    ],
    "dreamy": [
        ("Fmaj9", [53, 57, 60, 64, 67]),
        ("Gmaj9", [55, 59, 62, 66, 69]),
        ("Am9", [57, 60, 64, 67, 71]),
        ("Dmaj9", [50, 54, 57, 61, 64]),
    ],
    "melancholic": [
        ("Am7", [57, 60, 64, 67]),
        ("Cmaj7", [48, 52, 55, 59]),
        ("Dm7", [50, 53, 57, 60]),
        ("Fmaj7", [53, 57, 60, 64]),
    ],
    "nostalgic": [
        ("Am9", [57, 60, 64, 67, 71]),
        ("Fmaj9", [53, 57, 60, 64, 67]),
        ("Gmaj9", [55, 59, 62, 66, 69]),
        ("E7", [52, 56, 59, 62]),
    ],
    "dark": [
        ("Em7", [52, 55, 59, 62]),
        ("C7", [48, 52, 55, 58]),
        ("Am7", [57, 60, 64, 67]),
        ("B7", [59, 63, 66, 69]),
    ],
    "uplift": [
        ("Cmaj9", [48, 52, 55, 59, 62]),
        ("Gmaj9", [55, 59, 62, 66, 69]),
        ("Am9", [57, 60, 64, 67, 71]),
        ("Fmaj9", [53, 57, 60, 64, 67]),
    ],
    "rainy": [
        ("Em9", [52, 55, 59, 62, 66]),
        ("Bm7", [59, 62, 66, 69]),
        ("Am7", [57, 60, 64, 67]),
        ("D9", [50, 54, 57, 60, 64]),
    ],
    "sunset": [
        ("Fmaj7", [53, 57, 60, 64]),
        ("Em7", [52, 55, 59, 62]),
        ("Dm9", [50, 53, 57, 60, 64]),
        ("G7sus", [55, 60, 62, 65]),
    ],
    "hype": [
        ("Am", [57, 60, 64]),
        ("F", [53, 57, 60]),
        ("C", [48, 52, 55]),
        ("G", [55, 59, 62]),
    ],
    "spacey": [
        ("Cmaj7#11", [48, 52, 55, 59, 66]),
        ("Bbmaj7", [46, 50, 53, 57]),
        ("Am9", [57, 60, 64, 67, 71]),
        ("Gmaj7", [55, 59, 62, 66]),
    ],
    "cozy": [
        ("Am7", [57, 60, 64, 67]),
        ("Dm7", [50, 53, 57, 60]),
        ("G7", [55, 59, 62, 65]),
        ("Cmaj7", [48, 52, 55, 59]),
    ],
}

_KEY_TO_MIDI = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}

# GM programs
PROG_RHODES = 4
PROG_VIBES = 11
PROG_BASS = 33
PROG_BASS_SUB = 38  # synth bass
PROG_PAD = 89
PROG_PAD2 = 94
PROG_FLUTE = 73
PROG_ENS = 48  # string ensemble
PROG_ORGAN = 16
PROG_HARP = 46
PROG_GTR = 24  # nylon
PROG_MARIMBA = 12
PROG_SAX = 65
PROG_LEAD = 80  # square lead
PROG_WARM = 89

KICK, SNARE, HAT, OPEN_HAT, RIDE, RIM = 36, 38, 42, 46, 51, 37
CRASH, CLAVE, SHAKER, TOM_L, TOM_M, TOM_H = 49, 75, 82, 41, 45, 48
RIDE_BELL, CHINA, SPLASH = 53, 52, 55

# Alternate progressions so even the same mood never repeats the same changes
ALT_PROGRESSIONS = {
    "chill": [
        ("Em9", [52, 55, 59, 62, 66]),
        ("Cmaj9", [48, 52, 55, 59, 62]),
        ("G6", [55, 59, 62, 64, 69]),
        ("Am7", [57, 60, 64, 67]),
    ],
    "jazzy": [
        ("Fmaj7", [53, 57, 60, 64]),
        ("Bb9", [46, 50, 53, 56, 60]),
        ("Am7", [57, 60, 64, 67]),
        ("D7b9", [50, 54, 57, 60, 61]),
    ],
    "dreamy": [
        ("Abmaj9", [44, 48, 51, 55, 58]),
        ("Ebmaj9", [51, 55, 58, 62, 65]),
        ("Cm9", [48, 51, 55, 58, 62]),
        ("Gmaj7", [55, 59, 62, 66]),
    ],
    "melancholic": [
        ("Dm9", [50, 53, 57, 60, 64]),
        ("Gm7", [55, 58, 62, 65]),
        ("Bbmaj7", [46, 50, 53, 57]),
        ("A7", [57, 61, 64, 67]),
    ],
    "nostalgic": [
        ("Dmaj7", [50, 54, 57, 61]),
        ("Bm7", [59, 62, 66, 69]),
        ("Gmaj9", [55, 59, 62, 66, 69]),
        ("A7sus", [57, 62, 64, 67]),
    ],
    "dark": [
        ("Cm7", [48, 51, 55, 58]),
        ("Ab7", [44, 48, 51, 54]),
        ("Fm7", [53, 56, 60, 63]),
        ("G7alt", [55, 59, 62, 65, 68]),
    ],
    "uplift": [
        ("Dmaj9", [50, 54, 57, 61, 64]),
        ("Amaj9", [57, 61, 64, 68, 71]),
        ("Bm9", [59, 62, 66, 69, 73]),
        ("Gmaj9", [55, 59, 62, 66, 69]),
    ],
    "rainy": [
        ("Cm9", [48, 51, 55, 58, 62]),
        ("Gm7", [55, 58, 62, 65]),
        ("Ebmaj9", [51, 55, 58, 62, 65]),
        ("F9", [53, 57, 60, 63, 67]),
    ],
    "sunset": [
        ("Amaj7", [57, 61, 64, 68]),
        ("F#m7", [54, 57, 61, 64]),
        ("Dmaj9", [50, 54, 57, 61, 64]),
        ("E7sus", [52, 57, 59, 62]),
    ],
    "hype": [
        ("Em", [52, 55, 59]),
        ("C", [48, 52, 55]),
        ("G", [55, 59, 62]),
        ("D", [50, 54, 57]),
    ],
    "spacey": [
        ("Ebmaj7#11", [51, 55, 58, 62, 68]),
        ("C#maj7", [49, 53, 56, 60]),
        ("Bm9", [59, 62, 66, 69, 73]),
        ("Amaj7", [57, 61, 64, 68]),
    ],
    "cozy": [
        ("Fmaj7", [53, 57, 60, 64]),
        ("Em7", [52, 55, 59, 62]),
        ("Dm7", [50, 53, 57, 60]),
        ("G7", [55, 59, 62, 65]),
    ],
}

# Instrument pools: (keys, leads, basses, pads, sparkles)
_INSTRUMENT_POOLS = {
    "keys": [PROG_RHODES, 5, 6, 4, 3, 0, 16],  # Rhodes/Wurlitzer/grand/organ
    "lead": [PROG_RHODES, PROG_FLUTE, PROG_SAX, PROG_GTR, PROG_MARIMBA,
             PROG_VIBES, PROG_LEAD, 66, 71, 74, 65],  # sax/clarinet/flute/trumpet-ish
    "bass": [PROG_BASS, PROG_BASS_SUB, 32, 34, 35, 39],
    "pad": [PROG_PAD, PROG_PAD2, 88, 91, 92, 95],
    "sparkle": [PROG_VIBES, PROG_HARP, PROG_MARIMBA, 9, 10, 14, 46],
    "strings": [48, 49, 50, 51, 52],
}

_BASS_STYLES = ["simple_root", "long_roots", "walking", "bouncy", "pumping"]
_CHORD_RHYTHMS = ["hold", "comp", "stabs"]
_ARP_STYLES = ["none", "slow_up", "fast_up"]
_DENSITIES = ["sparse", "normal", "dense"]

# Drum style pools per mood family (allows cross-mood surprises)
_DRUM_POOLS = {
    "chill": ["boom_bap", "soft_rain", "half_time", "sparse_soft"],
    "jazzy": ["brush_swing", "boom_bap", "soft_rain"],
    "dreamy": ["sparse_soft", "ambient", "soft_rain", "half_time"],
    "melancholic": ["half_time", "minimal_dark", "sparse_soft"],
    "nostalgic": ["boom_bap", "brush_swing", "soft_rain"],
    "dark": ["minimal_dark", "half_time", "ambient"],
    "uplift": ["upbeat", "driving", "boom_bap"],
    "rainy": ["soft_rain", "sparse_soft", "brush_swing"],
    "sunset": ["boom_bap", "soft_rain", "brush_swing"],
    "hype": ["driving", "upbeat"],
    "spacey": ["ambient", "sparse_soft"],
    "cozy": ["boom_bap", "soft_rain", "brush_swing"],
}


def _fresh_seed(rng):
    """Draw a fully random 32-bit seed."""
    return int(rng.randint(0, 2**31 - 1))


def randomize_dna(mood, rng, key=None, bpm=None):
    """
    Build a unique DNA for one track. Every call produces different:
    key, tempo, progression variant, instruments, groove, density, structure.
    """
    base = dict(MOOD_DNA.get(mood, MOOD_DNA["chill"]))
    mood = mood if mood in MOOD_DNA else "chill"

    # --- tempo: jitter around mood default (unless user locked bpm) ---
    if bpm is None:
        base_bpm = base["default_bpm"]
        # ±10 BPM or wider for more separation
        base["default_bpm"] = int(np.clip(
            base_bpm + rng.randint(-10, 11), 55, 120))
    else:
        base["default_bpm"] = int(bpm)

    # --- key: always random unless explicitly locked ---
    if key is None:
        base["random_key"] = int(rng.choice(list(_KEY_TO_MIDI.values())))
    else:
        base["random_key"] = _KEY_TO_MIDI.get(key, 60)

    # --- progression: 50% alternate variant, then shuffle order ---
    if rng.random() < 0.5 and mood in ALT_PROGRESSIONS:
        prog_notes = list(ALT_PROGRESSIONS[mood])
    else:
        prog_notes = list(PROGRESSIONS.get(base["prog"], PROGRESSIONS["chill"]))
    # Rotate / lightly shuffle while keeping musical flow
    rot = int(rng.randint(0, len(prog_notes)))
    prog_notes = prog_notes[rot:] + prog_notes[:rot]
    # Occasionally swap adjacent pairs for more motion
    if rng.random() < 0.35 and len(prog_notes) >= 2:
        i = int(rng.randint(0, len(prog_notes) - 1))
        prog_notes[i], prog_notes[i + 1] = prog_notes[i + 1], prog_notes[i]
    base["_prog_notes"] = prog_notes

    # --- harmonic rhythm: 1 or 2 bars per chord ---
    base["bars_per_chord"] = int(rng.choice([1, 2, 2, 3]))

    # --- instruments: pick from pools (keeps mood character, varies timbre) ---
    base["keys_program"] = int(rng.choice(_INSTRUMENT_POOLS["keys"]))
    base["lead_program"] = int(rng.choice(_INSTRUMENT_POOLS["lead"]))
    base["bass_program"] = int(rng.choice(_INSTRUMENT_POOLS["bass"]))
    base["pad_program"] = int(rng.choice(_INSTRUMENT_POOLS["pad"]))
    base["sparkle_program"] = int(rng.choice(_INSTRUMENT_POOLS["sparkle"]))
    base["strings_program"] = int(rng.choice(_INSTRUMENT_POOLS["strings"]))

    # --- groove ---
    swing_lo, swing_hi = base["swing"]
    width = swing_hi - swing_lo
    center = rng.uniform(swing_lo, swing_hi)
    base["swing"] = (max(0.50, center - width), min(0.70, center + width))
    d_lo, d_hi = base["dilla_ms"]
    base["dilla_ms"] = (
        int(rng.randint(0, max(1, d_hi))),
        int(rng.randint(max(2, d_lo), max(3, d_hi + 10))),
    )

    # --- drums: sometimes stay in family, sometimes jump ---
    pool = _DRUM_POOLS.get(mood, ["boom_bap"])
    if rng.random() < 0.25:
        # wild card: any known style
        base["drum_style"] = str(rng.choice(list(DRUM_FNS.keys())))
    else:
        base["drum_style"] = str(rng.choice(pool))

    # --- bass / chord / arp styles ---
    if rng.random() < 0.55:
        base["bass_style"] = str(rng.choice(_BASS_STYLES))
    if rng.random() < 0.45:
        base["chord_rhythm"] = str(rng.choice(_CHORD_RHYTHMS))
    if rng.random() < 0.5:
        base["arp_style"] = str(rng.choice(_ARP_STYLES))

    # --- layers on/off ---
    base["pad_on"] = bool(rng.random() < 0.7)
    base["sparkle_on"] = bool(rng.random() < 0.65)
    base["strings_on"] = bool(rng.random() < 0.45)

    # --- melody personality ---
    base["melody_density"] = float(np.clip(
        base["melody_density"] + rng.uniform(-0.25, 0.3), 0.15, 0.95))
    base["melody_octave"] = int(rng.choice([0, 1, 1, 2]))
    # randomize note duration menu
    all_durs = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    k = int(rng.randint(3, 6))
    chosen = sorted(rng.choice(all_durs, size=k, replace=False).tolist())
    probs = rng.dirichlet(np.ones(len(chosen)))
    base["melody_durs"] = (chosen, probs.tolist())

    # --- structure fingerprint (used by arrangement envelopes) ---
    base["structure"] = {
        "intro_frac": float(rng.uniform(0.05, 0.18)),
        "break_start": float(rng.uniform(0.45, 0.65)),
        "break_len": float(rng.uniform(0.06, 0.15)),
        "outro_start": float(rng.uniform(0.82, 0.94)),
        "drop_bars": int(rng.randint(0, 4)),
        "fill_density": float(rng.uniform(0.3, 0.9)),
    }

    # unique id for this arrangement
    base["fingerprint"] = _fresh_seed(rng)
    base["mood"] = mood
    return base


# ---------------------------------------------------------------------------
# Mood DNA: each mood is a completely different musical personality
# ---------------------------------------------------------------------------
MOOD_DNA = {
    "chill": {
        "prog": "chill", "default_bpm": 75, "bars_per_chord": 2,
        "swing": (0.54, 0.60),
        "keys_program": PROG_RHODES, "lead_program": PROG_RHODES,
        "bass_program": PROG_BASS, "pad_program": PROG_PAD,
        "sparkle_program": PROG_VIBES,
        "drum_style": "boom_bap",  # classic kick 1 / snare 2,4
        "melody_density": 0.55, "melody_octave": 1,
        "melody_durs": ([0.5, 1.0, 1.5, 2.0], [0.2, 0.4, 0.25, 0.15]),
        "bass_style": "simple_root",  # root on 1, fifth on 3
        "pad_on": True, "sparkle_on": True, "strings_on": False,
        "arp_style": "none",
        "chord_rhythm": "hold",  # sustained
        "dilla_ms": (12, 18),
    },
    "jazzy": {
        "prog": "jazzy", "default_bpm": 88, "bars_per_chord": 2,
        "swing": (0.58, 0.66),
        "keys_program": PROG_RHODES, "lead_program": PROG_SAX,
        "bass_program": PROG_BASS, "pad_program": PROG_PAD,
        "sparkle_program": PROG_VIBES,
        "drum_style": "brush_swing",  # ride-focused, softer
        "melody_density": 0.7, "melody_octave": 1,
        "melody_durs": ([0.25, 0.5, 0.75, 1.0], [0.25, 0.4, 0.2, 0.15]),
        "bass_style": "walking",  # quarter-note walking
        "pad_on": False, "sparkle_on": True, "strings_on": False,
        "arp_style": "none",
        "chord_rhythm": "comp",  # short comping hits
        "dilla_ms": (5, 10),
    },
    "dreamy": {
        "prog": "dreamy", "default_bpm": 70, "bars_per_chord": 2,
        "swing": (0.52, 0.56),
        "keys_program": PROG_RHODES, "lead_program": PROG_FLUTE,
        "bass_program": PROG_BASS, "pad_program": PROG_PAD2,
        "sparkle_program": PROG_HARP,
        "drum_style": "sparse_soft",  # very few hits
        "melody_density": 0.35, "melody_octave": 2,
        "melody_durs": ([1.0, 1.5, 2.0, 3.0], [0.15, 0.35, 0.35, 0.15]),
        "bass_style": "long_roots",  # whole-note roots
        "pad_on": True, "sparkle_on": True, "strings_on": True,
        "arp_style": "slow_up",
        "chord_rhythm": "hold",
        "dilla_ms": (0, 5),
    },
    "melancholic": {
        "prog": "melancholic", "default_bpm": 68, "bars_per_chord": 2,
        "swing": (0.55, 0.58),
        "keys_program": PROG_RHODES, "lead_program": PROG_RHODES,
        "bass_program": PROG_BASS, "pad_program": PROG_PAD,
        "sparkle_program": PROG_VIBES,
        "drum_style": "half_time",  # snare only on 3
        "melody_density": 0.4, "melody_octave": 1,
        "melody_durs": ([1.0, 1.5, 2.0], [0.25, 0.5, 0.25]),
        "bass_style": "simple_root",
        "pad_on": True, "sparkle_on": False, "strings_on": True,
        "arp_style": "none",
        "chord_rhythm": "hold",
        "dilla_ms": (8, 15),
    },
    "nostalgic": {
        "prog": "nostalgic", "default_bpm": 72, "bars_per_chord": 2,
        "swing": (0.56, 0.62),
        "keys_program": PROG_RHODES, "lead_program": PROG_RHODES,
        "bass_program": PROG_BASS, "pad_program": PROG_PAD,
        "sparkle_program": PROG_VIBES,
        "drum_style": "boom_bap",
        "melody_density": 0.6, "melody_octave": 1,
        "melody_durs": ([0.5, 1.0, 1.5], [0.3, 0.45, 0.25]),
        "bass_style": "bouncy",  # root + octave hops
        "pad_on": True, "sparkle_on": True, "strings_on": False,
        "arp_style": "none",
        "chord_rhythm": "hold",
        "dilla_ms": (15, 25),
    },
    "dark": {
        "prog": "dark", "default_bpm": 70, "bars_per_chord": 2,
        "swing": (0.53, 0.57),
        "keys_program": PROG_ORGAN, "lead_program": PROG_LEAD,
        "bass_program": PROG_BASS_SUB, "pad_program": PROG_PAD2,
        "sparkle_program": PROG_VIBES,
        "drum_style": "minimal_dark",  # kick-heavy, sparse hats
        "melody_density": 0.3, "melody_octave": 0,
        "melody_durs": ([1.0, 2.0, 3.0], [0.3, 0.5, 0.2]),
        "bass_style": "long_roots",
        "pad_on": True, "sparkle_on": False, "strings_on": False,
        "arp_style": "none",
        "chord_rhythm": "hold",
        "dilla_ms": (10, 20),
    },
    "uplift": {
        "prog": "uplift", "default_bpm": 90, "bars_per_chord": 2,
        "swing": (0.52, 0.55),
        "keys_program": PROG_RHODES, "lead_program": PROG_MARIMBA,
        "bass_program": PROG_BASS, "pad_program": PROG_PAD,
        "sparkle_program": PROG_HARP,
        "drum_style": "upbeat",  # busier, open hats
        "melody_density": 0.75, "melody_octave": 2,
        "melody_durs": ([0.5, 0.75, 1.0], [0.4, 0.4, 0.2]),
        "bass_style": "bouncy",
        "pad_on": True, "sparkle_on": True, "strings_on": True,
        "arp_style": "fast_up",
        "chord_rhythm": "stabs",  # rhythmic chord stabs
        "dilla_ms": (0, 5),
    },
    "rainy": {
        "prog": "rainy", "default_bpm": 72, "bars_per_chord": 2,
        "swing": (0.55, 0.60),
        "keys_program": PROG_RHODES, "lead_program": PROG_FLUTE,
        "bass_program": PROG_BASS, "pad_program": PROG_PAD,
        "sparkle_program": PROG_VIBES,
        "drum_style": "soft_rain",  # soft brushes + shaker heavy
        "melody_density": 0.45, "melody_octave": 1,
        "melody_durs": ([1.0, 1.5, 2.0], [0.3, 0.45, 0.25]),
        "bass_style": "simple_root",
        "pad_on": True, "sparkle_on": True, "strings_on": False,
        "arp_style": "slow_up",
        "chord_rhythm": "hold",
        "dilla_ms": (10, 16),
    },
    "sunset": {
        "prog": "sunset", "default_bpm": 78, "bars_per_chord": 2,
        "swing": (0.54, 0.58),
        "keys_program": PROG_RHODES, "lead_program": PROG_GTR,
        "bass_program": PROG_BASS, "pad_program": PROG_PAD,
        "sparkle_program": PROG_VIBES,
        "drum_style": "boom_bap",
        "melody_density": 0.5, "melody_octave": 1,
        "melody_durs": ([0.5, 1.0, 1.5], [0.35, 0.45, 0.2]),
        "bass_style": "simple_root",
        "pad_on": True, "sparkle_on": True, "strings_on": True,
        "arp_style": "none",
        "chord_rhythm": "hold",
        "dilla_ms": (8, 14),
    },
    "hype": {
        "prog": "hype", "default_bpm": 95, "bars_per_chord": 1,  # faster harmony
        "swing": (0.50, 0.52),
        "keys_program": PROG_ORGAN, "lead_program": PROG_LEAD,
        "bass_program": PROG_BASS_SUB, "pad_program": PROG_PAD,
        "sparkle_program": PROG_MARIMBA,
        "drum_style": "driving",  # four-on-floor-ish energy
        "melody_density": 0.85, "melody_octave": 1,
        "melody_durs": ([0.25, 0.5, 0.75], [0.45, 0.4, 0.15]),
        "bass_style": "pumping",  # eighth notes
        "pad_on": False, "sparkle_on": True, "strings_on": False,
        "arp_style": "fast_up",
        "chord_rhythm": "stabs",
        "dilla_ms": (0, 3),
    },
    "spacey": {
        "prog": "spacey", "default_bpm": 65, "bars_per_chord": 2,
        "swing": (0.51, 0.54),
        "keys_program": PROG_PAD2, "lead_program": PROG_FLUTE,
        "bass_program": PROG_BASS_SUB, "pad_program": PROG_PAD2,
        "sparkle_program": PROG_HARP,
        "drum_style": "ambient",  # almost no drums, just texture
        "melody_density": 0.25, "melody_octave": 2,
        "melody_durs": ([2.0, 3.0, 4.0], [0.4, 0.4, 0.2]),
        "bass_style": "long_roots",
        "pad_on": True, "sparkle_on": True, "strings_on": True,
        "arp_style": "slow_up",
        "chord_rhythm": "hold",
        "dilla_ms": (0, 0),
    },
    "cozy": {
        "prog": "cozy", "default_bpm": 76, "bars_per_chord": 2,
        "swing": (0.56, 0.62),
        "keys_program": PROG_RHODES, "lead_program": PROG_GTR,
        "bass_program": PROG_BASS, "pad_program": PROG_PAD,
        "sparkle_program": PROG_VIBES,
        "drum_style": "boom_bap",
        "melody_density": 0.55, "melody_octave": 1,
        "melody_durs": ([0.5, 1.0, 1.5], [0.3, 0.5, 0.2]),
        "bass_style": "bouncy",
        "pad_on": True, "sparkle_on": True, "strings_on": False,
        "arp_style": "none",
        "chord_rhythm": "hold",
        "dilla_ms": (12, 20),
    },
}


def _transpose_notes(notes, delta):
    return [n + delta for n in notes]


def _scale_pitch_classes(root_midi):
    return set((root_midi + i) % 12 for i in [0, 2, 3, 5, 7, 8, 10])


def _nearest_in_scale(midi, pcs):
    for d in range(0, 6):
        for cand in (midi - d, midi + d):
            if cand % 12 in pcs:
                return cand
    return midi


def _chord_tone_set(chord_notes):
    root = chord_notes[0]
    pcs = set((n - root) % 12 for n in chord_notes)
    return pcs, root


def _add_drum(drums, pitch, start, vel, duration, rng, dilla_ms):
    """Add drum hit with optional Dilla-style late snare."""
    delay = rng.uniform(dilla_ms[0], dilla_ms[1]) / 1000.0 if dilla_ms[1] > 0 else 0.0
    ts = start + delay
    if ts < duration:
        drums.notes.append(pretty_midi.Note(
            velocity=int(max(20, min(127, vel))),
            pitch=pitch, start=ts, end=ts + 0.08))


def _drum_bar_boom_bap(bar_s, beat, duration, rng, dilla_ms, drums, swing):
    kicks = [0.0]
    pv = rng.randint(0, 3)
    if pv == 0:
        kicks.append(2.5)
    elif pv == 1:
        kicks.extend([1.75, 3.5])
    else:
        kicks.append(3.0)
    for kb in kicks:
        _add_drum(drums, KICK, bar_s + kb * beat, rng.uniform(95, 115), duration, rng, (0, 0))
    for sb in (1.0, 3.0):
        _add_drum(drums, SNARE, bar_s + sb * beat, rng.uniform(88, 108), duration, rng, dilla_ms)
        if rng.random() < 0.35:
            _add_drum(drums, SNARE, bar_s + (sb + 1.5) * beat, rng.uniform(30, 45), duration, rng, (0, 0))
    for i in range(8):
        pos = bar_s + (i * 0.5 + ((swing - 0.5) if i % 2 else 0)) * beat
        vel = rng.uniform(55, 80) if i % 2 == 0 else rng.uniform(40, 60)
        if i == 7 and rng.random() < 0.25:
            _add_drum(drums, OPEN_HAT, pos, vel + 5, duration, rng, (0, 0))
        else:
            _add_drum(drums, HAT, pos, vel, duration, rng, (0, 0))
    if rng.random() < 0.5:
        for i in [0.5, 1.5, 2.5, 3.5]:
            _add_drum(drums, SHAKER, bar_s + i * beat, rng.uniform(30, 45), duration, rng, (0, 0))


def _drum_bar_brush_swing(bar_s, beat, duration, rng, dilla_ms, drums, swing):
    # Ride-focused jazz swing
    _add_drum(drums, KICK, bar_s, rng.uniform(70, 85), duration, rng, (0, 0))
    if rng.random() < 0.6:
        _add_drum(drums, KICK, bar_s + 2.5 * beat, rng.uniform(55, 70), duration, rng, (0, 0))
    _add_drum(drums, SNARE, bar_s + 1 * beat, rng.uniform(55, 70), duration, rng, dilla_ms)
    _add_drum(drums, SNARE, bar_s + 3 * beat, rng.uniform(55, 70), duration, rng, dilla_ms)
    # Ride pattern: ding, ding-da, ding, ding-da
    for i in range(4):
        pos = bar_s + i * beat
        _add_drum(drums, RIDE, pos, rng.uniform(55, 75), duration, rng, (0, 0))
        if i % 2 == 1 or rng.random() < 0.7:
            _add_drum(drums, RIDE, pos + (swing - 0.5) * beat, rng.uniform(40, 55), duration, rng, (0, 0))
    # Soft hats on 2, 4
    for sb in (1.0, 3.0):
        _add_drum(drums, HAT, bar_s + sb * beat, rng.uniform(35, 50), duration, rng, (0, 0))


def _drum_bar_sparse_soft(bar_s, beat, duration, rng, dilla_ms, drums, swing):
    _add_drum(drums, KICK, bar_s, rng.uniform(70, 90), duration, rng, (0, 0))
    if rng.random() < 0.4:
        _add_drum(drums, KICK, bar_s + 2.5 * beat, rng.uniform(55, 70), duration, rng, (0, 0))
    _add_drum(drums, SNARE, bar_s + 2 * beat, rng.uniform(50, 65), duration, rng, dilla_ms)
    # sparse hats only on beats
    for i in range(4):
        if rng.random() < 0.7:
            _add_drum(drums, HAT, bar_s + i * beat, rng.uniform(35, 50), duration, rng, (0, 0))
    if rng.random() < 0.4:
        for i in [0.5, 1.5, 2.5, 3.5]:
            _add_drum(drums, SHAKER, bar_s + i * beat, rng.uniform(25, 40), duration, rng, (0, 0))


def _drum_bar_half_time(bar_s, beat, duration, rng, dilla_ms, drums, swing):
    _add_drum(drums, KICK, bar_s, rng.uniform(90, 110), duration, rng, (0, 0))
    if rng.random() < 0.5:
        _add_drum(drums, KICK, bar_s + 1.5 * beat, rng.uniform(60, 75), duration, rng, (0, 0))
    # snare only on beat 3 (half-time)
    _add_drum(drums, SNARE, bar_s + 2 * beat, rng.uniform(85, 100), duration, rng, dilla_ms)
    for i in range(8):
        pos = bar_s + (i * 0.5 + ((swing - 0.5) if i % 2 else 0)) * beat
        _add_drum(drums, HAT, pos, rng.uniform(35, 55) if i % 2 == 0 else rng.uniform(25, 40), duration, rng, (0, 0))


def _drum_bar_minimal_dark(bar_s, beat, duration, rng, dilla_ms, drums, swing):
    _add_drum(drums, KICK, bar_s, rng.uniform(100, 120), duration, rng, (0, 0))
    if rng.random() < 0.7:
        _add_drum(drums, KICK, bar_s + 1.75 * beat, rng.uniform(70, 90), duration, rng, (0, 0))
    if rng.random() < 0.6:
        _add_drum(drums, KICK, bar_s + 3.5 * beat, rng.uniform(65, 85), duration, rng, (0, 0))
    _add_drum(drums, SNARE, bar_s + 2 * beat, rng.uniform(75, 95), duration, rng, dilla_ms)
    # very sparse closed hats
    for i in [0, 2, 4, 6]:
        if rng.random() < 0.5:
            pos = bar_s + (i * 0.5 + ((swing - 0.5) if i % 2 else 0)) * beat
            _add_drum(drums, HAT, pos, rng.uniform(30, 45), duration, rng, (0, 0))
    if rng.random() < 0.3:
        _add_drum(drums, OPEN_HAT, bar_s + 3.5 * beat, rng.uniform(45, 60), duration, rng, (0, 0))


def _drum_bar_upbeat(bar_s, beat, duration, rng, dilla_ms, drums, swing):
    _add_drum(drums, KICK, bar_s, rng.uniform(95, 115), duration, rng, (0, 0))
    _add_drum(drums, KICK, bar_s + 1.5 * beat, rng.uniform(75, 95), duration, rng, (0, 0))
    _add_drum(drums, KICK, bar_s + 2.5 * beat, rng.uniform(80, 100), duration, rng, (0, 0))
    _add_drum(drums, SNARE, bar_s + 1 * beat, rng.uniform(90, 110), duration, rng, (0, 0))
    _add_drum(drums, SNARE, bar_s + 3 * beat, rng.uniform(90, 110), duration, rng, (0, 0))
    for i in range(8):
        pos = bar_s + (i * 0.5 + ((swing - 0.5) if i % 2 else 0)) * beat
        if i == 7:
            _add_drum(drums, OPEN_HAT, pos, rng.uniform(55, 75), duration, rng, (0, 0))
        else:
            _add_drum(drums, HAT, pos, rng.uniform(55, 85) if i % 2 == 0 else rng.uniform(40, 60), duration, rng, (0, 0))
    # fill every 4 bars
    if rng.random() < 0.3:
        for j, off in enumerate([3.0, 3.25, 3.5, 3.75]):
            _add_drum(drums, SNARE if j % 2 == 0 else TOM_M, bar_s + off * beat, 70 + j * 12, duration, rng, (0, 0))


def _drum_bar_soft_rain(bar_s, beat, duration, rng, dilla_ms, drums, swing):
    _add_drum(drums, KICK, bar_s, rng.uniform(75, 95), duration, rng, (0, 0))
    if rng.random() < 0.5:
        _add_drum(drums, KICK, bar_s + 2.75 * beat, rng.uniform(55, 70), duration, rng, (0, 0))
    _add_drum(drums, SNARE, bar_s + 1 * beat, rng.uniform(55, 70), duration, rng, dilla_ms)
    _add_drum(drums, SNARE, bar_s + 3 * beat, rng.uniform(55, 70), duration, rng, dilla_ms)
    # continuous soft shaker 8ths
    for i in range(8):
        pos = bar_s + (i * 0.5 + ((swing - 0.5) if i % 2 else 0)) * beat
        _add_drum(drums, SHAKER, pos, rng.uniform(30, 48) if i % 2 == 0 else rng.uniform(22, 35), duration, rng, (0, 0))
    # soft ride
    for i in range(4):
        if rng.random() < 0.6:
            _add_drum(drums, RIDE, bar_s + i * beat, rng.uniform(30, 45), duration, rng, (0, 0))


def _drum_bar_driving(bar_s, beat, duration, rng, dilla_ms, drums, swing):
    # Four-on-floor kick + backbeat snare
    for i in range(4):
        _add_drum(drums, KICK, bar_s + i * beat, rng.uniform(100, 120), duration, rng, (0, 0))
    _add_drum(drums, SNARE, bar_s + 1 * beat, rng.uniform(95, 115), duration, rng, (0, 0))
    _add_drum(drums, SNARE, bar_s + 3 * beat, rng.uniform(95, 115), duration, rng, (0, 0))
    for i in range(8):
        pos = bar_s + i * 0.5 * beat
        _add_drum(drums, HAT, pos, rng.uniform(50, 75) if i % 2 == 0 else rng.uniform(35, 55), duration, rng, (0, 0))
    if rng.random() < 0.4:
        _add_drum(drums, CRASH, bar_s, rng.uniform(70, 90), duration, rng, (0, 0))


def _drum_bar_ambient(bar_s, beat, duration, rng, dilla_ms, drums, swing):
    # Almost no drums - just soft texture hits
    if rng.random() < 0.4:
        _add_drum(drums, KICK, bar_s, rng.uniform(50, 70), duration, rng, (0, 0))
    if rng.random() < 0.25:
        _add_drum(drums, RIDE, bar_s + rng.uniform(0, 3) * beat, rng.uniform(25, 40), duration, rng, (0, 0))
    if rng.random() < 0.3:
        _add_drum(drums, SHAKER, bar_s + rng.uniform(0, 4) * beat, rng.uniform(20, 35), duration, rng, (0, 0))


DRUM_FNS = {
    "boom_bap": _drum_bar_boom_bap,
    "brush_swing": _drum_bar_brush_swing,
    "sparse_soft": _drum_bar_sparse_soft,
    "half_time": _drum_bar_half_time,
    "minimal_dark": _drum_bar_minimal_dark,
    "upbeat": _drum_bar_upbeat,
    "soft_rain": _drum_bar_soft_rain,
    "driving": _drum_bar_driving,
    "ambient": _drum_bar_ambient,
}


def _bass_pattern(style, s, e, notes, beat, bass, rng, duration):
    root_lo = notes[0] - 24
    while root_lo > 45:
        root_lo -= 12
    while root_lo < 28:
        root_lo += 12
    fifth = root_lo + 7
    span = e - s

    def add(pitch, start, end, vel):
        if start < duration:
            bass.notes.append(pretty_midi.Note(
                velocity=int(max(40, min(120, vel))),
                pitch=int(pitch), start=start, end=min(end, duration)))

    if style == "simple_root":
        add(root_lo, s, s + beat * 1.75, rng.uniform(85, 100))
        if span >= 3.5 * beat:
            add(fifth, s + 2 * beat, s + 3.5 * beat, rng.uniform(70, 85))
        if rng.random() < 0.4 and span > 3.5 * beat:
            add(root_lo + 12, s + 3.5 * beat, min(e, s + 4 * beat), rng.uniform(60, 75))
    elif style == "long_roots":
        add(root_lo, s, min(e, duration), rng.uniform(80, 95))
    elif style == "walking":
        # quarter notes: root, 3rd, 5th, approach
        steps = [root_lo, root_lo + 3 if (notes[1] - notes[0]) % 12 == 3 else root_lo + 4,
                 fifth, fifth + rng.choice([-1, 1, 2])]
        for i, p in enumerate(steps):
            st = s + i * beat
            if st < e:
                add(p, st, st + beat * 0.9, rng.uniform(75, 95))
    elif style == "bouncy":
        add(root_lo, s, s + beat * 1.5, rng.uniform(90, 105))
        if span >= 2 * beat:
            add(root_lo + 12, s + 2 * beat, s + 2.75 * beat, rng.uniform(70, 85))
        if span >= 3.5 * beat:
            add(root_lo, s + 3 * beat, s + 3.75 * beat, rng.uniform(75, 90))
    elif style == "pumping":
        # eighth notes
        n8 = int(span / (0.5 * beat))
        for i in range(n8):
            p = root_lo if i % 4 != 3 else fifth
            st = s + i * 0.5 * beat
            add(p, st, st + 0.45 * beat, rng.uniform(85, 105) if i % 2 == 0 else rng.uniform(65, 85))
    else:
        add(root_lo, s, min(e, duration), rng.uniform(85, 100))


def _chord_hits(rhodes, s, e, notes, rhythm, beat, rng, duration, chord_dur):
    vel = int(rng.uniform(55, 78))

    def add(pitch, start, end, v):
        if start < duration:
            rhodes.notes.append(pretty_midi.Note(
                velocity=int(max(35, min(110, v))),
                pitch=int(pitch), start=start, end=min(end, duration)))

    if rhythm == "hold":
        for n in notes:
            add(n, s + rng.uniform(0, 0.03), min(e - 0.05, s + chord_dur * 0.95),
                vel + rng.randint(-8, 9))
    elif rhythm == "comp":
        # jazz comping: short hits on offbeats
        positions = [0.0, 1.5, 2.75] if rng.random() < 0.7 else [0.0, 2.5]
        for pos in positions:
            st = s + pos * beat
            if st < e:
                hit_dur = beat * rng.uniform(0.4, 0.9)
                for n in notes:
                    add(n, st, st + hit_dur, vel + rng.randint(-6, 10))
    elif rhythm == "stabs":
        positions = [0.0, 1.5, 2.5, 3.5]
        for pos in positions:
            if rng.random() < 0.75:
                st = s + pos * beat
                if st < e:
                    for n in notes:
                        add(n, st, st + beat * 0.35, vel + rng.randint(-4, 12))


def compose_midi(duration, bpm=None, mood="chill", key=None, seed=None,
                 density="normal"):
    """Compose a full multi-track MIDI. Returns (PrettyMIDI, prog, dna)."""
    if seed is None:
        seed = int(np.random.RandomState().randint(0, 2**31 - 1))
    rng = np.random.RandomState(seed)

    # Heavy randomizer: unique key/tempo/prog/instruments/groove every run
    dna = randomize_dna(mood, rng, key=key, bpm=bpm)
    bpm = dna["default_bpm"]
    prog = dna.get("_prog_notes") or PROGRESSIONS.get(
        dna.get("prog", "chill"), PROGRESSIONS["chill"])

    root_pc = dna.get("random_key", 60)
    base_root = prog[0][1][0]
    first_pc = base_root % 12
    delta = (root_pc - first_pc) % 12
    if delta > 6:
        delta -= 12

    prog = [(name, _transpose_notes(notes, delta)) for name, notes in prog]
    scale_pcs = _scale_pitch_classes(prog[0][1][0])

    beat = 60.0 / bpm
    bar = 4.0 * beat
    bars_per_chord = dna["bars_per_chord"]
    chord_dur = bars_per_chord * bar
    loop_dur = len(prog) * chord_dur
    n_loops = max(1, int(np.ceil(duration / loop_dur)))

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    rhodes = pretty_midi.Instrument(program=dna["keys_program"], name="rhodes")
    melody = pretty_midi.Instrument(program=dna["lead_program"], name="melody")
    bass = pretty_midi.Instrument(program=dna["bass_program"], name="bass")
    pad = pretty_midi.Instrument(program=dna["pad_program"], name="pad")
    sparkle = pretty_midi.Instrument(program=dna["sparkle_program"], name="vibes")
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    strings = pretty_midi.Instrument(
        program=dna.get("strings_program", PROG_ENS), name="strings")

    dens_mult = {"sparse": 0.7, "normal": 1.0, "dense": 1.3}.get(density, 1.0)
    # density also randomized via DNA
    dens_mult *= rng.uniform(0.85, 1.2)
    swing_lo, swing_hi = dna["swing"]
    dilla_ms = dna["dilla_ms"]
    drum_fn = DRUM_FNS.get(dna["drum_style"], _drum_bar_boom_bap)

    mel_durs, mel_probs = dna["melody_durs"]
    mel_probs = np.asarray(mel_probs, dtype=np.float64)
    if mel_probs.sum() <= 0:
        mel_probs = np.ones(len(mel_durs)) / len(mel_durs)
    else:
        mel_probs = mel_probs / mel_probs.sum()
    mel_oct = dna["melody_octave"]
    mel_density = dna["melody_density"] * dens_mult

    # Unique motif every track
    motif_len = int(rng.randint(4, 8))
    motif_steps = rng.choice([-3, -2, -1, 0, 1, 2, 3, 4], size=motif_len)

    for loop in range(n_loops):
        loop_start = loop * loop_dur
        if loop_start >= duration:
            break

        timeline = []
        for ci, (_, notes) in enumerate(prog):
            s = loop_start + ci * chord_dur
            e = min(s + chord_dur, duration)
            if s >= duration:
                break
            timeline.append((s, e, notes))

        for s, e, notes in timeline:
            _chord_hits(rhodes, s, e, notes, dna["chord_rhythm"], beat, rng,
                        duration, chord_dur)

            if dna["pad_on"]:
                pad_notes = [notes[0] + 12, notes[2] + 12, notes[-1]]
                for n in pad_notes:
                    pad.notes.append(pretty_midi.Note(
                        velocity=int(rng.uniform(32, 48)),
                        pitch=n, start=s, end=min(e, duration)))

            if dna["strings_on"]:
                # low sustained strings: root + 5th
                for n in (notes[0], notes[min(2, len(notes)-1)]):
                    strings.notes.append(pretty_midi.Note(
                        velocity=int(rng.uniform(30, 45)),
                        pitch=n, start=s, end=min(e, duration)))

            _bass_pattern(dna["bass_style"], s, e, notes, beat, bass, rng, duration)

            # Sparkle / arpeggio
            if dna["sparkle_on"] and rng.random() < 0.5 * dens_mult:
                arp = dna["arp_style"]
                if arp == "none":
                    arp_pos = s + rng.choice([1.5, 2.5, 3.0]) * beat
                    for k, n in enumerate(notes[1:4]):
                        sparkle.notes.append(pretty_midi.Note(
                            velocity=int(rng.uniform(40, 60)),
                            pitch=n + 12, start=arp_pos + k * 0.06,
                            end=arp_pos + 0.4 + k * 0.06))
                elif arp in ("slow_up", "fast_up"):
                    step = 0.18 if arp == "slow_up" else 0.10
                    arp_notes = notes[1:] + [notes[0] + 12]
                    for k, n in enumerate(arp_notes):
                        st = s + k * step * beat
                        if st < e:
                            sparkle.notes.append(pretty_midi.Note(
                                velocity=int(rng.uniform(35, 55)),
                                pitch=n + 12, start=st, end=st + 0.5))

        # Drums
        n_bars = int(np.ceil(loop_dur / bar))
        for bi in range(n_bars):
            bar_s = loop_start + bi * bar
            if bar_s >= duration:
                break
            swing = rng.uniform(swing_lo, swing_hi)
            drum_fn(bar_s, beat, duration, rng, dilla_ms, drums, swing)
            # fills for busier styles
            if dna["drum_style"] in ("upbeat", "driving", "boom_bap") and bi % 8 == 7:
                if rng.random() < dna.get("structure", {}).get("fill_density", 0.55):
                    for j, off in enumerate([3.0, 3.25, 3.5, 3.75]):
                        _add_drum(drums, SNARE if j % 2 == 0 else RIM,
                                  bar_s + off * beat, 70 + j * 10, duration, rng, (0, 0))

        # Melody
        if rng.random() < mel_density or density == "dense":
            first_notes = timeline[0][2]
            midi = int(rng.choice(first_notes)) + 12 * (1 + mel_oct)
            while midi < 55 + 12 * mel_oct:
                midi += 12
            while midi > 84:
                midi -= 12

            n_notes = int(rng.choice([3, 4, 5, 6, 7],
                                     p=[0.2, 0.3, 0.25, 0.15, 0.1]))
            start_beats = [0.0, 1.5, 2.0, 3.0] if dna["chord_rhythm"] != "stabs" else [0.0, 1.5]
            pos = loop_start + float(rng.choice(start_beats)) * beat

            for ni in range(n_notes):
                if pos >= duration:
                    break
                chord_notes = timeline[0][2]
                for ts, te, cn in timeline:
                    if ts <= pos < te:
                        chord_notes = cn
                        break

                strong = abs((pos / beat) % 1.0) < 0.05 or abs((pos / beat) % 2.0) < 0.05
                use_motif = ni < len(motif_steps) and rng.random() < 0.6

                if strong and rng.random() < 0.75:
                    candidates = []
                    for n in chord_notes:
                        for oct_ in (-12, 0, 12):
                            cand = n + 12 * (1 + mel_oct // 2) + oct_
                            if 55 + 12 * (mel_oct // 2) <= cand <= 88:
                                candidates.append(cand)
                    if candidates:
                        midi = min(candidates, key=lambda c: abs(c - midi))
                else:
                    step = int(motif_steps[ni % len(motif_steps)]) if use_motif else int(rng.choice([-2, -1, 1, 2, 3]))
                    midi = _nearest_in_scale(midi + step, scale_pcs)
                    midi = max(55, min(88, midi))

                dur = float(rng.choice(mel_durs, p=mel_probs)) * beat
                vel = int(rng.uniform(55, 85) * (1.15 if strong else 0.9))
                melody.notes.append(pretty_midi.Note(
                    velocity=max(40, min(100, vel)),
                    pitch=midi, start=pos, end=min(pos + dur * 0.95, duration)))
                pos += dur

            pos += rng.uniform(0.5, 1.5) * beat

    # Clean notes
    all_inst = [rhodes, melody, bass, pad, sparkle, drums, strings]
    for inst in all_inst:
        inst.notes = [n for n in inst.notes if n.start < duration and n.end > 0]
        for n in inst.notes:
            if n.end > duration:
                n.end = duration

    pm.instruments.extend([drums, bass, rhodes, pad, strings, sparkle, melody])
    return pm, prog, dna
