"""Parse user text into structured music parameters."""

import re

MOOD_KEYWORDS = {
    "chill": "chill",
    "relax": "chill",
    "calm": "chill",
    "mellow": "chill",
    "cozy": "chill",
    "dreamy": "dreamy",
    "float": "dreamy",
    "ethereal": "dreamy",
    "space": "dreamy",
    "jazzy": "jazzy",
    "jazz": "jazzy",
    "swing": "jazzy",
    "blue": "melancholic",
    "sad": "melancholic",
    "melanchol": "melancholic",
    "dark": "melancholic",
    "rain": "melancholic",
    "lonely": "melancholic",
    "nostalgic": "nostalgic",
    "vintage": "nostalgic",
    "old": "nostalgic",
    "warm": "nostalgic",
    "happy": "chill",
    "upbeat": "chill",
    "sunny": "chill",
}

INSTRUMENT_KEYWORDS = {
    "piano": "piano",
    "keys": "piano",
    "rhodes": "piano",
    "electric piano": "piano",
    "guitar": "guitar",
    "acoustic guitar": "guitar",
    "sax": "sax",
    "saxophone": "sax",
    "flute": "flute",
    "synth": "synth",
    "synthpad": "synth",
    "pad": "synth",
    "strings": "strings",
    "violin": "strings",
}

EFFECT_KEYWORDS = {
    "rain": "rain",
    "rainy": "rain",
    "raining": "rain",
    "storm": "rain",
    "thunder": "rain",
    "cafe": "cafe",
    "coffee": "cafe",
    "coffee shop": "cafe",
    "ambient": "ambient",
    "wind": "wind",
    "windy": "wind",
    "vinyl": "vinyl",
    "crackle": "vinyl",
    "tape": "hiss",
    "hiss": "hiss",
    "snow": "wind",
}

TEMPO_KEYWORDS = {
    "slow": 72,
    "very slow": 65,
    "medium": 80,
    "fast": 90,
    "upbeat": 88,
    "laid back": 75,
    "laid-back": 75,
    "downtempo": 70,
}

SCALE_KEYWORDS = {
    "dark": "minor_pentatonic",
    "sad": "minor_pentatonic",
    "blues": "minor_blues",
    "blue": "minor_blues",
    "bright": "major_pentatonic",
    "happy": "major_pentatonic",
    "jazzy": "dorian",
    "jazz": "dorian",
    "chill": "mixolydian",
}

# binaural preset keywords -> preset name (checked longest-phrase first)
BINAURAL_KEYWORDS = [
    ("deep work", "focus"),
    ("binaural beat", "focus"),
    ("binaural", "focus"),
    ("brainwave", "focus"),
    ("concentrat", "focus"),
    ("productive", "focus"),
    ("focus", "focus"),
    ("work", "focus"),
    ("study", "study"),
    ("exam", "study"),
    ("relaxed alert", "study"),
    ("sharp attention", "peak"),
    ("gamma", "peak"),
    ("peak", "peak"),
    ("light focus", "clear"),
    ("calm focus", "flow"),
    ("meditat", "flow"),
    ("flow", "flow"),
    ("deep calm", "calm"),
    ("anxious", "calm"),
    ("wind down", "sleep"),
    ("bedtime", "sleep"),
    ("sleep", "sleep"),
    ("nap", "sleep"),
]


def parse_user_input(text: str) -> dict:
    text_lower = text.lower().strip()

    mood = "chill"
    for kw, m in MOOD_KEYWORDS.items():
        if kw in text_lower:
            mood = m
            break

    instruments = ["piano"]
    for kw, inst in INSTRUMENT_KEYWORDS.items():
        if kw in text_lower:
            if inst not in instruments:
                instruments.append(inst)

    textures = ["vinyl"]
    for kw, tex in EFFECT_KEYWORDS.items():
        if kw in text_lower:
            if tex not in textures:
                textures.append(tex)

    tempo = 80
    for kw, t in TEMPO_KEYWORDS.items():
        if kw in text_lower:
            tempo = t
            break

    scale = "minor_pentatonic"
    for kw, s in SCALE_KEYWORDS.items():
        if kw in text_lower:
            scale = s
            break

    binaural = "off"
    for kw, preset in BINAURAL_KEYWORDS:
        if kw in text_lower:
            binaural = preset
            break

    key = "C4"
    key_match = re.search(r'\b([A-G][#b]?)\s*(major|minor|maj|min|m)?\b', text_lower)
    if key_match:
        key_name = key_match.group(1).upper()
        if len(key_name) == 2 and key_name[1] in "#b":
            key_name = key_name
        else:
            key_name = key_name.replace("#", "#")
        key = f"{key_name}4"

    return {
        "mood": mood,
        "tempo": tempo,
        "key": key,
        "instruments": instruments,
        "textures": textures,
        "scale": scale,
        "binaural": binaural,
        "original_text": text,
    }


def format_chat_response(params: dict) -> str:
    mood = params["mood"]
    tempo = params["tempo"]
    instruments = ", ".join(params["instruments"])
    textures = ", ".join(params["textures"])
    binaural = params.get("binaural", "off")
    bb = "" if binaural == "off" else f"\nBinaural: {binaural} (headphones on)"
    return (
        f"Generating a {mood} lofi track at {tempo} BPM.\n"
        f"Instruments: {instruments}\n"
        f"Textures: {textures}\n"
        f"Key: {params['key']} | Scale: {params['scale']}{bb}"
    )
