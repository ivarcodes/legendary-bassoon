from .core import SR, seconds_to_samples, midi_to_freq, soft_clip, bitcrush, tape_saturate
from .drums import generate_loop_variants
from .chords import generate_loop as generate_chords, PROGRESSIONS
from .bass import generate_bass_for_chords
from .melody import generate_melody
from .texture import generate_vinyl, generate_ambience
from .effects import apply_effects_stereo
from .binaural import apply_binaural, PRESETS as BINAURAL_PRESETS
from .arranger import arrange_stereo
from .generate import generate, generate_to_wav
