Build a 100% offline lofi hip-hop music generator in Python (numpy + scipy 
only, no internet, no samples, no external audio libraries required). Goal: 
authentic-sounding lofi, clean enough for focus/study listening — not 
generic AI-slop synth music, and definitely not glitchy or noisy.

=================================================================
CORE SOUND DESIGN
=================================================================

1. RHODES / CHORD ENGINE (engine/chords.py)
   - Don't use plain sine waves — synthesize an electric-piano (Rhodes-style)
     tone via additive synthesis: fundamental sine + a 2nd harmonic sine that
     decays FASTER than the fundamental (gives the classic "bell" attack)
   - Fast attack (~5-8ms), no sustain plateau, slow exponential decay (~2-4s)
   - Slow tremolo: amplitude LFO at 4-5Hz, depth ~15-25%
   - Detune a second copy of each note by 3-6 cents and mix in for width/chorus
   - Real jazz 7th-chord voicings, not triads:
       dreamy:      Am7 - Fmaj7 - Am7 - Fmaj7
       jazzy:       Dm7 - G7 - Cmaj7 - Am7
       melancholic: Am7 - Fmaj7 - Dm7 - E7
       nostalgic:   Am7 - Ebmaj7 - Fmaj7 - G7
       chill:       Cmaj7 - Am7 - Fmaj7 - Dm7
   - ONE 4-bar loop repeated for the whole track. No new progressions later —
     repetition is the genre, not a limitation.

2. DRUMS (engine/drums.py) — only 3 elements, nothing else
   - Kick: sine with fast pitch envelope ~150Hz -> ~50Hz over 30-40ms,
     amplitude decay ~250-350ms, plus a short highpassed noise click at the
     transient for "thump"
   - Snare: bandpassed noise (200Hz-4kHz) layered with a ~180Hz tonal sine
     body, decay ~120-180ms
   - Hi-hat: highpassed white noise (>7kHz), 40-80ms decay closed, occasional
     longer open hat
   - Swing: delay every off-beat hit by 8-15% of a 16th note (non-negotiable
     for the groove)
   - Randomize velocity ±10-20% per hit, plus small timing jitter (±3-8ms) —
     perfect quantization is the #1 tell of fake-sounding generated drums

3. BASS (engine/bass.py)
   - Root note only, 1-2 octaves below the chord root
   - Sine/triangle wave with light tanh() saturation for warmth
   - Short pluck envelope, not sustained

4. MELODY (engine/melody.py) — optional, sparse
   - ~25-30% chance of a short phrase per 4-bar loop, otherwise silent
   - Pentatonic/blues scale over the current chord
   - Soft sine or lightly detuned two-oscillator patch, gentle attack

5. TEXTURE (engine/texture.py)
   - Vinyl crackle as discrete pops (Poisson-process impulses through a
     bandpass filter) layered under continuous filtered hiss
   - Sits at -18 to -24dB relative to the music — should read as warmth,
     never as audible "noise." Default off/minimal unless explicitly requested.

=================================================================
EFFECTS CHAIN (engine/effects.py)
=================================================================
- Lowpass filter ~8-10kHz (Butterworth via scipy.signal.sosfilt) — the
  signature lofi "warm/dusty" effect
- Light Schroeder reverb: 4 parallel comb filters + 2 series allpass filters.
  Subtle, not a hall.
- Mild tanh() saturation on the master bus for tape warmth
- Optional subtle wow/flutter: slow pitch wobble ~0.5-1Hz via tiny variable
  delay-line resampling, capped at ±0.3% pitch variation max

=================================================================
SONG STRUCTURE (engine/arranger.py)
=================================================================
One 4-bar loop for the whole song. Structure = layers dropping in/out, never
new musical material:
  chords only (intro)
  -> + drums
  -> + bass (main groove, bulk of track length)
  -> - drums (breakdown: chords + bass only)
  -> + drums again
  -> fade out over final 8-16 bars
Crossfade every layer transition over 1-2 bars — no hard on/off jumps.
No key changes, no bridge, no tempo changes.

=================================================================
CHAT PARSER (chat/parser.py)
=================================================================
Rule-based keyword extraction, no LLM needed:
- mood keywords -> one of the 5 progressions above
- tempo words: chill/slow/relaxed -> 60-70bpm, upbeat/energetic -> 80-90bpm
  (overall lofi range 60-90bpm, sweet spot ~70-75)
- duration via regex ("X minutes"/"X hours"), default 2 minutes
- optional toggles: vinyl crackle on/off, melody on/off

=================================================================
QUALITY CONTROL — NO NOISE, NO ERRORS, NOTHING STARTLING (non-negotiable)
=================================================================
This is focus/study music. Nothing should ever click, pop, clip, or jar the
listener out of flow.

1. Gain staging: each layer gets its own gain BEFORE mixing so the sum never
   exceeds ~0.8 full scale even when every layer/note is active at once.
2. Soft limiter (tanh-based, never hard clip) on the master after mixing.
3. Final target: peak -3dB to -6dB, RMS -18dB to -14dB (leave real dynamic
   range — lofi is not loudness-war mastered).
4. Every oscillator/hit must fade in/out over 2-5ms at start and end — never
   start or end a sound at a non-zero sample (that's what causes clicks).
5. Crossfade the 4-bar loop boundary a few ms so repeats don't click.
6. Apply filters/reverb to the full continuous buffer, not per-note, to avoid
   seams at buffer edges.
7. Track-level fade-in/fade-out (1-3s) at the very start/end.
8. Automated self-check BEFORE saving the WAV — must verify:
   - max(abs(samples)) <= 1.0 (zero clipped samples)
   - no NaN/Inf anywhere in the buffer
   - RMS in a sane range (not silent, not maxed)
   - duration matches requested duration within 0.1s
   If any check fails: regenerate or raise a clear error. Never write a
   broken/clipped file to disk.

=================================================================
OUTPUT / STRUCTURE
=================================================================
- Render via scipy.io.wavfile, 16-bit PCM, 44100Hz
- Package layout:
    lofi-chatbot/
    ├── app.py                # CLI entry point
    ├── engine/
    │   ├── core.py           # shared audio utils (envelopes, note->freq, etc.)
    │   ├── drums.py
    │   ├── chords.py
    │   ├── bass.py
    │   ├── melody.py
    │   ├── texture.py
    │   ├── effects.py
    │   ├── arranger.py
    │   └── render.py         # mixing, self-check, WAV export
    ├── chat/
    │   └── parser.py
    └── generated/             # output WAVs
- After building, render a 30-60 second test track and print the self-check
  results (peak, RMS, clip count, duration) to confirm the output is clean
  before considering the task done.