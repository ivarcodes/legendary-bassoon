"""Drums - Authentic lofi boom bap with MPC swing, ghost notes, humanization."""

import numpy as np
from scipy.signal import butter, sosfilt, lfilter
from .core import SR, seconds_to_samples, pan_stereo, soft_clip, bitcrush, tape_saturate

_KICK_SOS = butter(2, 100 / (SR / 2), btype="high", output="sos")
_SNARE_SOS = butter(2, np.array([150, 6000]) / (SR / 2), btype="band", output="sos")
_HH_SOS = butter(2, np.array([2000, 12000]) / (SR / 2), btype="band", output="sos")
_OH_SOS = butter(2, np.array([1500, 10000]) / (SR / 2), btype="band", output="sos")
_DRUM_BUS_LP = butter(2, 4000 / (SR / 2), btype="low", output="sos")


def _synth_kick(rng, pitch_env=True):
    """Warm, muted lofi kick - round thud not punchy boom."""
    dur = rng.uniform(0.3, 0.5)
    n = seconds_to_samples(dur)
    t = np.arange(n, dtype=np.float64) / SR

    if pitch_env:
        freq_start = rng.uniform(70, 110)
        freq_end = rng.uniform(35, 55)
        freq = freq_start * np.exp(-t * 25) + freq_end * (1 - np.exp(-t * 25))
        phase = 2.0 * np.pi * np.cumsum(freq) / SR
    else:
        freq = rng.uniform(45, 60)
        phase = 2.0 * np.pi * freq * t

    body = np.sin(phase) * np.exp(-t * rng.uniform(15, 22))
    click_dur = int(0.003 * SR)
    click = rng.randn(min(click_dur, n)) * np.exp(-np.arange(min(click_dur, n)) * 400)
    body[:len(click)] += click * 0.2

    body = sosfilt(_KICK_SOS, body)
    body = bitcrush(body, bits=10, downsample=1)
    body = tape_saturate(body, drive=0.3)
    body = soft_clip(body * 1.3) / 1.3
    return body * rng.uniform(0.85, 1.0)


def _synth_snare(rng, rim=False):
    """Lo-fi snare or rimshot - soft attack, dark tail."""
    if rim:
        dur = rng.uniform(0.06, 0.1)
        n = seconds_to_samples(dur)
        t = np.arange(n, dtype=np.float64) / SR
        tonal = np.sin(2.0 * np.pi * rng.uniform(800, 1200) * t) * np.exp(-t * 80) * 0.3
        noise = rng.randn(n) * np.exp(-t * 60) * 0.4
        body = tonal + noise
        body = bitcrush(body, bits=8, downsample=1)
        body = tape_saturate(body, drive=0.5)
        return body * rng.uniform(0.3, 0.5)

    dur = rng.uniform(0.15, 0.25)
    n = seconds_to_samples(dur)
    t = np.arange(n, dtype=np.float64) / SR

    tonal_freq = rng.uniform(140, 200)
    tonal = np.sin(2.0 * np.pi * tonal_freq * t) * np.exp(-t * rng.uniform(20, 35)) * 0.5

    noise = rng.randn(n) * np.exp(-t * rng.uniform(10, 18))
    noise = sosfilt(_SNARE_SOS, noise) * 0.5

    body = tonal + noise
    body = bitcrush(body, bits=10, downsample=1)
    body = tape_saturate(body, drive=0.35)
    body = soft_clip(body * 1.15) / 1.15
    return body * rng.uniform(0.7, 0.9)


def _synth_hihat(rng, closed=True):
    """Dusty hi-hats - brittle, filtered, velocity-ready."""
    if closed:
        dur = rng.uniform(0.03, 0.06)
    else:
        dur = rng.uniform(0.12, 0.25)
    n = seconds_to_samples(dur)
    t = np.arange(n, dtype=np.float64) / SR

    decay_rate = rng.uniform(35, 60) if closed else rng.uniform(12, 22)
    noise = rng.randn(n) * np.exp(-t * decay_rate)
    noise = sosfilt(_HH_SOS if closed else _OH_SOS, noise)

    if closed:
        noise = bitcrush(noise, bits=8, downsample=1)
        noise = tape_saturate(noise, drive=0.6)
    else:
        noise = bitcrush(noise, bits=10, downsample=1)
        noise = tape_saturate(noise, drive=0.3)

    noise = soft_clip(noise * 1.2) / 1.2
    return noise


def _synth_shaker(rng):
    """Soft shaker/percussion for texture."""
    dur = rng.uniform(0.08, 0.15)
    n = seconds_to_samples(dur)
    t = np.arange(n, dtype=np.float64) / SR
    noise = rng.randn(n) * np.exp(-t * rng.uniform(40, 80))
    noise = sosfilt(butter(2, np.array([3000, 10000]) / (SR / 2), btype="band", output="sos"), noise)
    noise = bitcrush(noise, bits=8, downsample=1)
    noise = soft_clip(noise * 1.5) / 1.5
    return noise * rng.uniform(0.2, 0.4)


def _generate_kit(rng):
    return {
        'kick': _synth_kick(rng),
        'snare': _synth_snare(rng, rim=False),
        'rim': _synth_snare(rng, rim=True),
        'hh_closed': _synth_hihat(rng, closed=True),
        'hh_open': _synth_hihat(rng, closed=False),
        'shaker': _synth_shaker(rng),
    }


def generate_bar(bpm, seed=None, swing=0.62, hat_pan=0.18):
    """Generate 4-bar lofi boom bap pattern with authentic swing and humanization."""
    rng = np.random.RandomState(seed)
    kit = _generate_kit(rng)

    beat_dur = 60.0 / bpm
    bar_len = seconds_to_samples(4.0 * beat_dur)
    total = bar_len * 4

    kick = kit['kick']
    snare = kit['snare']
    rim = kit['rim']
    hh_c = kit['hh_closed']
    hh_o = kit['hh_open']
    shaker = kit['shaker']

    hh_cl, hh_cr = pan_stereo(hh_c, hat_pan)
    hh_ol, hh_or = pan_stereo(hh_o, hat_pan * 0.8)
    sh_l, sh_r = pan_stereo(shaker, hat_pan * 1.2)

    swing_pct = swing
    swing_delay = (beat_dur / 4.0) * (swing_pct - 0.5) * 2.0

    out_l = np.zeros(total, dtype=np.float64)
    out_r = np.zeros(total, dtype=np.float64)

    for bar in range(4):
        base = bar * bar_len

        # KICK: beats 1 and 3 (boom bap), ghost kick on beat 3 at lower velocity
        kb = [0.0, 2.0]  # beat 1 and 3 (0-indexed: 0 and 2)
        if rng.random() < 0.3:
            kb.append(0.5)  # ghost kick on "and" of 1
        if rng.random() < 0.2:
            kb.append(2.5)  # ghost kick on "and" of 3

        kp = (np.array(kb) * beat_dur * SR + base).astype(int)
        kp = np.clip(kp + (rng.uniform(-0.005, 0.005, len(kp)) * SR).astype(int), 0, total - len(kick))
        for idx, p in enumerate(kp):
            v = rng.uniform(0.7, 0.95) if idx < 2 else rng.uniform(0.35, 0.55)
            out_l[p:p+len(kick)] += kick * v
            out_r[p:p+len(kick)] += kick * v

        # SNARE: beats 2 and 4, SLIGHTLY LATE (J Dilla feel) - 2-5ms behind
        snare_late_ms = rng.uniform(2, 5) / 1000.0
        sp = ((np.array([1.0, 3.0]) * beat_dur + snare_late_ms + swing_delay + base / SR) * SR).astype(int)
        sp = np.clip(sp + (rng.uniform(-0.003, 0.003, len(sp)) * SR).astype(int), 0, total - len(snare))
        for p, v in zip(sp, rng.uniform(0.6, 0.85, len(sp))):
            out_l[p:p+len(snare)] += snare * v
            out_r[p:p+len(snare)] += snare * v

        # RIM SHOT / GHOST SNARE: before beat 3 (J Dilla feel)
        if rng.random() < 0.7:
            rim_pos = int((2.75 * beat_dur + swing_delay + base / SR) * SR)
            rim_pos = np.clip(rim_pos + int(rng.uniform(-0.002, 0.002) * SR), 0, total - len(rim))
            out_l[rim_pos:rim_pos+len(rim)] += rim * rng.uniform(0.25, 0.4)
            out_r[rim_pos:rim_pos+len(rim)] += rim * rng.uniform(0.25, 0.4)

        # HI-HATS: 16th notes with swing, velocity variation 60-90%
        # Closed hats on all 16ths, open hats on offbeats
        ht = np.arange(16) * (beat_dur * 0.25)
        ht[1::2] += swing_delay  # swing on offbeat 16ths
        hp = ((ht + base / SR) * SR).astype(int)
        hp = np.clip(hp + (rng.uniform(-0.002, 0.002, len(hp)) * SR).astype(int), 0, total - len(hh_cl))

        for i, p in enumerate(hp):
            # Velocity variation: never constant, accent downbeats slightly
            base_vel = rng.uniform(0.55, 0.85)
            if i % 4 == 0:  # downbeats
                base_vel *= rng.uniform(1.0, 1.15)
            elif i % 2 == 1:  # offbeats
                base_vel *= rng.uniform(0.7, 0.9)

            if i % 8 == 7 and rng.random() < 0.3:  # open hat on last 16th of 2 bars
                end_ol = min(p + len(hh_ol), total)
                end_or = min(p + len(hh_or), total)
                out_l[p:end_ol] += hh_ol[:end_ol-p] * base_vel * 0.6
                out_r[p:end_or] += hh_or[:end_or-p] * base_vel * 0.6
            else:
                end_cl = min(p + len(hh_cl), total)
                end_cr = min(p + len(hh_cr), total)
                out_l[p:end_cl] += hh_cl[:end_cl-p] * base_vel
                out_r[p:end_cr] += hh_cr[:end_cr-p] * base_vel

        # SHAKER: 8th notes, subtle, on offbeats
        if rng.random() < 0.6:
            sh_t = (np.arange(8) * (beat_dur * 0.5) + swing_delay * 0.5 + base / SR) * SR
            sh_t = sh_t.astype(int)
            sh_t = np.clip(sh_t + (rng.uniform(-0.003, 0.003, len(sh_t)) * SR).astype(int), 0, total - len(shaker))
            for p in sh_t:
                end_sh = min(p + len(sh_l), total)
                out_l[p:end_sh] += sh_l[:end_sh-p] * rng.uniform(0.15, 0.3)
                out_r[p:end_sh] += sh_r[:end_sh-p] * rng.uniform(0.15, 0.3)

        # OCCASIONAL FILL: triplet hats at end of 4 bars
        if bar == 3 and rng.random() < 0.4:
            fill_pos = int(3.75 * beat_dur * SR + base)
            fill_pos = np.clip(fill_pos, 0, total - len(hh_cl))
            for j in range(3):
                fp = fill_pos + int(j * beat_dur * 0.25 * SR * 0.66)  # triplet feel
                if fp < total - len(hh_cl):
                    out_l[fp:fp+len(hh_cl)] += hh_cl * rng.uniform(0.25, 0.4)
                    out_r[fp:fp+len(hh_cr)] += hh_cr * rng.uniform(0.25, 0.4)

    # Drum bus processing: low-pass for warmth, slight saturation
    out_l = sosfilt(_DRUM_BUS_LP, out_l)
    out_r = sosfilt(_DRUM_BUS_LP, out_r)
    out_l = tape_saturate(out_l, drive=0.2)
    out_r = tape_saturate(out_r, drive=0.2)

    return out_l, out_r


def generate_loop_variants(bpm, num_variants=16, seed=None):
    """Generate variants with different swing, tonal character, and volume."""
    base = seed if seed is not None else np.random.randint(0, 2**31)
    variants = []
    for i in range(num_variants):
        rng = np.random.RandomState(base + i * 1000)
        # Swing 55-70% - THE critical lofi parameter
        swing = rng.uniform(0.55, 0.70)
        hat_pan = rng.uniform(0.15, 0.22)
        vl, vr = generate_bar(bpm, seed=base + i * 1000, swing=swing, hat_pan=hat_pan)

        # Per-variant dark filter: 3000-6000Hz for muffled lofi vibe
        cutoff = rng.uniform(3000, 6000)
        sos = butter(2, cutoff / (SR / 2), btype="low", output="sos")
        vl = sosfilt(sos, vl)
        vr = sosfilt(sos, vr)

        # Per-variant volume: -3dB to +3dB for section dynamics
        vol = 10 ** (rng.uniform(-3.0, 3.0) / 20.0)
        vl *= vol
        vr *= vol

        variants.append((vl, vr))
    return variants