"""Lofi Music Generator - 100% offline, high-quality lofi generation."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from engine.generate import generate_to_wav

import gradio as gr

GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")


def generate_lofi(bpm: int = None, mood: str = "chill", key: str = None, duration_min: float = 3.0, ambience: str = "rain", binaural: str = "focus", binaural_db: float = -20.0) -> tuple:
    duration = duration_min * 60
    # bpm None -> mood default tempo from DNA
    bpm_out = bpm if bpm else "auto"

    print(f"[Lofi] BPM={bpm_out} Style={mood} Key={key or 'random'} Duration={duration:.0f}s Ambience={ambience} Binaural={binaural}")

    t0 = time.time()

    ts = int(time.time())
    path = os.path.join(GENERATED_DIR, f"lofi_{ts}.wav")

    def report(progress, message):
        print(f"[Lofi] {progress * 100:.0f}% {message}")

    generate_to_wav(
        path,
        duration=duration,
        bpm=bpm,
        mood=mood,
        key=key,
        ambience_style=ambience,
        progress_callback=report,
        seed=None,  # always unique
        binaural=binaural,
        binaural_db=float(binaural_db),
    )

    print(f"[Lofi] Done in {time.time()-t0:.1f}s")
    size_mb = os.path.getsize(path) / (1024 * 1024)

    bb_note = ""
    if binaural and binaural != "off":
        bb_note = (
            f"\nBinaural: **{binaural}** @ {binaural_db:.0f} dB "
            f"(use stereo headphones; ~15–60 min sessions; not while driving)"
        )

    response = (
        f"**{mood.title()} lofi** at {bpm_out} BPM\n"
        f"Key: {key or 'random'}\n"
        f"Duration: {duration_min:.1f} min\n"
        f"Ambience: {ambience}"
        f"{bb_note}\n"
        f"Generated in {time.time()-t0:.1f}s ({size_mb:.1f} MB)"
    )
    return response, path


def build_ui():
    with gr.Blocks(title="Lofi Generator") as demo:
        gr.Markdown("# Lofi Music Generator")
        gr.Markdown("Set your vibe. Generate a track. 100% offline, no samples needed.")

        with gr.Row():
            with gr.Column(scale=2):
                mood = gr.Dropdown(
                    ["chill", "jazzy", "dreamy", "melancholic", "nostalgic",
                     "dark", "uplift", "rainy", "sunset", "hype", "spacey", "cozy"],
                    value="chill",
                    label="Mood"
                )
                bpm = gr.Slider(
                    50, 120, value=0, step=1,
                    label="BPM (0 = mood default)"
                )
                key = gr.Dropdown(
                    ["random", "C", "D", "E", "F", "G", "A", "B"],
                    value="random",
                    label="Key"
                )
                ambience = gr.Dropdown(
                    ["rain", "room", "none"],
                    value="rain",
                    label="Ambience"
                )
                duration = gr.Slider(
                    0.5,
                    60.0,
                    value=3.0,
                    step=0.5,
                    label="Duration (minutes)"
                )
                gr.Markdown(
                    "**Binaural** (stereo headphones required) — quiet "
                    "brainwave layer: *focus* 15 Hz beta for deep work, "
                    "*study* 10 Hz alpha for relaxed alertness, *peak* 40 Hz "
                    "gamma for sharp attention. Sit 15–60 min at moderate "
                    "volume; skip if epilepsy-prone or driving."
                )
                binaural = gr.Dropdown(
                    ["off", "focus", "study", "clear", "flow", "peak",
                     "calm", "sleep"],
                    value="focus",
                    label="Binaural beat"
                )
                binaural_db = gr.Slider(
                    -30.0, -12.0, value=-20.0, step=1.0,
                    label="Binaural level (dB, quieter = subtler)"
                )
                gen_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=3):
                audio_out = gr.Audio(label="Lofi Track", type="filepath")
                status = gr.Markdown()

        def gen(bpm_val, mood_val, key_val, amb_val, dur_val, binaural_val, binaural_db_val):
            key_param = None if key_val == "random" else key_val
            amb_param = None if amb_val == "none" else amb_val
            bpm_param = int(bpm_val) if bpm_val and bpm_val >= 50 else None
            response_text, audio_path = generate_lofi(
                bpm=bpm_param, mood=mood_val, key=key_param,
                duration_min=dur_val, ambience=amb_param,
                binaural=binaural_val, binaural_db=binaural_db_val,
            )
            return audio_path, response_text

        gen_btn.click(
            gen,
            [bpm, mood, key, ambience, duration, binaural, binaural_db],
            [audio_out, status]
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)