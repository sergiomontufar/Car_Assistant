"""
Quick test runner for Piper voices (ES and EN).

Usage:
  source ~/Rowdy/bin/activate
  PYTHONPATH=. python -u demos/demo_test_piper_voices.py --play

Flags:
  --voices  Comma-separated list of voice paths. If omitted, built-in ES and EN lists are tried.
  --text    Text to synthesize (default: "Hola, esta es una prueba de voz.").
  --play    If set, play each WAV via paplay/aplay. Otherwise just write files.
  --output_dir Directory to write wavs (default: /tmp/piper_voice_tests)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from audio.tts_es_piper import PiperSpanishTTS, PiperConfig
from audio.audio_out import play_wav


DEFAULT_VOICES = [
    # Spanish (valid locally)
    "~/.local/share/piper/voices/es_MX-claude-high.onnx",
    "~/.local/share/piper/voices/es_MX-ald-medium.onnx",
    "~/.local/share/piper/voices/es_ES-mls_9972-low.onnx",
    "~/.local/share/piper/voices/es_ES-mls_10246-low.onnx",
    "~/.local/share/piper/voices/es_ES-carlfm-medium.onnx",
    # English (valid locally)
    "~/.local/share/piper/voices/en_US-ryan-high.onnx",
    "~/.local/share/piper/voices/en_US-lessac-medium.onnx",
    "~/.local/share/piper/voices/en_GB-alba-medium.onnx",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voices", type=str, default="", help="Comma-separated voice paths (.onnx).")
    parser.add_argument("--text_en", type=str, default="Hello, this is an English voice test.", help="Text for English voices.")
    parser.add_argument("--text_es", type=str, default="Hola, esta es una prueba de voz en español.", help="Text for Spanish voices.")
    parser.add_argument("--output_dir", type=str, default="/tmp/piper_voice_tests", help="Directory to write wavs.")
    parser.add_argument("--play", action="store_true", help="Play each wav after synthesis.")
    args = parser.parse_args()

    voice_paths = (
        [v.strip() for v in args.voices.split(",") if v.strip()]
        if args.voices
        else DEFAULT_VOICES
    )
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for voice in voice_paths:
        vpath = Path(voice).expanduser()
        if not vpath.exists():
            print(f"[skip] missing voice: {vpath}")
            continue
        if vpath.stat().st_size < 1000:  # skip likely broken downloads
            print(f"[skip] tiny file (likely broken): {vpath}")
            continue
        # Pick text based on voice language prefix
        text = args.text_en if vpath.name.startswith("en_") else args.text_es

        print(f"[voice] {vpath.name}")
        tts = PiperSpanishTTS(PiperConfig(voice=vpath, output_dir=output_dir, sample_rate=22050))
        wav_path = tts.speak_to_wav(text, out_path=output_dir / f"{vpath.stem}.wav")
        print(f"  -> wrote: {wav_path}")
        if args.play:
            try:
                play_wav(wav_path)
            except Exception as exc:
                print(f"  ! play failed: {exc}")


if __name__ == "__main__":
    main()
