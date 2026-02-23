"""
Smoke test: synthesize one English and one Spanish sentence and play them.

Usage:
  source ~/Rowdy/bin/activate
  python -u demos/demo_tts_bilingual.py --piper_voice ~/.local/share/piper/voices/es_ES-mls_10246-low.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from audio.audio_out import play_wav
from audio.tts_en_tacotron2_hifigan import Tacotron2HiFiGanEnglishTTS, Tacotron2HiFiGanConfig
from audio.tts_es_piper import PiperSpanishTTS, PiperConfig
from audio.tts_router import TTSRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--piper_voice", type=Path, required=True, help="Path to Piper Spanish voice .onnx file")
    parser.add_argument("--device", type=str, default="cuda", help="Device for English Tacotron2 (cuda|cpu)")
    args = parser.parse_args()

    en_tts = Tacotron2HiFiGanEnglishTTS(Tacotron2HiFiGanConfig(device=args.device))
    es_tts = PiperSpanishTTS(PiperConfig(voice=args.piper_voice))
    router = TTSRouter(en_tts=en_tts, es_tts=es_tts)

    # English
    wav_en, sr_en = router.speak("Hello, good day, test from Tacotron2 plus HiFi-GAN on Jetson.", lang="en")
    print(f"English SR={sr_en}, file={wav_en}")
    play_wav(wav_en)


 
    # Spanish
    wav_es, sr_es = router.speak("Hola, buen dia, prueba de piper en el yetson.", lang="es")
    print(f"Spanish SR={sr_es}, file={wav_es}")
    play_wav(wav_es)


if __name__ == "__main__":
    main()
