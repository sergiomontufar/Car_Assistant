"""

Car Asistant Project

Language: English
Assistant Name: Car Assistant
Wake Word: Computer
Stop Word: Stop

Sergio Montufar


Bilingual (English+Spanish) v2 loop (no UI):
VAD -> STT (Whisper multilingual) -> Ollama -> TTS (EN: Tacotron2/WaveGlow, ES: XTTS-v2) -> Playback

Features:
- Runs fully locally on Jetson
- Barge-in: if user speaks while audio is playing, stop playback immediately
- Basic echo suppression window after playback starts
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import os
from pathlib import Path

MANUAL_PATH = Path("~/Rowdy_chatbot/GR86 user manual.pdf").expanduser()
MANUAL_MAX_CHARS = 20000


def load_manual_excerpt(path: Path = MANUAL_PATH, max_chars: int = MANUAL_MAX_CHARS) -> str:
    """Load a text excerpt from the GR86 manual (best-effort)."""
    try:
        import PyPDF2  # type: ignore

        if not path.exists():
            print(f"Warning: manual not found at {path}")
            return ""
        reader = PyPDF2.PdfReader(str(path))
        parts = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            if not text:
                continue
            if total + len(text) > max_chars:
                text = text[: max_chars - total]
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
        excerpt = "\n".join(parts)
        if excerpt:
            print(f"Loaded manual excerpt: {len(excerpt)} chars from {path}")
        else:
            print(f"Warning: manual excerpt is empty after parsing {path} (PDF may be scanned/without text).")
        return excerpt
    except Exception as exc:
        print(f"Warning: could not load manual excerpt: {exc}")
        return ""

from audio.stt import FasterWhisperSTT
from audio.vad import VADConfig, VADListener
from audio.audio_out import play_wav
from audio.tts_en_tacotron2_hifigan import Tacotron2HiFiGanEnglishTTS, Tacotron2HiFiGanConfig
from audio.tts_es_piper import PiperSpanishTTS, PiperConfig
from audio.tts_router import TTSRouter
from llm.ollama import OllamaClient
from llm.prompts import system_prompt_for_language


def _detect_whisper_device() -> str:
    try:
        import ctranslate2  # type: ignore
        if ctranslate2.get_cuda_device_count() > 0:  # type: ignore[attr-defined]
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        choices=["en", "es"],
        default="en",
        help="Force end-to-end language (en or es). Defaults to en.",
    )
    parser.add_argument(
        "--stt_device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for Faster-Whisper (auto: detect CUDA, else CPU).",
    )
    parser.add_argument(
        "--en_tts_device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for English Tacotron2+HiFi-GAN (ignored if --en_piper_voice is set).",
    )
    parser.add_argument(
        "--whisper_model_path",
        type=str,
        default=None,
        help="Optional faster-whisper model path or name (e.g., tiny.en, tiny).",
    )
    parser.add_argument(
        "--hf_offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 to avoid network calls (models must be cached).",
    )
    parser.add_argument(
        "--en_piper_voice",
        type=str,
        default=None,
        help="Optional Piper voice path for English (use to force a specific voice, e.g., male). If omitted, uses Tacotron2+HiFi-GAN.",
    )
    parser.add_argument(
        "--es_piper_voice",
        type=str,
        default="~/.local/share/piper/voices/es_MX-ald-medium.onnx",
        help="Piper voice path for Spanish.",
    )
    args = parser.parse_args()
    forced_lang = args.lang
    manual_excerpt = load_manual_excerpt()
    manual_note = (
        "Use the following GR86 user manual context when answering. "
        "If a question is unrelated to the manual, say you cannot answer without the manual context. "
        "GR86 manual excerpt:\n"
        f"{manual_excerpt[:MANUAL_MAX_CHARS]}"
    )
    # Optional: force HF offline if desired (models must be pre-cached locally)
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # VAD
    vad_config = VADConfig(sample_rate=16000, frame_duration_ms=30, padding_duration_ms=360, aggressiveness=2)

    # STT: force chosen language
    stt_device = _detect_whisper_device() if args.stt_device == "auto" else args.stt_device
    if args.whisper_model_path:
        whisper_model = Path(args.whisper_model_path).expanduser()
        whisper_model_arg = str(whisper_model) if whisper_model.exists() else args.whisper_model_path
    else:
        whisper_model_arg = "tiny.en" if forced_lang == "en" else "tiny"
    stt_model = FasterWhisperSTT(
        model_size_or_path=whisper_model_arg,
        device=stt_device,
        compute_type="int8",
        language=forced_lang,
    )

    # LLM (Ollama local)
    ollama_client = OllamaClient(
        url="http://localhost:11434/api/chat",
        model="gemma3:270m",
        stream=True,
        system_prompt=system_prompt_for_language("en") + "\n\n" + manual_note + "\n\nAnswer concisely.",
    )

    # TTS engines
    # English: default Tacotron2+HiFi-GAN (device selectable). If --en_piper_voice is provided and exists, use it instead.
    if args.en_piper_voice and Path(args.en_piper_voice).expanduser().exists():
        tts_en = PiperSpanishTTS(PiperConfig(voice=Path(args.en_piper_voice).expanduser()))
    else:
        tts_en = Tacotron2HiFiGanEnglishTTS(Tacotron2HiFiGanConfig(device=args.en_tts_device))

    # Spanish Piper voice
    es_voice = Path(args.es_piper_voice).expanduser()
    tts_es = PiperSpanishTTS(PiperConfig(voice=es_voice))

    tts_router = TTSRouter(en_tts=tts_en, es_tts=tts_es)

    vad_listener: Optional[VADListener] = None
    last_bot_response: str = ""
    last_play_started_ts: float = 0.0
    is_awake: bool = False
    is_listening: bool = False
    pending_text: list[str] = []

    def on_speech_detected(raw_bytes: bytes) -> None:
        nonlocal last_bot_response, last_play_started_ts, is_awake, is_listening, pending_text

        # Barge-in: stop audio immediately if it's playing
        # (Playback handled per utterance; no persistent player needed)

        # STT (forced language)
        try:
            recognized_text, lang = stt_model.run_stt_with_lang(
                raw_bytes, sample_rate=vad_listener.sample_rate if vad_listener else 16000
            )
        except Exception as exc:
            print("STT error:", exc)
            return

        recognized_text = (recognized_text or "").strip()
        lang = forced_lang  # force handling
        print(f"-> User ({lang}): {recognized_text}")

        if not recognized_text:
            return

        lower_text = recognized_text.lower()
        # State machine:
        # - "computer" when not awake => awake=True, listening=True, clear buffer
        # - while listening: accumulate text until "respond"
        # - "respond": stop listening, send accumulated text
        # - "stop": reset everything

        if not is_awake:
            if lower_text.startswith("computer"):
                is_awake = True
                is_listening = True
                status_msg = f"Awake: {is_awake}, Listening: {is_listening}"
                print(f"-> State: {status_msg}")
                pending_text = []
            return

        # Awake:
        if lower_text.startswith("stop"):
            is_awake = False
            is_listening = False
            pending_text = []
            return

        if lower_text.startswith("state"):
            status_msg = f"Awake: {is_awake}, Listening: {is_listening}"
            print(f"-> State: {status_msg}")
            wav, _sr = tts_router.speak(status_msg, lang)
            last_play_started_ts = time.time()
            play_wav(wav)
            
            return

        if not is_listening:
            # Not listening, wait for "respond" to do nothing; require listening True to process
            return

        if lower_text.startswith("respond"):
            # finalize and send ONLY what was previously buffered
            full_text = " ".join(pending_text).strip()
            pending_text = []
            is_listening = True
            if not full_text:
                return
            recognized_text = full_text #after “respond”, the buffered pending_text becomes recognized_text, and that string is sent to ollama_client.query(...).
        else:
            # accumulate and keep listening
            pending_text.append(recognized_text)
            return

        # Basic echo suppression: ignore transcripts that are basically the last response
        # Also ignore very-short transcripts right after playback starts (speaker bleed).
        now = time.time()
        if now - last_play_started_ts < 0.6:
            if len(recognized_text) < 8:
                return

        if last_bot_response and recognized_text.lower() in last_bot_response.lower():
            return

        # Switch system prompt language and enforce concise replies, always include manual context
        base_prompt = system_prompt_for_language("es" if lang == "es" else "en")
        ollama_client.system_prompt = base_prompt + "\n\n" + manual_note + "\n\nAnswer concisely."

        # LLM
        bot_text = ollama_client.query(recognized_text).strip()
        if not bot_text:
            return

        print("-> Bot:", bot_text)
        last_bot_response = bot_text

        # Choose TTS based on forced language
        wav, _sr = tts_router.speak(bot_text, lang)
        last_play_started_ts = time.time()
        # Router returns a Path; hand off to paplay/aplay
        play_wav(wav)

    vad_listener = VADListener(config=vad_config, device_index=None, on_speech_callback=on_speech_detected)

    print("-> Starting bilingual v2 (no UI). Ctrl+C to stop.")
    try:
        vad_listener.start()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        vad_listener.stop()


if __name__ == "__main__":
    main()
