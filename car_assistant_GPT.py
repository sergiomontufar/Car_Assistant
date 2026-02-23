"""
Car Assistant using ChatGPT (cloud) with built-in STT (Whisper) and TTS.

Workflow (push-to-talk):
1) Press Enter -> record for N seconds (configurable).
2) Transcribe via OpenAI Whisper.
3) Send to ChatGPT with GR86 manual context (excerpt).
4) TTS the reply with OpenAI speech, play via paplay/aplay.

Requirements:
  pip install openai PyPDF2 pyaudio
Env:
  export OPENAI_API_KEY=sk-...
Usage:
  source ~/Rowdy/bin/activate
  # put your key in env OR a file (see below)
  PYTHONPATH=. python -u car_assistant_GPT.py --record_seconds 10

API key:
  - Preferred: set env OPENAI_API_KEY
  - Or place the key (single line) in ~/.openai_api_key and it will be read automatically
  - Or pass --api_key_file /path/to/keyfile
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import wave
import subprocess
from pathlib import Path
from typing import Optional

import pyaudio
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


MANUAL_PATH = Path("~/Rowdy_chatbot/GR86 user manual.pdf").expanduser()
# Cap manual to stay within model context; adjust if needed
MANUAL_MAX_CHARS = 80_000
DEFAULT_API_KEY_FILE = Path("~/Rowdy_chatbot/OPEN_AI_KEY/key.txt").expanduser()
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"


def load_manual_excerpt(path: Path = MANUAL_PATH, max_chars: int | None = MANUAL_MAX_CHARS) -> str:
    """Load full text from the GR86 manual (best-effort)."""
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
            if max_chars is not None and total + len(text) > max_chars:
                text = text[: max_chars - total]
            parts.append(text)
            total += len(text)
            if max_chars is not None and total >= max_chars:
                break
        excerpt = "\n".join(parts)
        if excerpt:
            print(f"Loaded manual text: {len(excerpt)} chars from {path}")
            if max_chars is not None:
                print(f"(Truncated to {max_chars} chars to fit model context.)")
        else:
            print(
                f"Warning: manual text is empty after parsing {path} "
                "(PDF may be scanned/without text)."
            )
        return excerpt
    except Exception as exc:
        print(f"Warning: could not load manual excerpt: {exc}")
        return ""


def load_env_file(env_path: Path) -> None:
    """Populate os.environ from a simple .env file (KEY=VALUE per line)."""
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
    except Exception as exc:
        print(f"Warning: could not load env file {env_path}: {exc}")


def get_api_key(arg_path: Optional[str]) -> str:
    # Load default .env if present (only adds vars not already set)
    load_env_file(DEFAULT_ENV_FILE)
    # Load user-specified env file if provided
    if arg_path:
        load_env_file(Path(arg_path).expanduser())
    # 1) Env
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    # 2) File (arg) or default
    candidates = []
    if arg_path:
        candidates.append(Path(arg_path).expanduser())
    candidates.append(DEFAULT_API_KEY_FILE)
    for p in candidates:
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8").strip()
                if text:
                    return text
            except Exception:
                continue
    print("Error: OPENAI_API_KEY not set and no key file found. "
          "Set env OPENAI_API_KEY or create ~/.openai_api_key with the key.")
    sys.exit(1)


def record_audio_wav(duration_sec: float, sample_rate: int = 16000) -> Path:
    """Record mono audio for duration_sec and write to a temp WAV file."""
    pa = pyaudio.PyAudio()
    frames_per_buffer = 1024
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )
    frames = []
    print(f"Recording for {duration_sec} seconds...")
    start = time.time()
    while time.time() - start < duration_sec:
        data = stream.read(frames_per_buffer, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    pa.terminate()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    return Path(tmp.name)


def stt_transcribe(client: OpenAI, wav_path: Path, language: str = "en") -> str:
    with wav_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
        )
    return (resp.text or "").strip()


def chat_answer(client: OpenAI, question: str, manual_excerpt: str, lang: str = "en") -> str:
    if lang.startswith("es"):
        sys_prompt = (
            "Eres un asistente de automóvil. Usa el manual GR86 para responder. "
            "Si no está en el manual, dilo. Responde de forma concisa."
        )
    else:
        sys_prompt = (
            "You are a car assistant. Use the GR86 user manual to answer. "
            "If it's not in the manual, say so. Respond concisely."
        )
    if manual_excerpt:
        sys_prompt += "\n\nGR86 manual excerpt:\n" + manual_excerpt

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip()


def tts_and_play(client: OpenAI, text: str, voice: str = "alloy") -> None:
    speech_file = Path(tempfile.mktemp(suffix=".wav"))
    resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        response_format="wav",
    )
    # New SDK returns a binary response; write to file
    with speech_file.open("wb") as f:
        f.write(resp.read())

    # Play via paplay/aplay
    player = "paplay" if shutil.which("paplay") else "aplay"
    subprocess.run([player, str(speech_file)], check=False)
    try:
        speech_file.unlink()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_seconds", type=float, default=8.0, help="Recording length per turn (seconds).")
    parser.add_argument("--lang", choices=["en", "es"], default="en", help="Language for STT and prompting.")
    parser.add_argument("--api_key_file", type=str, default=None, help="Optional path to file containing OpenAI API key.")
    parser.add_argument("--env_file", type=str, default=None, help="Optional .env file path (KEY=VALUE).")
    args = parser.parse_args()

    api_key = get_api_key(args.env_file or args.api_key_file)
    client = OpenAI(api_key=api_key)
    manual_excerpt = load_manual_excerpt()

    print("Car Assistant (ChatGPT). Press Enter to record; Ctrl+C to quit.")
    try:
        while True:
            input("Press Enter to record...")
            wav_path = record_audio_wav(args.record_seconds)
            try:
                transcript = stt_transcribe(client, wav_path, language=args.lang)
            finally:
                try:
                    wav_path.unlink()
                except Exception:
                    pass

            if not transcript:
                print("No speech detected.")
                continue

            print(f"User: {transcript}")
            answer = chat_answer(client, transcript, manual_excerpt, lang=args.lang)
            print(f"Bot: {answer}")
            tts_and_play(client, answer)

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    import shutil

    main()
