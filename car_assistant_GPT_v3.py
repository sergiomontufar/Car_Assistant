"""
Car Assistant using ChatGPT (cloud) with built-in STT (Whisper) and TTS.

Workflow (hands-free):
- Continuously records short windows.
- Wake word: "Computer" / "Computadora" turns listening on and says "I am listening"/"Estoy escuchando".
- "Computer out"/"Computadora fuera": keep awake, stop listening.
- "Status"/"Estatus": report is_awake/is_listening.
- "Stop": set both flags false.
- When listening: answer; if user says "Respond based on the manual"/"Responde en base al manual", use manual QA; otherwise general LLM.

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

from manual_qa import ManualQA

from dotenv import load_dotenv
load_dotenv()


MANUAL_PATH = Path("~/Rowdy_chatbot/GR86 user manual.pdf").expanduser()
DEFAULT_API_KEY_FILE = Path("~/Rowdy_chatbot/OPEN_AI_KEY/key.txt").expanduser()
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"


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


def chat_answer_with_manual(qa: ManualQA, question: str, topk: int = 6) -> str:
    result = qa.ask(question, top_k_pages=topk)
    return result.get("answer", "").strip()


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


def chat_general(client: OpenAI, question: str, lang: str = "en") -> str:
    if lang.startswith("es"):
        sys_prompt = "Eres un asistente útil. Responde de forma concisa."
    else:
        sys_prompt = "You are a helpful assistant. Respond concisely."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip()


def _needs_fallback(answer: str, pages_used: list[int]) -> bool:
    al = (answer or "").lower()
    if not pages_used:
        return True
    triggers = [
        "couldn't find relevant pages",
        "could not find relevant pages",
        "cannot find relevant pages",
        "couldn't find",
        "could not find",
        "cannot find",
    ]
    return (not answer) or any(t in al for t in triggers)


def wants_manual(question: str) -> bool:
    ql = question.lower()
    return (
        "respond based on the manual" in ql
        or "responde en base al manual" in ql
        or "responde en base al manual" in ql
        or "responde usando el manual" in ql
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_seconds", type=float, default=6.0, help="Recording length per window (seconds).")
    parser.add_argument("--lang", choices=["en", "es"], default="en", help="Language for STT and prompting.")
    parser.add_argument("--api_key_file", type=str, default=None, help="Optional path to file containing OpenAI API key.")
    parser.add_argument("--env_file", type=str, default=None, help="Optional .env file path (KEY=VALUE).")
    parser.add_argument("--pdf_path", type=str, default=str(MANUAL_PATH), help="Path to GR86 manual PDF.")
    parser.add_argument("--manual_model", type=str, default="gpt-4.1-mini", help="Model for manual QA.")
    parser.add_argument("--manual_cache", type=str, default=".manual_cache", help="Cache dir for manual index.")
    parser.add_argument("--manual_topk", type=int, default=6, help="Top-K pages to retrieve from manual.")
    parser.add_argument("--manual_min_score", type=float, default=0.0, help="Min TF-IDF score to accept a page; lower = more recall.")
    args = parser.parse_args()

    api_key = get_api_key(args.env_file or args.api_key_file)
    client = OpenAI(api_key=api_key)
    # Manual QA (retrieval + citing)
    qa = ManualQA(
        pdf_path=args.pdf_path,
        cache_dir=args.manual_cache,
        model=args.manual_model,
    )

    is_awake = True
    is_listening = True
    print("Car Assistant (ChatGPT) hands-free. Say 'Computer' / 'Computadora' to wake. Ctrl+C to quit.")
    try:
        while True:
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

            lower = transcript.lower()
            print(f"User: {transcript}")

            # Wake word
            if "computer" in lower or "computadora" in lower:
                is_awake = True
                is_listening = True
                ack = "I am listening" if args.lang.startswith("en") else "Estoy escuchando"
                tts_and_play(client, ack)
                continue



            # Computer out -> stop listening but stay awake
            if "silence" in lower or "silencio" in lower:
                is_awake = False
                is_listening = False
                ack = "OK, I will stop listening" if args.lang.startswith("en") else "OK, ejo de escuchar"
                tts_and_play(client, ack)
                continue

            # Status command
            if "status" in lower or "estatus" in lower:
                status_msg = f"Awake: {is_awake}, Listening: {is_listening}"
                print(f"-> State: {status_msg}")
                tts_and_play(client, status_msg if args.lang.startswith("en") else f"Estado: despierto {is_awake}, escuchando {is_listening}")
                continue

            # If not awake, ignore until wake word
            if not is_awake:
                continue

            # If not listening, ignore until wake word
            if not is_listening:
                continue

            # Route: manual trigger phrases
            if wants_manual(transcript):
                result = qa.ask(transcript, top_k_pages=args.manual_topk, min_score=args.manual_min_score)
                pages_used = result.get("pages_used", [])
                answer = (result.get("answer", "") or "").strip()
                if _needs_fallback(answer, pages_used):
                    answer = chat_general(client, transcript, lang=args.lang)
            else:
                answer = chat_general(client, transcript, lang=args.lang)

            print(f"Bot: {answer}")
            tts_and_play(client, answer)

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    import shutil

    main()
