"""
Car Assistant using ChatGPT (cloud) with built-in STT (Whisper) and TTS.

Workflow (hands-free, continuous):
- Continuously reads mic frames (no fixed window).
- VAD detects speech; accumulate audio until ~1.3s of silence after speech, then transcribe.
- Wake word: "Computer" / "Computadora" turns listening on and says "I am listening"/"Estoy escuchando".
- "Silence"/"Silencio": keep awake, stop listening.
- "Status"/"Estatus": report is_awake/is_listening.
- "Stop"/"Detente": set both flags false.
- When listening: if text contains "Respond based on the manual"/"Responde en base al manual", use manual QA; otherwise general LLM.

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
from collections import deque
import subprocess
from pathlib import Path
from typing import Optional

import pyaudio
import webrtcvad
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


def write_wav(frames: list[bytes], sample_rate: int = 16000) -> Path:
    """Write buffered PCM frames to a temp WAV file."""
    pa = pyaudio.PyAudio()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    pa.terminate()
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


def tts_and_play(client: OpenAI, text: str, voice: str = "alloy") -> Optional[subprocess.Popen]:
    speech_file = Path(tempfile.mktemp(suffix=".wav"))
    resp = client.audio.speech.create(
        model="gpt-5.2-tts",
        #model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        response_format="wav",
    )
    # New SDK returns a binary response; write to file
    with speech_file.open("wb") as f:
        f.write(resp.read())

    # Play via paplay/aplay
    player = "paplay" if shutil.which("paplay") else "aplay"
    proc = subprocess.Popen([player, str(speech_file)])
    # we won't delete immediately to avoid race; caller can clean if needed
    return proc, speech_file


def chat_general(client: OpenAI, question: str, lang: str = "en") -> str:
    if lang.startswith("es"):
        sys_prompt = "Eres un asistente útil. Responde de forma concisa."
    else:
        sys_prompt = "You are a helpful assistant. Respond concisely."
    resp = client.chat.completions.create(
        model="gpt-5.2"
        #model="gpt-4o-mini",
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
    parser.add_argument("--lang", choices=["en", "es"], default="en", help="Language for STT and prompting.")
    parser.add_argument("--api_key_file", type=str, default=None, help="Optional path to file containing OpenAI API key.")
    parser.add_argument("--env_file", type=str, default=None, help="Optional .env file path (KEY=VALUE).")
    parser.add_argument("--pdf_path", type=str, default=str(MANUAL_PATH), help="Path to GR86 manual PDF.")
    #parser.add_argument("--manual_model", type=str, default="gpt-4.1-mini", help="Model for manual QA.")
    parser.add_argument("--general_model", type=str, default="gpt-5.2", help="Model for general QA.")
    parser.add_argument("--manual_cache", type=str, default=".manual_cache", help="Cache dir for manual index.")
    parser.add_argument("--manual_topk", type=int, default=6, help="Top-K pages to retrieve from manual.")
    parser.add_argument("--manual_min_score", type=float, default=0.0, help="Min TF-IDF score to accept a page; lower = more recall.")
    parser.add_argument("--vad_aggr", type=int, default=2, help="WebRTC VAD aggressiveness (0-3).")
    parser.add_argument("--silence_ms", type=int, default=1300, help="Silence threshold (ms) to end utterance after speech.")
    parser.add_argument("--frame_ms", type=int, default=20, help="Frame size in ms (10, 20, or 30).")
    args = parser.parse_args()

    api_key = get_api_key(args.env_file or args.api_key_file)
    client = OpenAI(api_key=api_key)
    # Manual QA (retrieval + citing)
    qa = ManualQA(
        pdf_path=args.pdf_path,
        cache_dir=args.manual_cache,
        model=args.manual_model,
    )

    # Audio + VAD setup
    sample_rate = 16000
    frame_ms = args.frame_ms
    frame_size = int(sample_rate * frame_ms / 1000)
    frame_bytes = frame_size * 2
    vad = webrtcvad.Vad(args.vad_aggr)
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=frame_size,
    )

    is_awake = False
    is_listening = False
    manual_preference = False
    speech_active = False
    last_speech_ts = 0.0
    silence_thresh = args.silence_ms / 1000.0
    buffered_frames: list[bytes] = []
    min_utt_secs = 0.3
    min_frames_for_utt = max(1, int(min_utt_secs / (frame_ms / 1000.0)))
    post_tts_mute_ms = 0  # no mute window
    post_tts_mute_until = 0.0
    current_player: Optional[subprocess.Popen] = None
    current_wav: Optional[Path] = None
    temp_wavs: deque[Path] = deque(maxlen=3)
    last_bot_text: str = ""
    last_tts_ts: float = 0.0
    mic_enabled = True
    reenable_after_playback = False

    def speak(text: str, remember_bot: bool = False) -> None:
        """Centralized TTS playback with mic-disable to avoid echo."""
        nonlocal current_player, current_wav, mic_enabled, reenable_after_playback, last_bot_text, last_tts_ts
        if current_player:
            current_player.terminate()
        if current_wav:
            try:
                current_wav.unlink()
            except Exception:
                pass
            current_wav = None
        try:
            stream.stop_stream()
        except Exception:
            pass
        mic_enabled = False
        reenable_after_playback = True
        current_player, current_wav = tts_and_play(client, text)
        if remember_bot:
            last_bot_text = text
            last_tts_ts = time.time()

    print("Car Assistant (ChatGPT) hands-free. Say 'Computer' / 'Computadora' to wake. Ctrl+C to quit.")
    try:
        while True:
            # Cleanup finished players and temp files
            if current_player is not None and current_player.poll() is not None:
                current_player = None
                if current_wav:
                    try:
                        current_wav.unlink()
                    except Exception:
                        pass
                    current_wav = None
                while temp_wavs:
                    try:
                        temp_wavs.popleft().unlink()
                    except Exception:
                        pass
                if reenable_after_playback:
                    try:
                        stream.start_stream()
                    except Exception:
                        pass
                    mic_enabled = True
                    reenable_after_playback = False

            # If mic is disabled (during TTS), skip reads
            if not mic_enabled:
                time.sleep(0.01)
                continue
            frame = stream.read(frame_size, exception_on_overflow=False)

            if len(frame) != frame_bytes:
                continue

            is_speech = vad.is_speech(frame, sample_rate)
            now = time.time()

            if is_speech:
                buffered_frames.append(frame)
                speech_active = True
                last_speech_ts = now
            else:
                if not speech_active:
                    continue
                if (now - last_speech_ts) >= silence_thresh:
                    # finalize utterance
                    if buffered_frames:
                        if len(buffered_frames) < min_frames_for_utt:
                            buffered_frames = []
                            speech_active = False
                            continue
                        wav_path = write_wav(buffered_frames, sample_rate=sample_rate)
                        buffered_frames = []
                        speech_active = False
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
                        # Cooldown after TTS: ignore any transcript that arrives too soon after last bot speak
                        if last_tts_ts > 0 and (time.time() - last_tts_ts) < 2.0:
                            continue

                        lower = transcript.lower()
                        print(f"User: {transcript}")

                        # If this transcript matches the last bot reply, ignore to avoid echo loops
                        if last_bot_text:
                            t = " ".join(transcript.strip().lower().split())
                            b = " ".join(last_bot_text.strip().lower().split())
                            if t == b or t.startswith(b) or b.startswith(t):
                                continue

                        # If playback is ongoing: allow only stop playback commands; ignore others to avoid echo loops
                        if current_player is not None:
                            if "stop audio" in lower or "callate" in lower or "silencio ahora" in lower:
                                current_player.terminate()
                                current_player = None
                            # Regardless, do not process further while playback active
                            continue

                        # Wake word
                        if "computer" in lower or "computadora" in lower:
                            is_awake = True
                            is_listening = True
                            manual_preference = False
                            ack = "I am listening" if args.lang.startswith("en") else "Estoy escuchando"
                            speak(ack, remember_bot=False)
                            continue

                        # Stop listening but stay awake
                        if "silence" in lower or "silencio" in lower:
                            is_listening = False
                            manual_preference = False
                            if current_player:
                                current_player.terminate()
                            continue

                        # Stop all
                        if "stop" in lower or "detente" in lower:
                            is_awake = False
                            is_listening = False
                            manual_preference = False
                            if current_player:
                                current_player.terminate()
                            continue

                        # Status command
                        if "status" in lower or "estatus" in lower:
                            status_msg = f"Awake: {is_awake}, Listening: {is_listening}, ManualPref: {manual_preference}"
                            print(f"-> State: {status_msg}")
                            speak(
                                status_msg
                                if args.lang.startswith("en")
                                else f"Estado: despierto {is_awake}, escuchando {is_listening}",
                                remember_bot=False,
                            )
                            continue

                        # Stop playback immediately command
                        if "stop audio" in lower or "callate" in lower or "silencio ahora" in lower:
                            if current_player:
                                current_player.terminate()
                                current_player = None
                            if current_wav:
                                try:
                                    current_wav.unlink()
                                except Exception:
                                    pass
                                current_wav = None
                            continue

                        if not is_awake or not is_listening:
                            continue
                        
                        
                        # if the user wants to respond based on the manual, do so

                        if wants_manual(transcript):
                            manual_preference = True

                        if manual_preference:
                            result = qa.ask(transcript, top_k_pages=args.manual_topk, min_score=args.manual_min_score)
                            pages_used = result.get("pages_used", [])
                            answer = (result.get("answer", "") or "").strip()
                            if _needs_fallback(answer, pages_used):
                                answer = chat_general(client, transcript, lang=args.lang)
                            manual_preference = False
                        else:
                            answer = chat_general(client, transcript, lang=args.lang)

                        print(f"Bot: {answer}")
                        speak(answer, remember_bot=True)
                        # Reset buffers/state to avoid echo loops
                        buffered_frames = []
                        speech_active = False
                        last_speech_ts = 0.0

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    import shutil

    main()
