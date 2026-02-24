Car Assistant (hands-free voice)
================================

Hands-free in-car assistant that listens for a wake word, transcribes speech with Whisper, answers with GPT (general Q&A or GR86 manual-grounded), and speaks replies back with OpenAI TTS. Works in English or Spanish and keeps listening until you tell it to stop.

Main capabilities
-----------------
- Always-on mic loop with WebRTC VAD to detect utterances without a fixed window.
- Wake words: "Computer" / "Computadora"; quick commands: "Silence"/"Silencio", "Stop"/"Detente", "Status"/"Estatus".
- Two answer modes: general GPT chat, or manual-grounded answers when you say "Respond based on the manual"/"Responde en base al manual".
- Manual mode builds a TF-IDF index of the GR86 PDF manual and cites pages; falls back to general chat if nothing relevant is found.
- TTS playback with automatic mic mute/unmute to avoid echo loops.

Prerequisites
-------------
- Python 3.10+ recommended.
- Working microphone.
- System deps:
  - PortAudio headers for `pyaudio` (e.g., `sudo apt-get install portaudio19-dev`).
  - `webrtcvad` wheels are available on most Linux distros; if building from source, ensure a C compiler is present.
- OpenAI API key with access to Whisper and GPT models.
- GR86 manual PDF available locally (default path is `~/Rowdy_chatbot/GR86 user manual.pdf`).

Setup
-----
1) Clone and enter the project:
```
git clone <repo-url>
cd Car_Assistant
```

2) Create and activate a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```

3) Install Python dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```

4) Provide your OpenAI API key (pick one):
- Export: `export OPENAI_API_KEY=sk-...`
- Or create `.env` in this folder with `OPENAI_API_KEY=sk-...`
- Or store the key file at `~/Rowdy_chatbot/OPEN_AI_KEY/key.txt` (default fallback)
- Or pass `--api_key_file /path/to/keyfile`

Optional: if your GR86 manual is elsewhere, set `--pdf_path /custom/path.pdf` or update `MANUAL_PATH` in `car_assistant_GPT_v4.py`.

How to run
----------
In the activated venv:
```
python -u car_assistant_GPT_v4.py --lang en
```
- `--lang {en,es}` selects STT/LLM language.
- `--pdf_path` points to the GR86 manual PDF.
- `--manual_cache` controls where the TF-IDF index is stored (default `.manual_cache`).
- `--manual_topk` and `--manual_min_score` tune manual retrieval.
- `Ctrl+C` to exit.

Usage flow
----------
- Wait for: `Car Assistant (ChatGPT) hands-free. Say 'Computer' / 'Computadora' to wake.`
- Say the wake word, then ask your question.
- Say "Silence"/"Stop" to pause or stop listening; "Status" reports awake/listening flags.
- To force manual-grounded answers, include "Respond based on the manual" (or the Spanish variant) in your request.

Notes and tips
--------------
- First manual-grounded question triggers PDF indexing; this may take a moment and writes to `.manual_cache`.
- If audio fails to play, ensure `paplay` (PulseAudio) or `aplay` (ALSA) is installed.
- Avoid running from inside a container without audio devices unless you provide virtual audio input/output.
- For troubleshooting VAD sensitivity, adjust `--vad_aggr` (0–3) and `--silence_ms`.
