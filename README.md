Car Assistant (hands-free voice)
================================

Hands-free in-car assistant for GR86 support. It listens continuously, transcribes speech with Whisper, answers with GPT, and replies with TTS audio. It supports:
- General conversation mode.
- Manual-grounded mode with page citations from the GR86 PDF manual.

Main capabilities
-----------------
- Continuous mic stream with WebRTC VAD utterance detection (no fixed recording window).
- Voice commands: wake, pause listening, status, and stop.
- Manual QA pipeline with two retrieval backends:
  - `tfidf` (default, `manual_qa.py`)
  - `vectors` (OpenAI embeddings, `manual_qa_vectors.py`)
- Spoken responses with mic mute/unmute during playback to reduce echo loops.

Prerequisites
-------------
- Python 3.8+ (Jetson-friendly).
- Working microphone and speaker output.
- System dependency for PyAudio:
  - `sudo apt-get install portaudio19-dev`
- OpenAI API key with access to:
  - STT (`whisper-1`)
  - TTS
  - Chat Completions
  - Embeddings (only when using `--manual_retriever vectors`)

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
- Export in shell: `export OPENAI_API_KEY=sk-...`
- Or create `.env` in this folder with `OPENAI_API_KEY=sk-...`
- Or store key file at `~/Rowdy_chatbot/OPEN_AI_KEY/key.txt`
- Or pass `--api_key_file /path/to/keyfile`

5) Ensure the GR86 manual PDF path is correct:
- Default: `GR86 user manual.pdf` in project root.
- Or pass `--pdf_path /full/path/to/GR86_user_manual.pdf`.

How to run (`car_assistant_GPT_v5.py`)
--------------------------------------
Default (TF-IDF retriever):
```
PYTHONPATH=. python3 -u car_assistant_GPT_v5.py
```

Vector retriever:
```
PYTHONPATH=. python3 -u car_assistant_GPT_v5.py --manual_retriever vectors
```

Vector retriever with explicit embedding model:
```
PYTHONPATH=. python3 -u car_assistant_GPT_v5.py --manual_retriever vectors --manual_embedding_model text-embedding-3-small
```

Useful options
--------------
- `--lang {en,es}`: STT/prompt language.
- `--pdf_path`: manual PDF location.
- `--manual_model`: answer model for manual QA.
- `--manual_retriever {tfidf,vectors}`: retrieval backend.
- `--manual_embedding_model`: embedding model (vectors backend only).
- `--manual_cache`: cache directory for manual index files.
- `--manual_topk`: number of retrieved pages.
- `--manual_min_score`: minimum retrieval score cutoff.
- `--vad_aggr`, `--silence_ms`, `--frame_ms`: voice activity tuning.

Voice command flow
------------------
- Say `Computer` for general mode.
- Say `Instruction` or `Instructions` for manual mode.
- In manual mode, follow-up questions stay in manual mode until you say commands such as `Computer`, `Stop`, or `Silence`.
- `Status` reports current state flags in terminal and voice.
- `Ctrl+C` to exit.

Notes
-----
- First manual query may take longer (index build + cache write).
- If playback fails, ensure `paplay` or `aplay` exists.
- For debugging manual routing, the app prints:
  - selected retriever backend
  - `-> Using manual QA: ...`
  - `-> Manual pages_used: [...]`
