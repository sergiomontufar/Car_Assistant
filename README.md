# SocialRobot v2 (Bilingual, no UI)

English + Spanish voice assistant loop for Jetson:
VAD -> Faster-Whisper (multilingual) -> Ollama -> TTS (EN Tacotron2/WaveGlow, ES XTTS-v2) -> Playback

## Install
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
python3 -m pip install --upgrade pip
pip3 install -r requirements_v2_bilingual.txt
```

## Run
```bash
python3 main_v2_bilingual.py
```

Notes:
- Tacotron2/WaveGlow and XTTS-v2 may download model weights on first run (cached after).
- Change Whisper model in `main_v2_bilingual.py` from `tiny` to `base` for better accuracy if you have headroom.
# AI_Asistant
