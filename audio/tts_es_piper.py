"""
Spanish TTS via Piper CLI.

We prefer the Piper binary for stability on Jetson. Provide a thin wrapper that
invokes Piper with a configured voice and writes a WAV, plus convenience to play
it via AudioPlayer utilities.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PiperConfig:
    voice: Path
    output_dir: Path = Path("/tmp")
    piper_bin: str = "piper"
    sample_rate: int = 22050  # Piper voice SR; used by caller


class PiperSpanishTTS:
    def __init__(self, cfg: PiperConfig) -> None:
        self.cfg = cfg
        self.voice = Path(cfg.voice).expanduser()
        if not self.voice.exists():
            raise FileNotFoundError(f"Piper voice not found: {self.voice}")

    @property
    def sample_rate(self) -> int:
        return int(self.cfg.sample_rate)

    def speak_to_wav(self, text: str, out_path: Optional[Path | str] = None) -> Path:
        out_path = Path(out_path) if out_path else Path(self.cfg.output_dir) / "tts_es.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        cmd = [
            self.cfg.piper_bin,
            "--model",
            str(self.voice),
            "--output_file",
            str(out_path),
        ]

        proc = subprocess.run(cmd, input=text.encode("utf-8"), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"Piper failed (code {proc.returncode}): {proc.stderr.decode(errors='ignore')}")
        return out_path
