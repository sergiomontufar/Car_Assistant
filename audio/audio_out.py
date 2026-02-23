"""
Simple WAV playback helpers for Jetson using PulseAudio.
- Prefer paplay; fall back to aplay if unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def play_wav(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if shutil.which("paplay"):
        cmd = ["paplay", str(path)]
    elif shutil.which("aplay"):
        cmd = ["aplay", str(path)]
    else:
        raise RuntimeError("No paplay or aplay found on PATH")

    subprocess.run(cmd, check=True)
