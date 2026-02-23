
"""Spanish (and multilingual) TTS using Coqui XTTS-v2.

Requires: pip install TTS
First run downloads the model into ~/.local/share/tts (or the library cache).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class XTTSConfig:
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = "cuda"  # 'cuda' or 'cpu'
    speaker_wav: Optional[str] = None  # Optional reference wav for voice cloning
    language: str = "es"


class XTTSTTS:
    def __init__(self, cfg: XTTSConfig):
        self.cfg = cfg
        from TTS.api import TTS  # type: ignore

        self.tts = TTS(cfg.model_name, gpu=(cfg.device == "cuda"))
        # XTTS sample rate depends on model; TTS API exposes it
        self._sr = int(getattr(self.tts.synthesizer, "output_sample_rate", 24000) or 24000)

    @property
    def sample_rate(self) -> int:
        return self._sr

    def synthesize(self, text: str, language: Optional[str] = None) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros((0,), dtype=np.float32)

        lang = language or self.cfg.language

        wav = self.tts.tts(
            text=text,
            speaker_wav=self.cfg.speaker_wav,
            language=lang,
        )
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        # normalize if needed
        if wav.size and (np.max(np.abs(wav)) > 1.0):
            wav = wav / (np.max(np.abs(wav)) + 1e-9)
        return wav
