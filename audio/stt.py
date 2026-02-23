"""Speech-to-text utilities powered by faster-whisper."""

from __future__ import annotations

from typing import Optional

import numpy as np
from faster_whisper import WhisperModel


class FasterWhisperSTT:
    """Wrapper around :mod:`faster_whisper` for low-latency transcription."""

    def __init__(
        self,
        model_size_or_path: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
    ) -> None:
        print(f"-> Loading faster-whisper model '{model_size_or_path}' on {device} ({compute_type}).")
        self.model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type)
        self.language = language
        self.last_language = ""
        self.last_language_probability = 0.0

    def run_stt_with_lang(self, raw_bytes: bytes, sample_rate: int = 16000) -> tuple[str, str]:
        """Return (text, detected_language). detected_language is 'en', 'es', etc."""
        if not raw_bytes:
            return "", ""

        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio_np.size == 0:
            return "", ""

        segments, info = self.model.transcribe(
            audio_np,
            language=self.language,      # None => auto-detect
            task="transcribe",
            beam_size=1,
            vad_filter=False,
            suppress_blank=True,
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()
        lang = getattr(info, "language", "") or ""
        self.last_language = lang
        self.last_language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        return text, lang

    def run_stt(self, raw_bytes: bytes, sample_rate: int = 16000) -> str:
        """Backwards-compatible: returns only text."""
        text, _lang = self.run_stt_with_lang(raw_bytes, sample_rate=sample_rate)
        return text


__all__ = ["FasterWhisperSTT"]
