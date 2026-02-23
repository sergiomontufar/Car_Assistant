"""
Language router for TTS engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class _TTS(Protocol):
    sample_rate: int
    def speak_to_wav(self, text: str, out_path: Path | str | None = None) -> Path: ...


@dataclass
class TTSRouter:
    en_tts: _TTS
    es_tts: _TTS

    def speak(self, text: str, lang: str) -> tuple[Path, int]:
        lang = (lang or "en").lower()
        if lang.startswith("es"):
            wav = self.es_tts.speak_to_wav(text)
            return wav, self.es_tts.sample_rate
        wav = self.en_tts.speak_to_wav(text)
        return wav, self.en_tts.sample_rate
