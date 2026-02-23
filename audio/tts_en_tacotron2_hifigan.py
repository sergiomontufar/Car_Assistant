"""
English TTS using Tacotron2 (TorchHub) + HiFi-GAN (local checkpoint).

- Based on the verified working script test_tacotron2_hifigan.py.
- Handles both TorchHub infer signatures (raw text vs token IDs).
- Stubs dllogger so we can import the TorchHub inference helper without extra deps.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
import sys

import numpy as np
import soundfile as sf
import torch


@dataclass
class Tacotron2HiFiGanConfig:
    device: str = "cuda"
    sample_rate: int = 22050
    text_cleaners: Tuple[str, ...] = ("english_cleaners",)
    out_dir: Path = Path("/tmp")


def _as_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _as_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_as_namespace(x) for x in d]
    return d


class _DLLLoggerStub:
    class _Backend:
        def __init__(self, *a, **kw): ...
        def log(self, *a, **kw): ...
        def flush(self): ...

    class _Verbosity:
        DEFAULT = 0
        INFO = 1
        DEBUG = 2

    @staticmethod
    def init(backends=None): ...
    @staticmethod
    def log(*a, **kw): ...
    @staticmethod
    def flush(): ...


class Tacotron2HiFiGanEnglishTTS:
    def __init__(self, cfg: Tacotron2HiFiGanConfig = Tacotron2HiFiGanConfig()) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

        base = Path(__file__).resolve().parent.parent
        self._hifigan_dir = base / "hifi-gan"
        self._hifigan_ckpt = base / "models" / "hifigan" / "generator_v1"
        self._hifigan_cfg = base / "models" / "hifigan" / "config_v1.json"

        if not self._hifigan_ckpt.exists():
            raise FileNotFoundError(f"Missing HiFi-GAN checkpoint: {self._hifigan_ckpt}")
        if not self._hifigan_cfg.exists():
            raise FileNotFoundError(f"Missing HiFi-GAN config: {self._hifigan_cfg}")

        # Load Tacotron2 with CPU map_location to reduce GPU use during load
        self.tacotron2 = self._load_tacotron2()
        self.tacotron2.eval().to(self.device)

        # Load HiFi-GAN
        sys.path.insert(0, str(self._hifigan_dir))
        from models import Generator  # type: ignore  # noqa: E402

        with open(self._hifigan_cfg, "r", encoding="utf-8") as f:
            h = _as_namespace(json.load(f))

        self.hifigan = Generator(h).eval().to(self.device)
        state = torch.load(self._hifigan_ckpt, map_location="cpu")
        if isinstance(state, dict) and "generator" in state:
            state = state["generator"]
        self.hifigan.load_state_dict(state, strict=True)

    @property
    def sample_rate(self) -> int:
        return int(self.cfg.sample_rate)

    def _load_tacotron2(self):
        real_load = torch.load

        def cpu_load(*args, **kw):
            kw.setdefault("map_location", "cpu")
            return real_load(*args, **kw)

        torch.load = cpu_load
        try:
            model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_tacotron2", pretrained=True)
        finally:
            torch.load = real_load
        return model

    def _prepare_sequence(self, text: str):
        hub_root = Path.home() / ".cache" / "torch" / "hub" / "NVIDIA_DeepLearningExamples_torchhub"
        tacotron_repo = hub_root / "PyTorch" / "SpeechSynthesis" / "Tacotron2"

        # Stub dllogger expected by the hub inference script
        sys.modules.setdefault("dllogger", _DLLLoggerStub)
        sys.modules["dllogger"].StdOutBackend = _DLLLoggerStub._Backend
        sys.modules["dllogger"].JSONStreamBackend = _DLLLoggerStub._Backend
        sys.modules["dllogger"].Verbosity = _DLLLoggerStub._Verbosity

        inference_path = tacotron_repo / "inference.py"
        spec = importlib.util.spec_from_file_location("tacotron2_inference", inference_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Tacotron2 inference helper from {inference_path}")
        inference_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference_mod)  # type: ignore

        return inference_mod.prepare_input_sequence([text], cpu_run=(self.device.type == "cpu"))

    def synthesize(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros((0,), dtype=np.float32)

        with torch.inference_mode():
            try:
                mel, _, _ = self.tacotron2.infer(text)
            except TypeError:
                seq, seq_len = self._prepare_sequence(text)
                seq = seq.to(self.device)
                seq_len = seq_len.to(self.device)
                mel, _, _ = self.tacotron2.infer(seq, seq_len)

            mel = mel.to(self.device)
            audio = self.hifigan(mel).squeeze().detach().cpu().numpy().astype(np.float32)
            audio = np.clip(audio, -1.0, 1.0)
            return audio

    def speak_to_wav(self, text: str, out_path: Path | str | None = None) -> Path:
        out_path = Path(out_path) if out_path else Path(self.cfg.out_dir) / "tts_en.wav"
        wav = self.synthesize(text)
        sf.write(out_path, wav, self.sample_rate)
        return out_path
