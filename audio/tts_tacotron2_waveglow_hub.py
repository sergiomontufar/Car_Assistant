
"""English TTS using Tacotron2 + WaveGlow via PyTorch Hub.

First run may download models into ~/.cache/torch/hub (needs internet once).
After that, it runs fully local/offline.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Tacotron2WaveGlowHubConfig:
    device: str = "cuda"
    sample_rate: int = 22050
    denoiser_strength: float = 0.01


class Tacotron2WaveGlowEnglishTTS:
    def __init__(self, cfg: Tacotron2WaveGlowHubConfig = Tacotron2WaveGlowHubConfig()) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

        # Load models
        self.tacotron2 = torch.hub.load("nvidia/DeepLearningExamples:torchhub", "nvidia_tacotron2")
        self.waveglow = torch.hub.load("nvidia/DeepLearningExamples:torchhub", "nvidia_waveglow")
        self.tacotron2 = self.tacotron2.to(self.device).eval()
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow).to(self.device).eval()

        # Utilities
        self.utils = torch.hub.load("nvidia/DeepLearningExamples:torchhub", "nvidia_tts_utils")
        self.denoiser = self.utils.Denoiser(self.waveglow).to(self.device)
        torch.set_grad_enabled(False)

    @property
    def sample_rate(self) -> int:
        return int(self.cfg.sample_rate)

    def synthesize(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros((0,), dtype=np.float32)

        sequences, lengths = self.utils.prepare_input_sequence([text])
        sequences = sequences.to(self.device)
        lengths = lengths.to(self.device)

        mel, mel_lengths, _ = self.tacotron2.infer(sequences, lengths)
        audio = self.waveglow.infer(mel, sigma=0.666)
        audio = self.denoiser(audio, strength=float(self.cfg.denoiser_strength))

        wav = audio.squeeze().detach().cpu().numpy().astype(np.float32)
        wav = np.clip(wav, -1.0, 1.0)
        return wav
