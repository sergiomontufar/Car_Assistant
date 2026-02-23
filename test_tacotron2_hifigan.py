import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch


# --------- paths ----------
BASE = os.path.expanduser("~/Rowdy_chatbot")
HIFIGAN_DIR = os.path.join(BASE, "hifi-gan")
HIFIGAN_CKPT = os.path.join(BASE, "models", "hifigan", "generator_v1")
HIFIGAN_CFG  = os.path.join(BASE, "models", "hifigan", "config_v1.json")
OUT_WAV = os.path.join(BASE, "test_en.wav")


def as_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: as_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [as_namespace(x) for x in d]
    return d


def main():
    print("CUDA:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load Tacotron2 (torchhub) with CPU checkpoint load patch ----
    _real_load = torch.load
    def cpu_load(*args, **kw):
        kw.setdefault("map_location", "cpu")
        return _real_load(*args, **kw)
    torch.load = cpu_load

    tacotron2 = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_tacotron2",
        pretrained=True
    )

    torch.load = _real_load
    tacotron2.eval().to(device)

    # ---- Load HiFi-GAN generator ----
    sys.path.insert(0, HIFIGAN_DIR)
    from models import Generator  # noqa: E402

    if not os.path.exists(HIFIGAN_CKPT):
        raise FileNotFoundError(f"Missing HiFi-GAN checkpoint: {HIFIGAN_CKPT}")
    if not os.path.exists(HIFIGAN_CFG):
        raise FileNotFoundError(f"Missing HiFi-GAN config: {HIFIGAN_CFG}")

    with open(HIFIGAN_CFG, "r", encoding="utf-8") as f:
        h = as_namespace(json.load(f))

    hifigan = Generator(h).eval().to(device)

    state = torch.load(HIFIGAN_CKPT, map_location="cpu")
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    hifigan.load_state_dict(state, strict=True)

    # ---- Prepare text and infer mel ----
    text = "Hello Sergio. English TTS with Tacotron2 plus HiFi-GAN is working."

    with torch.inference_mode():
        try:
            # Some torchhub variants accept raw text
            mel, _, _ = tacotron2.infer(text)
        except TypeError:
            # Others require tokenized inputs
            hub_root = os.path.expanduser("~/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub")
            tacotron_repo = os.path.join(hub_root, "PyTorch/SpeechSynthesis/Tacotron2")
            # Load the torchhub inference.py directly to avoid picking up local inference modules
            import importlib.util

            # Minimal stub for dllogger used inside the hub script
            class _DummyBackend:
                def __init__(self, *a, **kw): pass
                def log(self, *a, **kw): pass
                def flush(self): pass

            class _DummyVerbosity:
                DEFAULT = 0
                INFO = 1
                DEBUG = 2

            class _DummyDLLogger:
                @staticmethod
                def init(backends=None): pass
                @staticmethod
                def log(*a, **kw): pass
                @staticmethod
                def flush(): pass

            sys.modules.setdefault("dllogger", _DummyDLLogger)
            sys.modules.setdefault("tacotron2_inference_dllogger_stub", _DummyDLLogger)
            sys.modules["dllogger"].StdOutBackend = _DummyBackend
            sys.modules["dllogger"].JSONStreamBackend = _DummyBackend
            sys.modules["dllogger"].Verbosity = _DummyVerbosity

            inference_path = os.path.join(tacotron_repo, "inference.py")
            spec = importlib.util.spec_from_file_location("tacotron2_inference", inference_path)
            inference_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inference_mod)  # type: ignore
            prepare_input_sequence = inference_mod.prepare_input_sequence

            seq, seq_len = prepare_input_sequence([text], cpu_run=(device.type == "cpu"))
            seq = seq.to(device)
            seq_len = seq_len.to(device)
            mel, _, _ = tacotron2.infer(seq, seq_len)

        mel = mel.to(device)
        audio = hifigan(mel).squeeze().detach().cpu().numpy().astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)

    sf.write(OUT_WAV, audio, 22050)
    print("Wrote:", OUT_WAV)


if __name__ == "__main__":
    main()
