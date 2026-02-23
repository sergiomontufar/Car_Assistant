
"""Non-blocking audio playback with stop() for barge-in (PyAudio)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pyaudio

AmplitudeCallback = Callable[[float], None]


@dataclass
class _PlaybackState:
    stop_event: threading.Event
    is_playing: bool


class AudioPlayer:
    """Plays float32 audio in [-1,1] at a given sample rate."""

    def __init__(self, sample_rate: int, device_index: Optional[int] = None) -> None:
        self.sample_rate = int(sample_rate)
        self.device_index = device_index
        self._lock = threading.Lock()
        self._state = _PlaybackState(stop_event=threading.Event(), is_playing=False)
        self._thread: Optional[threading.Thread] = None

    def is_playing(self) -> bool:
        with self._lock:
            return self._state.is_playing

    def stop(self) -> None:
        with self._lock:
            self._state.stop_event.set()

        t = self._thread
        if t and t.is_alive():
            t.join(timeout=1.0)

        with self._lock:
            self._state = _PlaybackState(stop_event=threading.Event(), is_playing=False)
            self._thread = None

    def play(self, audio: np.ndarray, amplitude_callback: Optional[AmplitudeCallback] = None) -> None:
        """Start playback in a background thread. Stops any current playback first."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return

        # Stop any current playback
        self.stop()

        with self._lock:
            self._state.is_playing = True
            stop_event = self._state.stop_event

        def _run() -> None:
            pa = pyaudio.PyAudio()
            stream = None
            try:
                stream = pa.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.device_index,
                    frames_per_buffer=1024,
                )

                idx = 0
                block = 1024
                while idx < audio.size and not stop_event.is_set():
                    chunk = audio[idx: idx + block]
                    if chunk.size == 0:
                        break
                    stream.write(chunk.tobytes())

                    if amplitude_callback is not None:
                        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
                        amplitude_callback(rms)

                    idx += block
                    time.sleep(0.001)
            finally:
                try:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
                try:
                    pa.terminate()
                except Exception:
                    pass
                with self._lock:
                    self._state.is_playing = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
