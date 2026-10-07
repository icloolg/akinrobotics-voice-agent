"""Voice activity detection: cuts the continuous mic stream into utterances.

The client streams audio non-stop. Silero VAD scores every 32 ms chunk as
speech / not speech. An utterance starts at the first speech chunk and ends
after `min_silence_ms` of silence; then the whole utterance goes to STT.
"""
import collections

import numpy as np
from silero_vad import VADIterator, load_silero_vad

from app.providers.base import SAMPLE_RATE

CHUNK = 512  # Silero needs exactly 512 samples (32 ms) per call at 16 kHz


class Segmenter:
    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 500,
                 pre_roll_ms: int = 300, max_seconds: float = 15.0):
        self.vad = VADIterator(load_silero_vad(onnx=True), threshold=threshold,
                               sampling_rate=SAMPLE_RATE, min_silence_duration_ms=min_silence_ms)
        # Keep a little audio from *before* speech was detected so the first
        # syllable is not cut off.
        self.pre_roll = collections.deque(maxlen=pre_roll_ms * SAMPLE_RATE // 1000 // CHUNK)
        self.max_chunks = int(max_seconds * SAMPLE_RATE / CHUNK)
        self.pending = np.zeros(0, dtype=np.float32)
        self.speech: list[np.ndarray] | None = None  # None = not inside an utterance

    def push(self, audio: np.ndarray) -> list[np.ndarray]:
        """Add mic audio (any length). Returns finished utterances (usually 0 or 1)."""
        self.pending = np.concatenate([self.pending, audio])
        done = []
        while len(self.pending) >= CHUNK:
            chunk, self.pending = self.pending[:CHUNK], self.pending[CHUNK:]
            if (utterance := self._process(chunk)) is not None:
                done.append(utterance)
        return done

    def _process(self, chunk: np.ndarray) -> np.ndarray | None:
        event = self.vad(chunk)
        if self.speech is None:
            self.pre_roll.append(chunk)
            if event and "start" in event:
                self.speech = list(self.pre_roll)
            return None

        self.speech.append(chunk)
        if (event and "end" in event) or len(self.speech) >= self.max_chunks:
            utterance = np.concatenate(self.speech)
            self.reset()
            return utterance
        return None

    @property
    def in_speech(self) -> bool:
        return self.speech is not None

    def reset(self) -> None:
        self.vad.reset_states()
        self.pre_roll.clear()
        self.pending = np.zeros(0, dtype=np.float32)
        self.speech = None
