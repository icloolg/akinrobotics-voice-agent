"""Voice activity detection: cuts the continuous mic stream into utterances.

The client streams audio non-stop. Silero VAD scores every 32 ms chunk as
speech / not speech. An utterance starts at the first speech chunk and ends
after `min_silence_ms` of silence; then the whole utterance goes to STT.

The segmenter also counts how much of the current utterance was really
voiced (`voiced_ms`). Two uses:
  - barge-in: the server interrupts the agent only after `barge_in_ms` of
    voiced audio, so a click or a bit of speaker echo does not cut the answer;
  - utterances with less than `min_speech_ms` of voice (cough, door, key
    press) are dropped instead of being sent to Whisper, which would invent
    words for them.
"""
import collections

import numpy as np
from silero_vad import VADIterator, load_silero_vad

from app.providers.base import SAMPLE_RATE

CHUNK = 512  # Silero needs exactly 512 samples (32 ms) per call at 16 kHz
CHUNK_MS = CHUNK * 1000 / SAMPLE_RATE


class _ProbeModel:
    """Wraps the Silero model to remember the last speech probability, which
    VADIterator computes but does not expose."""

    def __init__(self, model):
        self.model = model
        self.last = 0.0

    def __call__(self, x, sr):
        prob = self.model(x, sr)
        self.last = float(prob)
        return prob

    def reset_states(self):
        self.model.reset_states()


class Segmenter:
    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 500,
                 pre_roll_ms: int = 300, max_seconds: float = 15.0,
                 min_speech_ms: int = 100):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.model = _ProbeModel(load_silero_vad(onnx=True))
        self.vad = VADIterator(self.model, threshold=threshold,
                               sampling_rate=SAMPLE_RATE, min_silence_duration_ms=min_silence_ms)
        # Keep a little audio from *before* speech was detected so the first
        # syllable is not cut off.
        self.pre_roll = collections.deque(maxlen=pre_roll_ms * SAMPLE_RATE // 1000 // CHUNK)
        self.max_chunks = int(max_seconds * SAMPLE_RATE / CHUNK)
        self.pending = np.zeros(0, dtype=np.float32)
        self.speech: list[np.ndarray] | None = None  # None = not inside an utterance
        self.voiced_ms = 0.0       # voiced audio in the current utterance
        self.last_voiced_ms = 0.0  # ... and in the last finished one

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
                self.voiced_ms = CHUNK_MS
            return None

        self.speech.append(chunk)
        if self.model.last >= self.threshold:
            self.voiced_ms += CHUNK_MS
        if (event and "end" in event) or len(self.speech) >= self.max_chunks:
            utterance, self.last_voiced_ms = np.concatenate(self.speech), self.voiced_ms
            self.reset(keep_pending=True)
            return utterance if self.last_voiced_ms >= self.min_speech_ms else None
        return None

    @property
    def in_speech(self) -> bool:
        return self.speech is not None

    def reset(self, keep_pending: bool = False) -> None:
        self.vad.reset_states()
        self.pre_roll.clear()
        if not keep_pending:
            self.pending = np.zeros(0, dtype=np.float32)
        self.speech = None
        self.voiced_ms = 0.0
