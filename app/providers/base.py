"""Interfaces every component must implement.

The pipeline only talks to these classes, never to a concrete library.
Adding a new model/service = one new subclass + one line in factory.py.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np

SAMPLE_RATE = 16000  # all audio inside the server is 16 kHz mono float32


@dataclass
class Transcript:
    text: str
    language: str        # "tr" | "en"
    audio_seconds: float
    processing_seconds: float


class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        """language None = detect it; "tr"/"en" = the user chose it (no detection)."""


class LLMProvider(ABC):
    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield the answer piece by piece (tokens)."""

    def warmup(self) -> None:
        """Load the model before the first user turn (optional)."""


class TTSProvider(ABC):
    sample_rate: int

    @abstractmethod
    def synthesize(self, text: str, language: str) -> np.ndarray:
        """Return int16 PCM audio at self.sample_rate."""
