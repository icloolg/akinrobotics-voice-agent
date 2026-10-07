import logging
from pathlib import Path

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig

from app.providers.base import TTSProvider

log = logging.getLogger(__name__)


class PiperTTS(TTSProvider):
    """Fast offline TTS (VITS models exported to ONNX). One voice per language."""

    def __init__(self, voices_dir: str, voices: dict[str, str], length_scale: float = 1.0):
        self.syn_config = SynthesisConfig(length_scale=length_scale)
        self.voices = {
            lang: PiperVoice.load(Path(voices_dir) / f"{name}.onnx")
            for lang, name in voices.items()
        }
        rates = {v.config.sample_rate for v in self.voices.values()}
        if len(rates) != 1:
            raise ValueError(f"All voices must share one sample rate, got {rates}")
        self.sample_rate = rates.pop()
        log.info("Piper voices loaded: %s @ %d Hz", voices, self.sample_rate)

    def synthesize(self, text: str, language: str) -> np.ndarray:
        voice = self.voices.get(language) or next(iter(self.voices.values()))
        chunks = [c.audio_int16_array for c in voice.synthesize(text, self.syn_config)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
