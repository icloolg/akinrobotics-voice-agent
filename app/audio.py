"""Small audio helpers shared by server and scripts."""
import numpy as np


def pcm16_to_float(data: bytes) -> np.ndarray:
    """int16 PCM bytes -> float32 in [-1, 1] (what Whisper and Silero expect)."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampling. Good enough for test/warm-up audio."""
    if src_rate == dst_rate:
        return audio
    n = int(len(audio) * dst_rate / src_rate)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
