import glob
import logging
import os
import sys
import time

import numpy as np

from app.providers.base import SAMPLE_RATE, STTProvider, Transcript

log = logging.getLogger(__name__)


def _add_cuda_dlls() -> None:
    """On Windows, make the pip-installed CUDA libraries (nvidia-cublas/cudnn) findable."""
    if sys.platform != "win32":
        return
    site = os.path.dirname(os.path.dirname(np.__file__))
    for d in glob.glob(os.path.join(site, "nvidia", "*", "bin")):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


class FasterWhisperSTT(STTProvider):
    """Speech-to-text + language detection with faster-whisper (CTranslate2).

    Whisper detects the language itself, so no separate language-ID model is
    needed. The result is limited to `languages` (e.g. tr/en): short clips are
    sometimes detected as a similar language, which we then correct.
    """

    def __init__(self, model: str, device: str, compute_type: str, languages: list[str],
                 hotwords: str | None = None, default_language: str = "tr",
                 switch_min_prob: float = 0.3, switch_ratio: float = 2.0):
        if device == "cuda":
            _add_cuda_dlls()
        from faster_whisper import WhisperModel  # after the DLL paths are set

        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.languages = languages
        self.hotwords = hotwords
        self.default_language = default_language
        self.switch_min_prob = switch_min_prob
        self.switch_ratio = switch_ratio
        log.info("Whisper loaded: %s (%s, %s)", model, device, compute_type)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        start = time.perf_counter()

        # language=None -> Whisper detects the language in the same pass (no extra cost).
        segments, info = self._run(audio, language=None)
        language = self._choose_language(dict(info.all_language_probs))
        if language != info.language:
            log.info("Detected '%s' (%.2f) -> using '%s'", info.language, info.language_probability, language)
            segments, _ = self._run(audio, language=language)

        # segments is a lazy generator: the actual decoding happens here.
        text = " ".join(s.text.strip() for s in segments).strip()

        return Transcript(
            text=text,
            language=language,
            audio_seconds=len(audio) / SAMPLE_RATE,
            processing_seconds=time.perf_counter() - start,
        )

    def _choose_language(self, probs: dict[str, float]) -> str:
        """Default language unless another allowed language is clearly more likely.

        Measured on our 18 recordings: short Turkish words ("Nerede?", "Evet.")
        are often detected as English/Russian/French with low confidence, while
        real English sentences get P(en) >= 0.5. Rules compared:
          plain detection 16/18, "most likely of tr/en" 15/18, this rule 18/18.
        """
        p_default = probs.get(self.default_language, 0.0)
        others = [l for l in self.languages if l != self.default_language]
        best = max(others, key=lambda l: probs.get(l, 0.0), default=None)
        if best and probs.get(best, 0.0) >= self.switch_min_prob \
                and probs[best] > self.switch_ratio * p_default:
            return best
        return self.default_language

    def _run(self, audio: np.ndarray, language: str | None):
        return self.model.transcribe(
            audio,
            language=language,
            beam_size=1,                      # greedy decoding: fastest, accurate enough for short questions
            condition_on_previous_text=False,
            without_timestamps=True,          # we only need the text
            vad_filter=False,                 # silence is already cut by our own VAD
            hotwords=self.hotwords,
        )
