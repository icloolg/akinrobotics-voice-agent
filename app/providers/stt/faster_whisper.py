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

    def __init__(self, model: str, device: str, compute_type: str,
                 languages: list[str], hotwords: str | None = None):
        if device == "cuda":
            _add_cuda_dlls()
        from faster_whisper import WhisperModel  # after the DLL paths are set

        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.languages = languages
        self.hotwords = hotwords
        log.info("Whisper loaded: %s (%s, %s)", model, device, compute_type)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        start = time.perf_counter()

        # language=None -> Whisper detects the language in the same pass (no extra cost).
        segments, info = self._run(audio, language=None)
        language = info.language
        if language not in self.languages:
            # Rare: short clip detected as a similar language. Redo with the best allowed one.
            allowed = {lang: p for lang, p in info.all_language_probs if lang in self.languages}
            language = max(allowed, key=allowed.get)
            log.info("Detected '%s' not allowed -> retrying as '%s'", info.language, language)
            segments, _ = self._run(audio, language=language)

        # segments is a lazy generator: the actual decoding happens here.
        text = " ".join(s.text.strip() for s in segments).strip()

        return Transcript(
            text=text,
            language=language,
            audio_seconds=len(audio) / SAMPLE_RATE,
            processing_seconds=time.perf_counter() - start,
        )

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
