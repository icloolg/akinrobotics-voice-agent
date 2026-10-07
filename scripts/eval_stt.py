"""Measure STT on the recorded voice test set.

    python -m scripts.eval_stt

For every recording: transcript, detected language, WER (word error rate)
against the reference sentence, processing time and RTF.
"""
import re
import wave

import numpy as np

from app.config import load_config
from app.factory import build_stt
from app.logging_setup import setup_logging
from app.metrics import rtf
from scripts.record_testset import OUT, load_sentences


def normalize(text: str) -> list[str]:
    """Lowercase (Turkish-aware), drop punctuation, split into words ("Ada-7" == "Ada 7")."""
    text = text.replace("I", "ı").replace("İ", "i").lower()
    return re.sub(r"[^\w\s]", " ", text).split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate = (substitutions + deletions + insertions) / reference words."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    # Classic edit-distance table over words.
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = range(len(ref) + 1)
    d[0, :] = range(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[-1, -1] / max(len(ref), 1)


def read_wav(path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


if __name__ == "__main__":
    cfg = load_config()
    setup_logging("WARNING")
    stt = build_stt(cfg)

    rows = []
    for s in load_sentences():
        path = OUT / f"{s['id']}.wav"
        if not path.exists():
            continue
        audio = read_wav(path)
        if not rows:
            stt.transcribe(audio)  # warm-up, not measured
        t = stt.transcribe(audio)
        rows.append((s, t, wer(s["text"], t.text)))

    if not rows:
        raise SystemExit("Kayıt yok. Önce: python -m scripts.record_testset")

    print(f"\n{'id':3} {'dil':6} {'WER':>5} {'ms':>6} {'RTF':>6}  metin")
    for s, t, e in rows:
        lang = t.language + ("" if t.language == s["language"] else "✗")
        print(f"{s['id']:3} {lang:6} {e:5.2f} {t.processing_seconds * 1000:6.0f} "
              f"{rtf(t.processing_seconds, t.audio_seconds):6.3f}  {t.text}")

    n = len(rows)
    print(f"\nKayıt: {n} | Dil doğru: {sum(t.language == s['language'] for s, t, _ in rows)}/{n}"
          f" | Ortalama WER: {sum(e for *_, e in rows) / n:.2f}"
          f" | Tam doğru: {sum(e == 0 for *_, e in rows)}/{n}"
          f" | Ortalama süre: {sum(t.processing_seconds for _, t, _ in rows) / n * 1000:.0f} ms"
          f" | Ortalama RTF: {sum(rtf(t.processing_seconds, t.audio_seconds) for _, t, _ in rows) / n:.3f}")
