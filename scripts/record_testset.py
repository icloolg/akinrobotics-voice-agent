"""Record the voice test set (tests/voice/sentences.csv) with your own voice.

    python -m scripts.record_testset          # record all missing sentences
    python -m scripts.record_testset 07       # re-record one sentence

For each sentence: press Enter, read it aloud, press Enter again.
Files are saved to tests/voice/recordings/<id>.wav (16 kHz, mono, 16-bit).
"""
import csv
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

RATE = 16000
ROOT = Path("tests/voice")
OUT = ROOT / "recordings"


def load_sentences() -> list[dict]:
    with open(ROOT / "sentences.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def record_until_enter() -> np.ndarray:
    chunks = []
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                        callback=lambda data, *_: chunks.append(data.copy())):
        input("  ● Kayıt... (bitince Enter)")
    return np.concatenate(chunks)[:, 0]


def save_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.tobytes())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None

    for s in load_sentences():
        path = OUT / f"{s['id']}.wav"
        if only and s["id"] != only:
            continue
        if path.exists() and not only:
            continue
        print(f"\n[{s['id']}] ({s['language']})  \"{s['text']}\"")
        input("  Hazır olunca Enter, sonra cümleyi okuyun")
        audio = record_until_enter()
        save_wav(path, audio)
        print(f"  Kaydedildi: {path} ({len(audio) / RATE:.1f} sn)")

    print("\nBitti. Ölçmek için: python -m scripts.eval_stt")
