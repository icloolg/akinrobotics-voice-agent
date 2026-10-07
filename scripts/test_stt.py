"""Day-2 test: record from the microphone and transcribe.

    python -m scripts.test_stt            # 5 s recording
    python -m scripts.test_stt 8          # 8 s recording
Prints the text, detected language, processing time and RTF.
"""
import sys

import sounddevice as sd

from app.config import load_config
from app.factory import build_stt
from app.logging_setup import setup_logging
from app.metrics import rtf
from app.providers.base import SAMPLE_RATE

if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])
    stt = build_stt(cfg)

    while True:
        if input(f"\nEnter = {seconds:.0f} sn kayıt, q = çıkış: ").strip().lower() == "q":
            break
        print("Konuşun...")
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()

        t = stt.transcribe(audio[:, 0])
        print(f"Metin : {t.text}")
        print(f"Dil   : {t.language}")
        print(f"Süre  : {t.processing_seconds * 1000:.0f} ms işlem / {t.audio_seconds:.1f} sn ses"
              f" | RTF = {rtf(t.processing_seconds, t.audio_seconds)}")
