"""Day-2 test: type a sentence, hear it spoken.

    python -m scripts.test_tts
Prints synthesis time and RTF. Language is guessed like in chat_text.
"""
import time

import sounddevice as sd

from app.config import load_config
from app.factory import build_tts
from app.logging_setup import setup_logging
from app.metrics import rtf
from scripts.chat_text import guess_language

if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])
    tts = build_tts(cfg)
    print("\nCümle yazın (çıkmak için boş satır).\n")

    while text := input("Metin: ").strip():
        lang = guess_language(text)
        start = time.perf_counter()
        audio = tts.synthesize(text, lang)
        took = time.perf_counter() - start
        seconds = len(audio) / tts.sample_rate
        print(f"  [dil={lang} | {took * 1000:.0f} ms işlem / {seconds:.1f} sn ses | RTF = {rtf(took, seconds)}]")
        sd.play(audio, tts.sample_rate)
        sd.wait()
