"""Day-1 test: talk to the agent by typing (no audio yet).

    python -m scripts.chat_text
Language is guessed from the text here; in the voice pipeline Whisper detects it.
"""
import time

from app.agent.agent import AgentResult
from app.config import load_config
from app.factory import build_agent
from app.logging_setup import setup_logging

TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
TR_WORDS = {"ne", "nedir", "nasıl", "kaç", "mi", "mı", "mu", "mü", "bir", "ve", "hangi", "neden",
            "sen", "kimsin", "merhaba", "selam", "teşekkürler"}


def guess_language(text: str) -> str:
    words = set(text.lower().replace("?", " ").split())
    return "tr" if (TR_CHARS & set(text)) or (TR_WORDS & words) else "en"


if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])
    agent = build_agent(cfg)
    agent.llm.warmup()
    print("\nSoru yazın (çıkmak için boş satır).\n")

    while question := input("Sen: ").strip():
        lang = guess_language(question)
        result = AgentResult()
        start = time.perf_counter()
        first = None
        print("Ajan: ", end="", flush=True)
        for piece in agent.answer(question, lang, result):
            first = first or time.perf_counter()
            print(piece, end="", flush=True)
        total = time.perf_counter() - start
        print(f"\n  [dil={lang} | kaynak={[c.source for c in result.chunks if result.grounded]}"
              f" | skorlar={[c.score for c in result.chunks]}"
              f" | ilk token={(first - start) * 1000:.0f} ms | toplam={total * 1000:.0f} ms]\n")
