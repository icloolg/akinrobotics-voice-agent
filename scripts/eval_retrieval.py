"""Search quality without the LLM: is the expected answer in the chunks the LLM gets?

    python -m scripts.eval_retrieval

For every bilgi/cikarim question of both question sets (development and
held-out), the expected value ("a|b" any, "a&b" all) must appear in the text
of the top-k chunks. Compares dense (embedding), bm25 (keywords) and hybrid
(both, Reciprocal Rank Fusion). Section titles are not used as labels, so
this works for the downloaded web pages too.
"""
import csv
import time

from app.config import load_config
from app.factory import build_retriever
from scripts.eval_agent import normalize

SETS = {"geliştirme": "tests/agent_questions.csv", "ayrılmış": "tests/agent_questions_heldout.csv"}


def found(expected: str, text: str) -> bool:
    """"a&b|c": (a and b) or c. "!x" parts (must-not-appear) are ignored here."""
    text = normalize(text)
    return any(all(normalize(p) in text for p in group.split("&") if p and not p.startswith("!"))
               for group in expected.split("|"))


if __name__ == "__main__":
    cfg = load_config()
    retriever = build_retriever(cfg)
    k = cfg["rag"]["top_k"]
    questions = []
    for name, path in SETS.items():
        with open(path, encoding="utf-8") as f:
            questions += [(name, q) for q in csv.DictReader(f) if q["category"] in ("bilgi", "cikarim")]

    print(f"{len(questions)} soru, ilk {k} parça\n")
    print(f"{'yöntem':8s} {'geliştirme':>12s} {'ayrılmış':>10s} {'toplam':>8s} {'süre':>8s}")
    for mode in ("dense", "bm25", "hybrid"):
        hits = {name: [0, 0] for name in SETS}
        start = time.perf_counter()
        for name, q in questions:
            chunks = retriever.search(q["question"], k, mode=mode).chunks
            hits[name][0] += found(q["expected"], " ".join(c.text for c in chunks))
            hits[name][1] += 1
        ms = (time.perf_counter() - start) * 1000 / len(questions)
        total = sum(h[0] for h in hits.values())
        print(f"{mode:8s} " + " ".join(f"{h[0]:>6d}/{h[1]:<4d}" for h in hits.values())
              + f" {total:>4d}/{len(questions):<3d} {ms:5.0f} ms")
