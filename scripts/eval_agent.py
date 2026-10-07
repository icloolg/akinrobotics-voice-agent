"""Accuracy + latency test of the agent (text in, no audio).

    python -m scripts.eval_agent
    python -m scripts.eval_agent --top-k 2 --history 1      # try other settings

Questions: tests/agent_questions.csv
  bilgi       -> answer must contain the expected value(s)
                 ("a|b" = any of them, "a&b" = all of them)
  sohbet      -> must be routed to small talk
  kapsam_disi -> must NOT state facts: routed to "no answer", or the LLM says it
                 has no information (e.g. "Ada-7 uçabilir mi?" passes retrieval)

Latency is measured like the voice pipeline sees it: time until the first
complete sentence is ready for TTS ("first sentence"), and total time.
Questions are asked in order as one conversation (history is used).
"""
import argparse
import csv
import random
import time

from app.agent import prompts
from app.agent.agent import AgentResult
from app.config import load_config
from app.factory import build_agent
from app.logging_setup import setup_logging
from app.text import SentenceSplitter

NO_INFO_WORDS = ("bilgi", "bulamadım", "bilmiyorum", "don't have", "couldn't find", "no information", "not ")


def normalize(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").lower()


def is_correct(category: str, expected: str, result: AgentResult) -> bool:
    answer = normalize(result.answer)
    if category == "sohbet":
        return result.route == "chitchat"
    if category == "kapsam_disi":
        return result.route == "no_answer" or any(w in answer for w in NO_INFO_WORDS)
    if "&" in expected:
        return all(normalize(e) in answer for e in expected.split("&"))
    return any(normalize(e) in answer for e in expected.split("|"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--history", type=int)
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    cfg = load_config()
    if args.top_k is not None:
        cfg["rag"]["top_k"] = args.top_k
    if args.history is not None:
        cfg["llm"]["history_turns"] = args.history
    setup_logging("WARNING")
    # Ollama keeps earlier prompts in a cache, so re-running the same questions
    # looks much faster than reality (measured: 2416 -> 1075 ms with no real change).
    # A random tag at the start of the system prompts makes every run start cold;
    # caching *within* the run still works like in real use.
    tag = f"[run {random.randint(0, 10**9)}]\n"
    prompts.SYSTEM = tag + prompts.SYSTEM
    prompts.CHITCHAT_SYSTEM = {k: tag + v for k, v in prompts.CHITCHAT_SYSTEM.items()}
    agent = build_agent(cfg)
    agent.llm.warmup()

    with open("tests/agent_questions.csv", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    rows = []
    for q in questions:
        result, splitter = AgentResult(), SentenceSplitter()
        start = time.perf_counter()
        first_sentence = None
        for token in agent.answer(q["question"], q["language"], result):
            if first_sentence is None and splitter.push(token):
                first_sentence = time.perf_counter() - start
        total = time.perf_counter() - start
        first_sentence = first_sentence or total
        ok = is_correct(q["category"], q["expected"], result)
        rows.append((q, result, ok, first_sentence, total))
        if not args.quiet:
            print(f"{'✓' if ok else '✗'} {first_sentence * 1000:5.0f} ms {total * 1000:5.0f} ms "
                  f"[{result.route:9s}] {q['question']}\n      → {result.answer[:110]}")

    print(f"\nAyarlar: top_k={cfg['rag']['top_k']} history={cfg['llm']['history_turns']}")
    for cat in ("bilgi", "sohbet", "kapsam_disi"):
        sel = [r for r in rows if r[0]["category"] == cat]
        print(f"  {cat:12s} doğru {sum(r[2] for r in sel)}/{len(sel)}")
    rag = [r for r in rows if r[1].route == "rag"]
    avg = lambda xs: sum(xs) / len(xs) * 1000 if xs else 0
    print(f"  TOPLAM doğru {sum(r[2] for r in rows)}/{len(rows)}")
    print(f"  RAG cevapları: ilk cümle ort. {avg([r[3] for r in rag]):.0f} ms, "
          f"en kötü {max((r[3] for r in rag), default=0) * 1000:.0f} ms, toplam ort. {avg([r[4] for r in rag]):.0f} ms")


if __name__ == "__main__":
    main()
