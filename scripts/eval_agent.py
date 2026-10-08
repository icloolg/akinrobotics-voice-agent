"""Accuracy + latency test of the agent (text in, no audio).

    python -m scripts.eval_agent
    python -m scripts.eval_agent --top-k 2 --history 1      # try other settings
    python -m scripts.eval_agent --questions tests/agent_questions_heldout.csv
        # questions never used while tuning: run once, report as is, do not tune on them

Questions: tests/agent_questions.csv
  bilgi       -> answer must contain the expected value(s)
                 ("a|b" = any of them, "a&b" = all of them, "a&!b" = a but not b)
  sohbet      -> must be routed to small talk
  kapsam_disi -> off-topic ("Bugün hava nasıl?"): must say it has no information
  cevapsiz    -> on-topic but NOT in the documents ("Ada-7 uçabilir mi?"): passes
                 the retrieval threshold, so only the LLM can refuse. Must say it
                 has no information; any yes/no/number is a hallucination.
  cikarim     -> needs two facts combined ("AKINSOFT ile AKINROBOTICS farkı"); checked like bilgi
  takip       -> follow-up without the topic ("Ne zaman kurulmuş?" right after
                 "AKINSOFT nedir?"); checked like bilgi
  arac:<name> -> must be routed to that tool and the answer must contain the expected value

If the mock robot API (served by app/main.py) is not running, it is started
in a background thread so the robot_status tool can be tested on its own.

Latency is measured like the voice pipeline sees it: time until the first
complete sentence is ready for TTS ("first sentence"), and total time.
Questions are asked in order as one conversation (history is used).
"""
import argparse
import csv
import random
import time

import requests

from app.agent import prompts
from app.agent.agent import AgentResult
from app.config import load_config
from app.agent.verify import verified_sentences
from app.factory import build_agent, build_verifier
from app.logging_setup import setup_logging

# An answer counts as "no information" only if it says so. "Hayır, yüzemez" is
# also wrong for an unanswerable question: the documents do not say that either.
NO_INFO_WORDS = ("bilgi", "bulamadım", "bilmiyorum", "belirtilmem", "yer almıyor",
                 "information", "don't know", "not mentioned", "not specified", "couldn't find")


def normalize(text: str) -> str:
    # Fold Turkish dotless/dotted i so "AKINCI-5" == "Akinci-5" when comparing.
    return text.lower().replace("ı", "i").replace("i̇", "i")


def is_correct(category: str, expected: str, result: AgentResult) -> bool:
    answer = normalize(result.answer)
    if category.startswith("arac:"):
        if result.tool != category.split(":", 1)[1]:
            return False
        category = "bilgi"
    if category in ("takip", "cikarim"):
        category = "bilgi"
    if category == "sohbet":
        return result.route == "chitchat"
    if category in ("kapsam_disi", "cevapsiz"):
        refused = result.route == "no_answer" or any(w in answer for w in NO_INFO_WORDS)
        # "!x&!y": words that must NOT appear (an invented fact after the refusal)
        invented = any(normalize(e[1:]) in answer for e in expected.split("&") if e.startswith("!"))
        return refused and not invented
    if "&" in expected:  # "a&b&!c": a and b must appear, c must not
        parts = expected.split("&")
        return (all(normalize(e) in answer for e in parts if not e.startswith("!"))
                and not any(normalize(e[1:]) in answer for e in parts if e.startswith("!")))
    return any(normalize(e) in answer for e in expected.split("|"))


def ensure_mock_api(url: str) -> None:
    try:
        requests.get(url, timeout=1)
        return
    except requests.RequestException:
        pass
    import threading
    import uvicorn
    from fastapi import FastAPI
    from app.mock_robot_api import router
    app = FastAPI()
    app.include_router(router)
    port = int(url.split(":")[2].split("/")[0])
    threading.Thread(target=uvicorn.run, args=(app,), kwargs={"port": port, "log_level": "warning"},
                     daemon=True).start()
    time.sleep(1.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--history", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--verify", choices=["on", "off"], help="NLI sentence check (default: config)")
    parser.add_argument("--combine", choices=["on", "off"],
                        help="prompt lets the LLM combine facts (default: on when verify is on)")
    parser.add_argument("--questions", default="tests/agent_questions.csv",
                        help="tests/agent_questions_heldout.csv = questions never used while tuning")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    cfg = load_config()
    if args.top_k is not None:
        cfg["rag"]["top_k"] = args.top_k
    if args.temperature is not None:
        cfg["llm"]["temperature"] = args.temperature
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
    if "robot_status" in cfg.get("tools", {}).get("enabled", []):
        ensure_mock_api(cfg["tools"]["robot_status"]["url"])
    if args.verify:
        cfg.setdefault("verify", {})["enabled"] = args.verify == "on"
    agent = build_agent(cfg)
    if args.combine:
        agent.allow_combining = args.combine == "on"
    verifier = build_verifier(cfg)
    agent.llm.warmup()

    with open(args.questions, encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    rows = []
    for q in questions:
        result = AgentResult()
        start = time.perf_counter()
        first_sentence = None
        # Same sentence check as the voice pipeline: we measure what would be spoken.
        for _ in verified_sentences(agent.answer(q["question"], q["language"], result),
                                    result, verifier, q["language"]):
            if first_sentence is None:
                first_sentence = time.perf_counter() - start
        result.answer = result.spoken or result.answer
        total = time.perf_counter() - start
        first_sentence = first_sentence or total
        ok = is_correct(q["category"], q["expected"], result)
        rows.append((q, result, ok, first_sentence, total))
        if not args.quiet:
            print(f"{'✓' if ok else '✗'} {first_sentence * 1000:5.0f} ms {total * 1000:5.0f} ms "
                  f"[{(result.tool or result.route):12s}] {q['question']}\n      → {result.answer[:110]}")

    print(f"\nAyarlar: top_k={cfg['rag']['top_k']} history={cfg['llm']['history_turns']} "
          f"temperature={cfg['llm']['temperature']}")
    for cat in ("bilgi", "cikarim", "takip", "sohbet", "kapsam_disi", "cevapsiz", "arac"):
        sel = [r for r in rows if r[0]["category"].split(":")[0] == cat]
        print(f"  {cat:12s} doğru {sum(r[2] for r in sel)}/{len(sel)}")
    rag = [r for r in rows if r[1].route == "rag"]
    avg = lambda xs: sum(xs) / len(xs) * 1000 if xs else 0
    print(f"  TOPLAM doğru {sum(r[2] for r in rows)}/{len(rows)}")
    print(f"  RAG cevapları: ilk cümle ort. {avg([r[3] for r in rag]):.0f} ms, "
          f"en kötü {max((r[3] for r in rag), default=0) * 1000:.0f} ms, toplam ort. {avg([r[4] for r in rag]):.0f} ms")


if __name__ == "__main__":
    main()
