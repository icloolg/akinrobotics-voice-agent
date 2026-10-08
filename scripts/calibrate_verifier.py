"""Collect NLI scores of answer sentences to choose where/how to verify.

    python -m scripts.calibrate_verifier

Runs the agent with the combining prompt and NO verification, splits each
answer into the sentences that would be spoken, and scores every sentence
against its context with the NLI model. Each sentence is labeled:
  keep -> from a correctly answered bilgi/cikarim/takip question
  drop -> from a cevapsiz/kapsam_disi question and not a refusal (= invented)
Prints the scores by route and language, and how many of each group a
threshold would let through.
"""
import random
import time

from app.agent import prompts
from app.agent.agent import AgentResult
from app.config import load_config
from app.factory import build_agent
from app.logging_setup import setup_logging
from app.text import SentenceSplitter
from scripts.eval_agent import NO_INFO_WORDS, ensure_mock_api, is_correct, normalize

import csv

if __name__ == "__main__":
    cfg = load_config()
    setup_logging("WARNING")
    cfg["verify"]["enabled"] = False
    tag = f"[run {random.randint(0, 10**9)}]\n"          # cold Ollama cache, like eval_agent
    prompts.SYSTEM = tag + prompts.SYSTEM
    ensure_mock_api(cfg["tools"]["robot_status"]["url"])
    agent = build_agent(cfg)
    agent.allow_combining = True
    agent.llm.warmup()
    from app.agent.verify import Verifier
    verifier = Verifier(cfg["verify"]["model_dir"])

    with open("tests/agent_questions.csv", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    rows = []
    for q in questions:
        result = AgentResult()
        splitter, sentences = SentenceSplitter(), []
        for token in agent.answer(q["question"], q["language"], result):
            sentences += splitter.push(token)
        sentences += splitter.flush()
        if not result.passages:                     # small talk / fixed reply: nothing to verify
            continue
        cat = q["category"].split(":")[0]
        correct = is_correct(q["category"], q["expected"], result)
        for s in sentences:
            refusal = any(w in normalize(s) for w in NO_INFO_WORDS)
            if cat in ("bilgi", "cikarim", "takip", "arac") and correct and not refusal:
                label = "keep"
            elif cat in ("cevapsiz", "kapsam_disi") and not refusal:
                label = "drop"
            else:
                continue
            t = time.perf_counter()
            score = verifier.score(s, result.passages)
            rows.append((label, result.route, q["language"], score, (time.perf_counter() - t) * 1000, s))

    print(f"\n{'etiket':6s} {'yol':5s} {'dil':3s} {'skor':>5s}  cümle")
    for label, route, lang, score, ms, s in sorted(rows, key=lambda r: (r[0], r[3])):
        print(f"{label:6s} {route:5s} {lang:3s} {score:5.2f}  {s[:90]}")

    print("\nEşik  | geçen 'keep' (doğru)            | geçen 'drop' (uydurma)")
    for th in (0.05, 0.1, 0.2, 0.3, 0.5):
        for name, sel in (("hepsi", lambda r: True), ("sadece rag+tr", lambda r: r[1] == "rag" and r[2] == "tr")):
            keep = [r for r in rows if r[0] == "keep" and sel(r)]
            drop = [r for r in rows if r[0] == "drop" and sel(r)]
            print(f"{th:4.2f} {name:14s} {sum(r[3] >= th for r in keep):3d}/{len(keep):<3d}"
                  f"                     {sum(r[3] >= th for r in drop):3d}/{len(drop)}")
    print(f"\nOrtalama NLI süresi: {sum(r[4] for r in rows) / len(rows):.0f} ms/cümle")
