"""Answer verification: is each sentence supported by the context? (NLI)

A strict "refuse unless clearly answered" prompt prevents hallucination but
also refuses answers that combine two facts from the context. Instead, the
prompt allows combining, and every sentence the LLM writes is checked before
it is spoken: a natural language inference (NLI) model decides whether the
context entails it. Unsupported sentences are dropped; if none remain, the
fixed "no information" reply is spoken. This is the faithfulness metric of RAG
evaluation frameworks (e.g. RAGAS), applied online, sentence by sentence. It
checks faithfulness, not relevance.

Model: mDeBERTa-v3-base-xnli (multilingual, Turkish included), run with ONNX
Runtime. Measured on 12 sentences from our documents: 11/12 right decisions
("Mini Ada yüzebilir", "24 saat dayanır", "1990'da kuruldu" rejected; the
AKINSOFT/AKINROBOTICS combination accepted). In PyTorch it took 1.6 s per
check on this CPU, in ONNX Runtime 142 ms (~340 ms with real 2-passage contexts).

Threshold 0.10, not the usual 0.5 (scripts/calibrate_verifier.py): correct
answer sentences scored anywhere from 0.01 to 1.00 (numbers, lists, broken
Turkish grammar, English answers to Turkish context); at 0.5, 13 of 55 correct
sentences were dropped and knowledge accuracy fell from 21/27 to 13/27.
The ONNX file is made by: python -m scripts.download_models --all
"""
import json
import re
import logging
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)

MAX_TOKENS = 512  # the model's input limit (context + sentence)
# Letters of scripts other than Latin (CJK, Cyrillic, Greek, Arabic, Hebrew, Hangul...).
# Qwen sometimes drifts into Chinese ("AKINCI-5机器人是最快的。"); the multilingual NLI
# model accepts such a sentence because its meaning is right, and the TTS cannot say it.
_FOREIGN_SCRIPT = re.compile(r"[^\W\d_a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛéèêëàáäôóòñ]")
# Clause boundaries where an appended claim usually starts.
_CLAUSE = re.compile(r",\s+|;\s+|\s+(?:veya|ya da|yani|olup|ve bu|or|which is|that is)\s+", re.IGNORECASE)


class Verifier:
    def __init__(self, model_dir: str, threshold: float = 0.10, threads: int = 6):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        id2label = json.loads((Path(model_dir) / "config.json").read_text())["id2label"]
        self.entail = next(int(i) for i, name in id2label.items() if name == "entailment")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.log_severity_level = 3  # hide ~100 harmless "can't constant fold" warnings at load
        self.session = ort.InferenceSession(str(Path(model_dir) / "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.threshold = threshold

    def _entailment(self, premise: str, hypothesis: str) -> float:
        enc = self.tokenizer(premise, hypothesis, return_tensors="np",
                             truncation="only_first", max_length=MAX_TOKENS)
        logits = self.session.run(None, {"input_ids": enc["input_ids"],
                                         "attention_mask": enc["attention_mask"]})[0][0]
        probs = np.exp(logits - logits.max())
        return float(probs[self.entail] / probs.sum())

    def score(self, sentence: str, passages: list[str]) -> float:
        """Entailment probability of the sentence given the context.

        Each passage on its own first. Joining them first (the earlier order)
        dropped a sentence copied from the context: "AKINCI-5'in en yüksek hızı
        saniyede 2,5 metredir." scored 0.994 against the AKINCI-5 passage alone
        but 0.008 against it joined with a passage listing other robots'
        speeds; the model reads the other numbers as a contradiction. The
        joined context is only tried when no single passage supports the
        sentence (a fact combined from two passages)."""
        best = max(self._entailment(p, sentence) for p in passages)
        if best >= self.threshold or len(passages) == 1:
            return best
        joined = "\n".join(passages)
        if len(self.tokenizer(joined, sentence)["input_ids"]) <= MAX_TOKENS:
            best = max(best, self._entailment(joined, sentence))
        return best

    @staticmethod
    def _unknown_numbers(sentence: str, passages: list[str]) -> list[str]:
        """Standalone numbers in the sentence that the context does not contain.

        The NLI model judges the main claim and misses an added number:
        "Ada-7'in boyu 166 santimetre veya 1,66 metre olup, bu 5,43 metreye
        eşittir." scored 0.97, although 5,43 is invented and 1,66 a conversion
        the LLM did itself. An answer only states numbers its sources state.
        "2,5" and "2.5" count as the same number."""
        numbers = lambda t: set(re.findall(r"(?<![\w-])\d+(?:[.,]\d+)*", t.replace(",", ".")))
        known = numbers(" ".join(passages))
        return sorted(n for n in numbers(sentence) if n not in known)

    @staticmethod
    def _copied(sentence: str, passages: list[str], min_share: float = 0.85) -> bool:
        """True if the sentence is (almost) words taken from the context.

        The NLI model gave 0.05 to a sentence copied word for word from the
        context ("Ada-7 farklı sesleri algılayabilir: müzik, hapşırma, ...")
        because of the colon/list form, and 0.87 to the same fact as a plain
        sentence. A sentence whose words nearly all appear in the context needs
        no NLI. Never for sentences with numbers: "Mini Ada'nın bataryası 24 saat
        dayanır" has every word in the context but pairs the numbers wrongly
        (NLI caught it: contradiction 0.96). Invented claims contain a word
        that is not in the context ("yüzebilir", "uçabilir") and go to NLI.
        """
        # A standalone number ("24 saat", "2,5"), not a digit inside a name ("Ada-7", "V3").
        if re.search(r"(?<![\w-])\d", sentence):
            return False
        words = lambda t: set(re.findall(r"\w{3,}", t.lower()))
        claim, context = words(sentence), words(" ".join(passages))
        return bool(claim) and len(claim & context) / len(claim) >= min_share

    def supported_prefix(self, sentence: str, passages: list[str]) -> str | None:
        """The longest leading part of a rejected sentence that is supported.

        The model sometimes appends an invented clause to a correct fact,
        e.g. a unit conversion: "Ada-7'in boyu 166 santimetre veya 1,66 metre
        olur." (the prompt forbids converting; a 3B model still does). The
        sentence is split at clause boundaries and the longest prefix that
        passes the same checks is kept: "Ada-7'in boyu 166 santimetre."."""
        clauses = _CLAUSE.split(sentence.strip().rstrip(".!?"))
        for k in range(len(clauses) - 1, 0, -1):
            prefix = " ".join(c for c in clauses[:k]).strip(" ,;")
            if len(prefix.split()) >= 3 and self.supported(prefix, passages):
                return prefix + "."
        return None

    def supported(self, sentence: str, passages: list[str]) -> bool:
        unknown = self._unknown_numbers(sentence, passages)
        if unknown:
            log.info("Dropped sentence with numbers not in the context %s: %r", unknown, sentence)
            return False
        if self._copied(sentence, passages):
            return True
        score = self.score(sentence, passages)
        if score < self.threshold:
            log.info("Dropped unsupported sentence (%.2f): %r", score, sentence)
        return score >= self.threshold


def verified_sentences(tokens: Iterator[str], result, verifier: Verifier | None,
                       language: str) -> Iterator[str]:
    """LLM token stream -> speakable sentences that the context supports.

    Used by the voice pipeline and by scripts/eval_agent.py, so the evaluation
    measures exactly what is spoken. Answers without context (small talk,
    fixed replies) are not checked. If every sentence is dropped, the fixed
    "no information" sentence is said instead.
    """
    from app.agent.prompts import NO_ANSWER
    from app.text import SentenceSplitter

    splitter, spoken, dropped = SentenceSplitter(), [], []
    # Long sentences are cut at commas for latency. A later piece ("bağırma ve
    # sessizlik gibi.") has no subject of its own, so each piece is checked
    # together with the earlier pieces of the same sentence.
    sentence_so_far = ""

    def check(pieces):
        nonlocal sentence_so_far
        for s in pieces:
            first_piece = not sentence_so_far
            claim = f"{sentence_so_far} {s}".strip()
            sentence_so_far = "" if s.rstrip().endswith((".", "!", "?")) else claim
            if _FOREIGN_SCRIPT.search(s):  # never speak another language's script
                log.info("Dropped sentence in another script: %r", s)
                dropped.append(s)
                continue
            if verifier is None or not result.passages or verifier.supported(claim, result.passages):
                spoken.append(s)
                yield s
                continue
            dropped.append(s)
            # Keep the supported start of a sentence whose ending was invented.
            if first_piece and (kept := verifier.supported_prefix(s, result.passages)):
                log.info("Kept supported part: %r", kept)
                spoken.append(kept)
                yield kept

    for token in tokens:
        yield from check(splitter.push(token))
    yield from check(splitter.flush())
    if dropped and not spoken:
        spoken.append(NO_ANSWER[language])
        yield spoken[-1]
    result.spoken = " ".join(spoken)
    result.dropped = dropped
