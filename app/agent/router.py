"""Semantic intent routing: small talk, a tool, or a knowledge (RAG) question?

Each intent has a few example sentences. The user's question is embedded
with the same model the retriever uses and compared to every example.

An intent wins when, for its best-matching example:
  - score >= min_score, and
  - score - best knowledge-chunk score >= margin, and
  - (if the intent has keywords) one of them appears in the question.
If several intents pass, the highest score wins. If none pass -> RAG.

Why the margin (measured, see NOTLAR.md): "Robotlarınız neler yapabilir?"
looks like the small-talk example "neler yapabilirsin" (0.916) but is a
knowledge question (best chunk 0.863), so it must stay on the RAG path.
"""
import re
from dataclasses import dataclass, field

import numpy as np

CHITCHAT_EXAMPLES = [
    # Turkish
    "merhaba", "selam", "günaydın", "iyi akşamlar", "nasılsın", "naber",
    "sen kimsin", "adın ne", "kendini tanıt", "neler yapabilirsin", "sana ne sorabilirim",
    "teşekkür ederim", "sağ ol", "görüşürüz", "hoşça kal",
    # English
    "hello", "hi there", "good morning", "how are you", "who are you",
    "what is your name", "introduce yourself", "what can you do", "what can I ask you",
    "thank you", "thanks", "goodbye", "see you later",
]


@dataclass
class Intent:
    name: str
    examples: np.ndarray  # normalized example vectors
    min_score: float
    margin: float
    keywords: list[str] = field(default_factory=list)


def _lower(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").lower()


def _words(text: str) -> str:
    """Lowercase, punctuation -> space, padded: keyword " en " then matches the
    word "en" but not "neden", and "fastest?" still contains "est "."""
    return " " + re.sub(r"[^\w\s'-]", " ", _lower(text)) + " "


class Router:
    def __init__(self, encoder):
        """encoder: function list[str] -> normalized vectors (the retriever's query encoder)."""
        self.encoder = encoder
        self.intents: list[Intent] = []

    def add(self, name: str, examples: list[str], min_score: float, margin: float,
            keywords: list[str] | None = None) -> None:
        self.intents.append(Intent(name, np.array(self.encoder(examples)), min_score, margin,
                                   [_lower(k) for k in keywords or []]))

    def scores(self, query_vector) -> dict[str, float]:
        # Vectors are normalized, so the dot product is the cosine similarity.
        q = np.array(query_vector)
        return {i.name: float(np.max(i.examples @ q)) for i in self.intents}

    def route(self, question: str, query_vector, best_knowledge_score: float) -> str | None:
        """Name of the winning intent, or None for the RAG path."""
        scores = self.scores(query_vector)
        text = _words(question)
        passing = [i for i in self.intents
                   if scores[i.name] >= i.min_score
                   and scores[i.name] - best_knowledge_score >= i.margin
                   and (not i.keywords or any(k in text for k in i.keywords))]
        if not passing:
            return None
        return max(passing, key=lambda i: scores[i.name]).name
