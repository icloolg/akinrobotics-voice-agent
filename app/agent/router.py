"""Semantic intent routing: small talk, a tool, or a knowledge (RAG) question?

Each intent has a few example sentences. The user's question is embedded
with the same model the retriever uses and compared to every example.

An intent wins when, for its best-matching example:
  - score >= min_score, and
  - score - best knowledge-chunk score >= margin, and
  - (if the intent has keywords) one of them appears in the question, and
  - none of its `exclude` words appears in it.
If several intents pass, the highest score wins. If none pass -> RAG.

Why the margin (measured, see NOTLAR.md): "Robotlarınız neler yapabilir?"
looks like the small-talk example "neler yapabilirsin" (0.916) but is a
knowledge question (best chunk 0.863), so it must stay on the RAG path.

The small-talk examples live in app/agent/smalltalk.py (one list for both
"is it small talk?" and "which kind?").
"""
import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Intent:
    name: str
    examples: np.ndarray  # normalized example vectors
    min_score: float
    margin: float
    keywords: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


# A question or a request: question mark, question particle (mı/mi/mu/mü),
# question word, or a request verb ("anlat", "bilgi ver", "tell"). Measured on
# the 62 development questions: all match; on 12 unseen reactions ("Süper!",
# "Tamam.", "Bravo!", "Nice."): none match. Embedding scores could not separate
# them (reactions scored 0.80-0.83 against the knowledge base, like follow-ups).
_REQUEST = re.compile(
    r"\?|kaç|\b(?:m[ıiuü](?:s[ıiuü]n(?:[ıiuü]z)?|y[ıiuü]m)?|ne|neler|neden|nedir|nasıl|hangi\w*|nere\w*|kim\w*|niçin"
    r"|what|which|who|where|when|why|how|is|are|does|do|can|tell|describe|explain|list"
    r"|anlat\w*|söyle\w*|açıkla\w*|bilgi|listele\w*|tanıt\w*)\b", re.IGNORECASE)


def is_request(text: str) -> bool:
    """True if the text asks something; False for remarks such as "Süper!"."""
    return bool(_REQUEST.search(_lower(text)))


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
            keywords: list[str] | None = None, exclude: list[str] | None = None) -> None:
        self.intents.append(Intent(name, np.array(self.encoder(examples)), min_score, margin,
                                   [_lower(k) for k in keywords or []],
                                   [_lower(k) for k in exclude or []]))

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
                   and (not i.keywords or any(k in text for k in i.keywords))
                   and not any(k in text for k in i.exclude)]
        if not passing:
            return None
        return max(passing, key=lambda i: scores[i.name]).name
