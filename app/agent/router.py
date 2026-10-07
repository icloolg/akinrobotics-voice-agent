"""Semantic intent routing: is this small talk or a knowledge question?

Each intent has a few example sentences. The user's question is embedded
with the same model the retriever uses and compared to every example.
Small talk is answered conversationally (no facts); everything else goes
through RAG.

Two conditions, both measured on test questions (see NOTLAR.md):
  - chitchat score >= min_score          ("Sen kimsin?" 0.979, off-topic max 0.878)
  - chitchat score - knowledge score >= margin
    ("Robotlarınız neler yapabilir?" looks like "neler yapabilirsin" (0.916),
     but its knowledge score is close (0.863), so it must stay on the RAG path.)

Adding a new intent = adding a new list of examples.
"""
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


class Router:
    def __init__(self, encoder, min_score: float = 0.90, margin: float = 0.10):
        """encoder: function list[str] -> normalized vectors (the retriever's query encoder)."""
        self.min_score = min_score
        self.margin = margin
        self.examples = np.array(encoder(CHITCHAT_EXAMPLES))

    def chitchat_score(self, query_vector) -> float:
        # Vectors are normalized, so the dot product is the cosine similarity.
        return float(np.max(self.examples @ np.array(query_vector)))

    def is_chitchat(self, query_vector, best_knowledge_score: float) -> bool:
        score = self.chitchat_score(query_vector)
        return score >= self.min_score and score - best_knowledge_score >= self.margin
