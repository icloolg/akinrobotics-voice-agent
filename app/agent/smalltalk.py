"""Small talk by similarity: category of the sentence -> ready-made reply.

    "Sağ ol" / "Eyvallah" / "Thanks a lot"  -> thanks -> "Rica ederim, ..."

Each category has a few example sentences. A new sentence goes to the
category of its most similar example (same embedding model as RAG), so
phrasings that are not in the list still work. The reply is a fixed
sentence in the user's language: no LLM call.

Why not let the LLM reply (measured, see NOTLAR.md): with one example in the
prompt the 3B model answered "Teşekkürler" and "Görüşürüz" with "İyiyim,
teşekkür ederim"; even with an example per kind it answered "Sağ ol" that
way and echoed the user ("Teşekkür ederim, harikasın"). A sentence that is
small talk but close to no category (< min_score) still goes to the LLM.

Measured on 24 phrasings that are NOT in the examples ("Eyvallah", "Süpersin",
"You rock", "Kimsin sen?", ...): 22/24 nearest category correct. Correct ones
scored 0.869-0.968; min_score 0.88 -> 19 template replies, 4 to the LLM,
1 wrong ("Kendine iyi bak" -> identity).

Adding a kind of small talk = adding one entry to CATEGORIES.
"""
import numpy as np

CATEGORIES = {
    "greeting": {
        "examples": ["merhaba", "selam", "günaydın", "iyi akşamlar", "hello", "hi there", "good morning"],
        "tr": "Merhaba, size nasıl yardımcı olabilirim?",
        "en": "Hello, how can I help you?",
    },
    "how_are_you": {
        "examples": ["nasılsın", "naber", "merhaba nasılsın", "how are you", "how is it going"],
        "tr": "İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?",
        "en": "I'm fine, thank you. How can I help you?",
    },
    "thanks": {
        "examples": ["teşekkür ederim", "teşekkürler", "sağ ol", "çok teşekkürler", "thank you", "thanks"],
        "tr": "Rica ederim. Başka bir sorunuz var mı?",
        "en": "You're welcome. Anything else?",
    },
    "compliment": {
        "examples": ["harikasın", "çok iyisin", "çok güzel cevap verdin", "aferin sana",
                     "you are great", "good job", "well done"],
        "tr": "Teşekkür ederim, yardımcı olabildiysem ne mutlu.",
        "en": "Thank you, glad I could help.",
    },
    "goodbye": {
        "examples": ["görüşürüz", "hoşça kal", "güle güle", "goodbye", "see you later"],
        "tr": "Görüşmek üzere, iyi günler.",
        "en": "Goodbye, have a nice day.",
    },
    "identity": {
        "examples": ["sen kimsin", "adın ne", "kendini tanıt", "neler yapabilirsin", "sana ne sorabilirim",
                     "who are you", "introduce yourself", "what is your name", "what can you do", "what can I ask you"],
        "tr": "Ben AkınVoice. AKINROBOTICS robotları, AKINSOFT yazılımları ve iki şirket hakkındaki "
              "sorularınızı Türkçe ya da İngilizce cevaplayabilirim.",
        "en": "I'm AkınVoice. I can answer your questions about AKINROBOTICS robots, AKINSOFT software "
              "and the two companies, in Turkish or English.",
    },
}

# All examples: the router uses them to decide "is this small talk at all?".
ALL_EXAMPLES = [e for c in CATEGORIES.values() for e in c["examples"]]

# A sentence asking for an amount or a measure is never small talk, however
# similar it sounds: "How fast is it?" scored 0.886 against "how is it going".
NOT_SMALLTALK = [" kaç", "ne kadar", "how fast", "how much", "how many", "how long",
                 "how tall", "how big", "how heavy", "how far"]


class SmallTalk:
    def __init__(self, encoder, min_score: float = 0.88):
        """encoder: function list[str] -> normalized vectors (the retriever's query encoder)."""
        self.min_score = min_score
        self.labels = [name for name, c in CATEGORIES.items() for _ in c["examples"]]
        self.examples = np.array(encoder(ALL_EXAMPLES))

    def classify(self, query_vector) -> tuple[str, float]:
        """(category of the most similar example, its similarity)."""
        scores = self.examples @ np.array(query_vector)
        i = int(np.argmax(scores))
        return self.labels[i], float(scores[i])

    def reply(self, query_vector, language: str) -> str | None:
        """Ready-made reply, or None if no category is close enough (then the LLM answers)."""
        category, score = self.classify(query_vector)
        if score < self.min_score:
            return None
        return CATEGORIES[category]["tr" if language == "tr" else "en"]
