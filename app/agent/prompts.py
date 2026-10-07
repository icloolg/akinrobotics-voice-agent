"""All text the agent sends to the LLM or speaks without the LLM.

Design: ONE system prompt for every language, and the language instruction
at the very end of the user message.
  - Ollama caches the processed prompt prefix. An identical system prompt is
    processed once and reused on every turn (prompt processing is the main
    latency cost on the GTX 1650: ~200 tokens/s).
  - Small models follow the *last* instruction best, so "answer in English"
    right after the question fixes Turkish answers to English questions.
"""

SYSTEM = (
    "You are a voice assistant for AKINROBOTICS. Your answers are spoken aloud.\n"
    "Rules:\n"
    "1. Use ONLY the information in the CONTEXT. Never use outside knowledge.\n"
    "2. If the CONTEXT does not contain the answer, say you don't have that information.\n"
    "3. Be brief: at most 3 sentences.\n"
    "4. No lists, tables, emoji or markdown; plain spoken language only."
)

ANSWER_IN = {
    "tr": "Cevabı Türkçe ver.",
    "en": "Answer in English.",
}

# Spoken directly when retrieval finds nothing relevant (LLM is not called).
NO_ANSWER = {
    "tr": "Bu konuda bilgi kaynaklarımda bir bilgi bulamadım.",
    "en": "I couldn't find information about that in my knowledge sources.",
}


def build_context(chunks) -> str:
    return "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)


def user_message(question: str, context: str, language: str) -> str:
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\n{ANSWER_IN[language]}"
