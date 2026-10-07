"""All text the agent sends to the LLM or speaks without the LLM."""

SYSTEM = {
    "tr": (
        "Sen bir sesli asistansın. Cevapların sesli okunacak.\n"
        "Kurallar:\n"
        "1. SADECE aşağıdaki BAĞLAM içindeki bilgileri kullan.\n"
        "2. Cevap bağlamda yoksa uydurma, 'Bu konuda bilgim yok.' de.\n"
        "3. Türkçe cevap ver. Kısa ol: en fazla 3 cümle.\n"
        "4. Madde işareti, tablo, emoji veya markdown kullanma; düz konuşma dili kullan."
    ),
    "en": (
        "You are a voice assistant. Your answers will be spoken aloud.\n"
        "Rules:\n"
        "1. Use ONLY the information in the CONTEXT below.\n"
        "2. If the answer is not in the context, do not guess; say 'I don't have information about that.'\n"
        "3. Answer in English. Be brief: at most 3 sentences.\n"
        "4. No bullet points, tables, emoji or markdown; use plain spoken language."
    ),
}

# Spoken directly when retrieval finds nothing relevant (LLM is not called).
NO_ANSWER = {
    "tr": "Bu konuda bilgi kaynaklarımda bir bilgi bulamadım.",
    "en": "I couldn't find information about that in my knowledge sources.",
}


def build_context(chunks) -> str:
    return "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)


def user_message(question: str, context: str, language: str) -> str:
    label = "BAĞLAM" if language == "tr" else "CONTEXT"
    q_label = "SORU" if language == "tr" else "QUESTION"
    return f"{label}:\n{context}\n\n{q_label}: {question}"
