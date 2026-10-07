"""Turns the LLM's token stream into speakable sentences."""
import re

# A sentence ends at . ! ? followed by whitespace. Requiring the whitespace
# keeps decimals like "2.5" and names like "Ada-7." mid-stream intact.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN = re.compile(r"[*#_`>|]")


class SentenceSplitter:
    """Feed tokens with push(); get back every sentence that is complete.

    Example:  push("Ada-7 65 ")      -> []
              push("kg. Boyu ")      -> ["Ada-7 65 kg."]
              flush()                -> ["Boyu ..."]
    """

    def __init__(self, min_chars: int = 12):
        self.buffer = ""
        self.min_chars = min_chars  # avoid sending tiny fragments like "Evet." alone

    def push(self, token: str) -> list[str]:
        self.buffer += token
        parts = _SENTENCE_END.split(self.buffer)
        if len(parts) == 1:
            return []
        complete, self.buffer = parts[:-1], parts[-1]
        return self._merge_short(complete)

    def flush(self) -> list[str]:
        rest, self.buffer = self.buffer.strip(), ""
        return [clean_for_speech(rest)] if rest else []

    def _merge_short(self, sentences: list[str]) -> list[str]:
        out, carry = [], ""
        for s in sentences:
            carry = f"{carry} {s}".strip()
            if len(carry) >= self.min_chars:
                out.append(clean_for_speech(carry))
                carry = ""
        if carry:  # too short: keep it in front of the next sentence
            self.buffer = f"{carry} {self.buffer}"
        return out


def clean_for_speech(text: str) -> str:
    """Remove markdown symbols the TTS would otherwise read out."""
    return _MARKDOWN.sub("", text).strip()
