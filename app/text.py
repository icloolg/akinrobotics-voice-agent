"""Turns the LLM's token stream into speakable sentences."""
import re

# A sentence ends at . ! ? followed by whitespace, or at a line break (lists).
# Requiring the whitespace keeps decimals like "2.5" intact.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
# Where a too-long sentence may be cut: after a comma, semicolon or colon.
_SOFT_BREAK = re.compile(r"[,;:]\s+")
_MARKDOWN = re.compile(r"[*#_`>|]")
_LIST_MARKER = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)


class SentenceSplitter:
    """Feed tokens with push(); get back every piece that is ready to speak.

    Example:  push("Ada-7 65 ")      -> []
              push("kg. Boyu ")      -> ["Ada-7 65 kg."]
              flush()                -> ["Boyu ..."]

    A sentence longer than `max_chars` is cut at its last comma so the first
    audio does not wait for a very long sentence (measured: one 250-char
    sentence delayed the first audio to 7 s).
    """

    def __init__(self, min_chars: int = 12, max_chars: int = 100):
        self.buffer = ""
        self.carry = ""  # finished but too-short pieces waiting for the next sentence
        self.min_chars = min_chars  # avoid sending tiny fragments like "Evet." alone
        self.max_chars = max_chars

    def push(self, token: str) -> list[str]:
        self.buffer += token
        parts = _SENTENCE_END.split(self.buffer)
        complete, self.buffer = parts[:-1], parts[-1]
        out = self._merge_short([p for p in complete if p.strip()])
        if len(self.buffer) > self.max_chars:
            out += self._cut_long()
        return out

    def flush(self) -> list[str]:
        rest = self._join(self.carry, clean_for_speech(self.buffer))
        self.buffer = self.carry = ""
        return [rest] if rest else []

    def _cut_long(self) -> list[str]:
        breaks = [m.end() for m in _SOFT_BREAK.finditer(self.buffer) if m.start() >= self.min_chars]
        if not breaks:
            return []
        head, self.buffer = self.buffer[:breaks[-1]], self.buffer[breaks[-1]:]
        head, self.carry = self._join(self.carry, clean_for_speech(head)), ""
        return [head]

    def _merge_short(self, sentences: list[str]) -> list[str]:
        out = []
        for s in map(clean_for_speech, sentences):
            self.carry = self._join(self.carry, s)
            if len(self.carry) >= self.min_chars:
                out.append(self.carry)
                self.carry = ""
        return out

    @staticmethod
    def _join(a: str, b: str) -> str:
        """Short list items are joined with a comma so they are read as a list."""
        if not a or not b:
            return a or b
        return f"{a}{' ' if a[-1] in '.!?:,;' else ', '}{b}"


def clean_for_speech(text: str) -> str:
    """Remove list markers and markdown symbols the TTS would otherwise read out."""
    return _MARKDOWN.sub("", _LIST_MARKER.sub("", text)).strip()
