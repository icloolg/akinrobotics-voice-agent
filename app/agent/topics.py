"""Follow-up questions: remember what the conversation is about.

    User: "AKINSOFT nedir?"         -> topic = AKINSOFT
    User: "Ne zaman kurulmuş?"      -> searched and answered as
                                       "AKINSOFT: Ne zaman kurulmuş?"

Each question is searched on its own, so a follow-up without a name found the
wrong document or none (measured: 1/4 follow-ups correct). The usual fix is to
let the LLM rewrite the question, but that is one more LLM call (~1-2 s on the
GTX 1650). Here the known names (robots, products, companies) are matched in
the text instead: if a question names none of them, the last named one is
put in front of it. No extra model, ~0 ms.
"""
import re


def _fold(text: str) -> str:
    """Case- and Turkish-i-insensitive form: "AKINCI-5" == "Akıncı 5"."""
    return text.lower().replace("ı", "i").replace("i̇", "i")


class TopicTracker:
    def __init__(self, topics: list[str]):
        # Longest names first: "Servis Robotu V3" before a shorter name it contains.
        self.topics = sorted(topics, key=len, reverse=True)
        self.patterns = [
            # Spaces and hyphens are optional between parts: "Ada-7", "Ada 7", "ada7".
            (name, re.compile(r"[\s-]?".join(re.escape(part) for part in re.split(r"[\s-]+", _fold(name)))))
            for name in self.topics
        ]
        self.current: str | None = None
        self.question_named = False  # the last question named a topic itself

    def find_all(self, text: str) -> list[str]:
        """Every known name in the text, longest names first."""
        folded = _fold(text)
        return [name for name, pattern in self.patterns if pattern.search(folded)]

    def find(self, text: str) -> str | None:
        """The first known name in the text, or None."""
        names = self.find_all(text)
        return names[0] if names else None

    def contextualize(self, question: str) -> str:
        """The question to search and answer with: unchanged if it names a topic
        itself (and that becomes the current topic), otherwise prefixed with the
        current topic."""
        named = self.find(question)
        self.question_named = named is not None
        if named:
            self.current = named
            return question
        return f"{self.current}: {question}" if self.current else question

    def observe_answer(self, answer: str) -> None:
        """A question that names nothing is answered with a name ("En hızlı robot
        hangisi?" -> "AKINCI-5 ..."): the next follow-up ("Ne kadar hızlı?") refers
        to that name. Only when the answer names exactly one topic; a name the
        user said in the question always takes precedence."""
        if self.question_named:
            return
        names = self.find_all(answer)
        if len(names) == 1:
            self.current = names[0]

    def reset(self) -> None:
        self.current = None
        self.question_named = False
