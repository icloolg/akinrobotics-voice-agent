"""Tools: live information sources the agent can use (clock, APIs, databases).

How a tool is chosen: the router compares the question to the tool's
`examples` (semantic similarity, same embedding model as RAG). No LLM
function-calling is needed, which a 3B model does unreliably and which
would cost an extra LLM call (~1-2 s on this GPU).

How a tool answers: run() returns plain-text facts. They are given to the
LLM as CONTEXT with the same "use only the context" rules as RAG, so tool
answers are grounded too. If the tool fails it raises ToolUnavailable and
the agent says that message as is, instead of guessing.

Adding a tool = subclass Tool + register it in app/factory.py + enable it
in config.yaml.
"""
from abc import ABC, abstractmethod


class ToolUnavailable(Exception):
    """Raised by a tool that cannot answer (API down, ...). The message is
    spoken to the user directly, without the LLM: measured, the LLM turned
    "servise ulaşılamıyor" into the wrong "servis altında durumda"."""


class ToolNotApplicable(Exception):
    """Raised by a tool that was routed to but cannot use this question (e.g. the
    specs database has no column for "En akıllı robot hangisi?"). The agent then
    falls back to RAG. Measured: before this, the specs tool sent its whole table
    as context for that question and the first audio took 8.5 s."""


class Tool(ABC):
    name: str
    examples: list[str]  # questions (tr + en) that should trigger this tool
    # Optional: at least one of these must appear in the question (hybrid
    # routing). Used when meaning alone cannot separate the tool from
    # knowledge questions, e.g. "şu an şarjı kaç" (live) vs "kaç saat şarj olur" (spec).
    keywords: list[str] = []

    @abstractmethod
    def run(self, question: str, language: str) -> str:
        """Return the facts needed to answer, as short plain text."""
