"""The agent: question in -> grounded answer out (as a text stream).

Flow:
  1. Search the knowledge base.
  2. If the best match is below min_score -> fixed "no information" answer,
     the LLM is never called (cheapest and safest anti-hallucination step).
  3. Otherwise build [system prompt, short history, context + question]
     and stream the LLM's answer.
"""
import logging
from dataclasses import dataclass, field
from typing import Iterator

from app.agent import prompts
from app.agent.retriever import Chunk, Retriever
from app.providers.base import LLMProvider

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Filled in while the answer streams; read it after the stream ends."""
    chunks: list[Chunk] = field(default_factory=list)
    grounded: bool = False
    answer: str = ""


class Agent:
    def __init__(self, llm: LLMProvider, retriever: Retriever,
                 top_k: int = 3, min_score: float = 0.8, history_turns: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k
        self.min_score = min_score
        self.history_turns = history_turns
        self.history: list[dict] = []

    def answer(self, question: str, language: str, result: AgentResult) -> Iterator[str]:
        result.chunks = self.retriever.search(question, self.top_k)
        best = result.chunks[0].score if result.chunks else 0.0
        log.info("Retrieval best=%.3f sources=%s", best, [c.source for c in result.chunks])

        if best < self.min_score:
            result.grounded = False
            result.answer = prompts.NO_ANSWER[language]
            yield result.answer
            return

        result.grounded = True
        relevant = [c for c in result.chunks if c.score >= self.min_score]
        messages = [
            {"role": "system", "content": prompts.SYSTEM[language]},
            *self.history,
            {"role": "user", "content": prompts.user_message(
                question, prompts.build_context(relevant), language)},
        ]

        parts = []
        for piece in self.llm.stream(messages):
            parts.append(piece)
            yield piece
        result.answer = "".join(parts).strip()

        # History stores the plain question (not the context) to keep prompts short.
        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": result.answer}]
        self.history = self.history[-2 * self.history_turns:]

    def reset(self) -> None:
        self.history.clear()
