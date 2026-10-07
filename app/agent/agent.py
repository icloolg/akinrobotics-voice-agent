"""The agent: question in -> grounded answer out (as a text stream).

Flow:
  1. Embed the question once; search the knowledge base with it.
  2. Router: small talk ("Sen kimsin?") -> short conversational reply,
     no context and no facts allowed.
  3. If the best match is below min_score -> fixed "no information" answer,
     the LLM is never called (cheapest and safest anti-hallucination step).
  4. Otherwise build [system prompt, short history, context + question]
     and stream the LLM's answer.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

from app.agent import prompts
from app.agent.retriever import Chunk, Retriever
from app.agent.router import Router
from app.providers.base import LLMProvider

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Filled in while the answer streams; read it after the stream ends."""
    chunks: list[Chunk] = field(default_factory=list)
    route: str = ""        # "rag" | "chitchat" | "no_answer"
    grounded: bool = False
    answer: str = ""
    retrieval_ms: float = 0.0


class Agent:
    def __init__(self, llm: LLMProvider, retriever: Retriever, router: Router,
                 top_k: int = 3, min_score: float = 0.8, history_turns: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.router = router
        self.top_k = top_k
        self.min_score = min_score
        self.history_turns = history_turns
        self.history: list[dict] = []

    def answer(self, question: str, language: str, result: AgentResult) -> Iterator[str]:
        start = time.perf_counter()
        vector = self.retriever.encode_query(question)
        result.chunks = self.retriever.search(question, self.top_k, vector=vector)
        best = result.chunks[0].score if result.chunks else 0.0
        chitchat = self.router.is_chitchat(vector, best)
        result.retrieval_ms = round((time.perf_counter() - start) * 1000, 1)
        log.info("Retrieval best=%.3f chitchat=%s sources=%s",
                 best, chitchat, [c.source for c in result.chunks])

        if chitchat:
            result.route = "chitchat"
            # No history and no context: small talk must not turn into facts.
            messages = [
                {"role": "system", "content": prompts.CHITCHAT_SYSTEM[language]},
                {"role": "user", "content": prompts.chitchat_message(question, language)},
            ]
            yield from self._stream(messages, result)
            return

        if best < self.min_score:
            result.route = "no_answer"
            result.answer = prompts.NO_ANSWER[language]
            yield result.answer
            return

        result.route = "rag"
        result.grounded = True
        relevant = [c for c in result.chunks if c.score >= self.min_score]
        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            *self.history,
            {"role": "user", "content": prompts.user_message(
                question, prompts.build_context(relevant), language)},
        ]
        yield from self._stream(messages, result)

        # History stores the plain question (not the context) to keep prompts short.
        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": result.answer}]
        self.history = self.history[-2 * self.history_turns:]

    def _stream(self, messages: list[dict], result: AgentResult) -> Iterator[str]:
        parts = []
        for piece in self.llm.stream(messages):
            parts.append(piece)
            yield piece
        result.answer = "".join(parts).strip()

    def reset(self) -> None:
        self.history.clear()
