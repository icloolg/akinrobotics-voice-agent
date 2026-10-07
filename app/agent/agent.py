"""The agent: question in -> grounded answer out (as a text stream).

Flow:
  1. Embed the question once; search the knowledge base with it.
  2. Router: small talk ("Sen kimsin?") -> short conversational reply,
     no context and no facts allowed. Tool question ("Saat kaç?") -> run the
     tool, its output becomes the CONTEXT (same grounded prompt as RAG).
  3. If the best match is below min_score -> fixed "no information" answer,
     the LLM is never called (cheapest and safest anti-hallucination step).
  4. Otherwise build [system prompt, short history, context + question]
     and stream the LLM's answer.
"""
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

from app.agent import prompts
from app.agent.retriever import Chunk, Retriever
from app.agent.router import Router
from app.agent.topics import TopicTracker
from app.providers.base import LLMProvider
from app.tools.base import Tool, ToolNotApplicable, ToolUnavailable

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Filled in while the answer streams; read it after the stream ends."""
    chunks: list[Chunk] = field(default_factory=list)
    route: str = ""        # "rag" | "chitchat" | "tool" | "no_answer"
    tool: str = ""         # name of the tool used, if any
    grounded: bool = False
    answer: str = ""
    retrieval_ms: float = 0.0
    question: str = ""     # set when a follow-up was rewritten with the current topic


class Agent:
    def __init__(self, llm: LLMProvider, retriever: Retriever, router: Router,
                 tools: dict[str, Tool] | None = None,
                 top_k: int = 3, min_score: float = 0.8, history_turns: int = 3,
                 topics: TopicTracker | None = None, followup_max_score: float = 0.85):
        self.llm = llm
        self.retriever = retriever
        self.router = router
        self.tools = tools or {}
        self.topics = topics
        self.followup_max_score = followup_max_score
        self.top_k = top_k
        self.min_score = min_score
        self.history_turns = history_turns
        self.history: list[dict] = []

    def answer(self, question: str, language: str, result: AgentResult) -> Iterator[str]:
        start = time.perf_counter()
        vector = self.retriever.encode_query(question)
        result.chunks = self.retriever.search(question, self.top_k, vector=vector)
        best = result.chunks[0].score if result.chunks else 0.0
        intent = self.router.route(question, vector, best)

        # Follow-up without a name ("Ne zaman kurulmuş?"): add the current topic and
        # search again. Only when the question matches nothing well on its own
        # (measured: follow-ups 0.785-0.823, self-contained questions 0.869-0.890)
        # and is not small talk or a tool question ("Saat kaç?" is not about Ada-7).
        # Without these two conditions tool accuracy fell from 10/11 to 2/11.
        if self.topics is not None:
            followup = self.topics.contextualize(question)
            if (followup != question and intent is None
                    and best < self.followup_max_score):
                log.info("Follow-up: %r -> %r", question, followup)
                question = result.question = followup
                vector = self.retriever.encode_query(question)
                result.chunks = self.retriever.search(question, self.top_k, vector=vector)
                best = result.chunks[0].score if result.chunks else 0.0
                intent = self.router.route(question, vector, best)
        result.retrieval_ms = round((time.perf_counter() - start) * 1000, 1)
        log.info("Retrieval best=%.3f intent=%s sources=%s",
                 best, intent, [c.source for c in result.chunks])

        context = None
        if intent in self.tools:
            try:
                context = self.tools[intent].run(question, language)
            except ToolNotApplicable:
                log.info("Tool %s not applicable -> RAG", intent)
                intent = None
            except ToolUnavailable as e:
                result.route, result.tool = "tool", intent
                result.answer = str(e)  # exact message, not paraphrased by the LLM
                yield result.answer
                return

        if context is not None:
            result.route, result.tool, result.grounded = "tool", intent, True
            log.info("Tool %s -> %s", intent, context.replace("\n", " | "))
            messages = [
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.user_message(question, context, language)},
            ]
            # Not added to history: live data is outdated by the next turn.
            yield from self._stream(messages, result)
            return

        if intent == "chitchat":
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
        # Not history[-2 * n:]: for n = 0 that is history[0:], the WHOLE list (a bug
        # that kept unlimited history when history_turns was 0).
        self.history = self.history[max(0, len(self.history) - 2 * self.history_turns):]

    def _stream(self, messages: list[dict], result: AgentResult) -> Iterator[str]:
        parts = []
        for piece in self.llm.stream(messages):
            parts.append(piece)
            yield piece
        result.answer = "".join(parts).strip()

    def reset(self) -> None:
        self.history.clear()
        if self.topics is not None:
            self.topics.reset()

    def fork(self) -> "Agent":
        """A new conversation: same models, router and tools, its own empty history
        and topic. The server makes one per client so two users never share them."""
        other = copy.copy(self)
        other.history = []
        if self.topics is not None:
            other.topics = TopicTracker(self.topics.topics)
        return other
