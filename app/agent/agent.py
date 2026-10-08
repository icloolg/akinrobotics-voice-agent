"""The agent: question in, grounded answer out (as a token stream).

One turn:
  1. Retrieve  embed the question once, search the knowledge base, route it
               (small talk / a tool / knowledge question).
  2. Resolve   a remark that asks nothing is acknowledged; a follow-up without
               a name is searched again with the conversation topic.
  3. Answer    by route:
                 tool       tool output is the context, same grounded prompt as RAG
                 chitchat   template reply, or the LLM without any context
                 no_answer  best match below rag.min_score: fixed reply, no LLM
                 rag        top-k chunks as context
"""
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

from app.agent import prompts
from app.agent.retriever import Chunk, Retriever, SearchResult
from app.agent.router import Router, is_request
from app.agent.smalltalk import SmallTalk
from app.agent.topics import TopicTracker
from app.providers.base import LLMProvider
from app.tools.base import Tool, ToolNotApplicable, ToolUnavailable

log = logging.getLogger(__name__)

# Conversational fillers that may also occur inside knowledge-base words (as 5-letter stems).
DISCOURSE_MARKERS = {"tamam", "peki", "evet", "hayır", "olur", "okay", "ok", "yes", "no", "sure"}


@dataclass
class AgentResult:
    """Filled in while the answer streams; read it after the stream ends."""
    chunks: list[Chunk] = field(default_factory=list)
    route: str = ""        # rag | tool | chitchat | no_answer
    tool: str = ""         # tool name when route == "tool"
    grounded: bool = False
    answer: str = ""       # the LLM's (or template's) full answer
    retrieval_ms: float = 0.0
    question: str = ""     # the rewritten question when a follow-up was resolved
    passages: list[str] = field(default_factory=list)  # context the answer must be supported by
    spoken: str = ""       # what was said after verification (verify.verified_sentences)
    dropped: list[str] = field(default_factory=list)   # sentences the verifier rejected


@dataclass
class _Retrieval:
    question: str
    vector: list[float]
    search: SearchResult
    intent: str | None


class Agent:
    def __init__(self, llm: LLMProvider, retriever: Retriever, router: Router,
                 tools: dict[str, Tool] | None = None,
                 top_k: int = 3, min_score: float = 0.8, history_turns: int = 0,
                 topics: TopicTracker | None = None, followup_max_score: float = 0.85,
                 smalltalk: SmallTalk | None = None, allow_combining: bool = False):
        self.llm = llm
        self.retriever = retriever
        self.router = router
        self.tools = tools or {}
        self.topics = topics
        self.followup_max_score = followup_max_score
        self.smalltalk = smalltalk
        # With sentence verification on, the prompt may let the LLM combine facts.
        self.allow_combining = allow_combining
        self.top_k = top_k
        self.min_score = min_score
        self.history_turns = history_turns
        self.history: list[dict] = []

    # ---- public API -----------------------------------------------------------
    def answer(self, question: str, language: str, result: AgentResult) -> Iterator[str]:
        start = time.perf_counter()
        r = self._retrieve(question)

        if self._is_remark(r):
            yield from self._reply(result, "chitchat", prompts.ACKNOWLEDGE[language])
            return

        r = self._resolve_followup(r, result)
        result.chunks = r.search.chunks
        result.retrieval_ms = round((time.perf_counter() - start) * 1000, 1)
        log.info("Retrieval best=%.3f intent=%s sources=%s",
                 r.search.best_score, r.intent, [c.source for c in r.search.chunks])

        if r.intent in self.tools:
            handled = yield from self._answer_tool(r, language, result)
            if handled:
                return
            r.intent = None  # the tool does not apply: answer from the documents
        if r.intent == "chitchat":
            yield from self._answer_chitchat(r, language, result)
        elif r.search.best_score < self.min_score:
            yield from self._reply(result, "no_answer", prompts.NO_ANSWER[language])
        else:
            yield from self._answer_rag(r, language, result)

    def set_topics(self, names: list[str]) -> None:
        """Replace the known topic names (admin panel); new conversations use them."""
        self.topics = TopicTracker(names)

    def reset(self) -> None:
        self.history.clear()
        if self.topics is not None:
            self.topics.reset()

    def fork(self) -> "Agent":
        """A new conversation: shared models, router and tools; its own history
        and topic. The server creates one per client."""
        other = copy.copy(self)
        other.history = []
        if self.topics is not None:
            other.topics = TopicTracker(self.topics.topics)
        return other

    # ---- steps ----------------------------------------------------------------
    def _retrieve(self, question: str) -> _Retrieval:
        vector = self.retriever.encode_query(question)
        search = self.retriever.search(question, self.top_k, vector=vector)
        return _Retrieval(question, vector, search, self.router.route(question, vector, search.best_score))

    def _is_remark(self, r: _Retrieval) -> bool:
        """A remark ("Süper!", "Tamam.") asks nothing, names nothing and uses no word
        of the knowledge base. It must not be searched or become a follow-up
        ("AKINCI-5: Süper!" returned AKINCI-5 facts). An elliptic follow-up whose
        question mark was lost in transcription ("Ekranı.") does use such a word.
        Measured: 13/14 reactions contain no knowledge-base word, 7/8 elliptic
        follow-ups do (the exceptions: "Tamam", a discourse marker, ignored below;
        "Fiyatı", which the knowledge base cannot answer anyway)."""
        return (r.intent is None and not is_request(r.question)
                and (self.topics is None or not self.topics.find(r.question))
                and not self.retriever.mentions_known_terms(r.question, DISCOURSE_MARKERS))

    def _resolve_followup(self, r: _Retrieval, result: AgentResult) -> _Retrieval:
        """A follow-up without a name ("Ne zaman kurulmuş?") is searched as
        "<topic>: <question>". Only for questions that match no document well on
        their own and are not small talk or tool questions (see NOTLAR.md: without
        these conditions tool routing fell from 10/11 to 2/11)."""
        if self.topics is None:
            return r
        followup = self.topics.contextualize(r.question)
        if followup == r.question or r.intent is not None or r.search.best_score >= self.followup_max_score:
            return r
        log.info("Follow-up: %r -> %r", r.question, followup)
        result.question = followup
        return self._retrieve(followup)

    # ---- answers --------------------------------------------------------------
    def _answer_tool(self, r: _Retrieval, language: str, result: AgentResult):
        """Generator; returns True if the turn was answered."""
        tool = self.tools[r.intent]
        try:
            context = tool.run(r.question, language)
        except ToolNotApplicable:
            log.info("Tool %s not applicable -> documents", r.intent)
            return False
        except ToolUnavailable as e:
            # Spoken verbatim: the LLM paraphrased error messages into wrong statements.
            result.tool = r.intent
            yield from self._reply(result, "tool", str(e))
            return True
        log.info("Tool %s -> %s", r.intent, context.replace("\n", " | "))
        result.route, result.tool, result.grounded = "tool", r.intent, True
        result.passages = [context]
        # No history: live data is outdated by the next turn.
        yield from self._grounded_answer(r.question, context, language, result, history=[])
        return True

    def _answer_chitchat(self, r: _Retrieval, language: str, result: AgentResult) -> Iterator[str]:
        if self.smalltalk and (reply := self.smalltalk.reply(r.vector, language)):
            yield from self._reply(result, "chitchat", reply)
            return
        # No context and no history: small talk must not turn into factual claims.
        result.route = "chitchat"
        yield from self._stream([
            {"role": "system", "content": prompts.CHITCHAT_SYSTEM[language]},
            {"role": "user", "content": prompts.chitchat_message(r.question, language)},
        ], result)

    def _answer_rag(self, r: _Retrieval, language: str, result: AgentResult) -> Iterator[str]:
        # All top-k chunks are used: hybrid search may rank a keyword match first
        # although its embedding score is lower; sentence verification guards the answer.
        result.route, result.grounded = "rag", True
        result.passages = [c.text for c in r.search.chunks]
        context = prompts.build_context(r.search.chunks)
        yield from self._grounded_answer(r.question, context, language, result, history=self.history)
        # The history keeps the plain question, not the context, to keep prompts short.
        self.history += [{"role": "user", "content": r.question},
                         {"role": "assistant", "content": result.answer}]
        self.history = self.history[max(0, len(self.history) - 2 * self.history_turns):]

    def _grounded_answer(self, question: str, context: str, language: str, result: AgentResult,
                         history: list[dict]) -> Iterator[str]:
        yield from self._stream([
            {"role": "system", "content": prompts.SYSTEM},
            *history,
            {"role": "user", "content": prompts.user_message(question, context, language, self.allow_combining)},
        ], result)
        if self.topics is not None:
            self.topics.observe_answer(result.answer)

    # ---- helpers --------------------------------------------------------------
    def _reply(self, result: AgentResult, route: str, text: str) -> Iterator[str]:
        """A fixed reply (template, error message): no LLM call."""
        result.route, result.answer = route, text
        yield text

    def _stream(self, messages: list[dict], result: AgentResult) -> Iterator[str]:
        parts = []
        for piece in self.llm.stream(messages):
            parts.append(piece)
            yield piece
        result.answer = "".join(parts).strip()
