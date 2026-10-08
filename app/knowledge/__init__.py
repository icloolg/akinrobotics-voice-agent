"""Knowledge base management: documents -> chunks -> index (+ robot specs DB).

Two phases, so a running server keeps answering while documents are indexed:
  prepare  read documents (PDF OCR), chunk, embed  -- slow, no lock needed
  commit   swap the index, rebuild the specs DB    -- ~1 s, under the turn lock

`rebuild()` runs both in one go (scripts/ingest.py); `Indexer` runs them as a
background job with progress reporting (admin panel).
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field

from app.knowledge.documents import DOCUMENT_TYPES, Passage, knowledge_dirs, list_documents, load_passages
from app.knowledge.pdf import Progress
from app.knowledge.topics import active_topics, suggest_topics

__all__ = ["DOCUMENT_TYPES", "Indexer", "knowledge_dirs", "list_documents", "rebuild"]

log = logging.getLogger(__name__)


@dataclass
class Prepared:
    passages: list[Passage]
    embeddings: list[list[float]]
    texts: dict[str, str]  # file name -> document text


def prepare(cfg: dict, retriever, progress: Progress | None = None) -> Prepared:
    passages, texts = load_passages(cfg, progress)
    if progress:
        progress(f"Embedding: {len(passages)} parça")
    embeddings = retriever.embed_passages([p.text for p in passages]) if passages else []
    return Prepared(passages, embeddings, texts)


def commit(cfg: dict, retriever, prepared: Prepared) -> dict:
    chunks = retriever.replace([p.text for p in prepared.passages], [p.source for p in prepared.passages],
                               [p.url for p in prepared.passages], prepared.embeddings)
    result = {"chunks": chunks, "documents": len(prepared.texts)}
    specs = cfg.get("tools", {}).get("robot_specs")
    if specs:
        from app.tools.robot_specs import build_database
        result["robots"] = build_database(specs["csv_path"], specs["db_path"])
    return result


def rebuild(cfg: dict, retriever, progress: Progress | None = None) -> dict:
    """Re-index every document and rebuild the robot specs database."""
    return commit(cfg, retriever, prepare(cfg, retriever, progress))


def topic_suggestions(cfg: dict, texts: dict[str, str]) -> dict[str, list[str]]:
    """Per document, candidate topic names not yet known (see app/knowledge/topics.py)."""
    known = active_topics(cfg)
    return {name: s for name, text in texts.items() if (s := suggest_topics(text, known))}


@dataclass
class IndexStatus:
    state: str = "idle"          # idle | running | done | error
    step: str = ""               # current step, for the progress display
    started: float = 0.0
    seconds: float = 0.0
    result: dict = field(default_factory=dict)
    suggestions: dict = field(default_factory=dict)
    error: str = ""


class Indexer:
    """Background re-indexing for the server. Requests made while a job runs
    are merged into one follow-up run (several uploads -> at most two runs)."""

    def __init__(self, cfg: dict, retriever, lock: asyncio.Lock):
        self.cfg, self.retriever, self.lock = cfg, retriever, lock
        self.status = IndexStatus()
        self._task: asyncio.Task | None = None
        self._again = False
        self._indexed = self._signatures()  # documents as of the last index; new/changed ones get suggestions

    def _signatures(self) -> dict[str, tuple[int, int]]:
        return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in list_documents(self.cfg)}

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def request(self) -> None:
        if self.running:
            self._again = True
        else:
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            self._again = False
            self.status = IndexStatus(state="running", started=time.time())
            try:
                signatures = self._signatures()
                changed = {name for name, sig in signatures.items() if self._indexed.get(name) != sig}
                prepared = await asyncio.to_thread(prepare, self.cfg, self.retriever, self._progress)
                self._progress("İndeks değiştiriliyor")
                async with self.lock:  # no question is answered from a half-written index
                    result = await asyncio.to_thread(commit, self.cfg, self.retriever, prepared)
                    await asyncio.to_thread(self.retriever.reload_if_changed)
                self._indexed = signatures
                self.status.suggestions = topic_suggestions(
                    self.cfg, {n: t for n, t in prepared.texts.items() if n in changed})
                self.status.result, self.status.state, self.status.step = result, "done", ""
            except Exception as e:
                log.exception("Indexing failed")
                self.status.state, self.status.error = "error", str(e)
            self.status.seconds = round(time.time() - self.status.started, 1)
            if not self._again:
                return

    def _progress(self, step: str) -> None:
        self.status.step = step
