"""Knowledge documents on disk: which files are indexed and how they are read.

Indexed folders: rag.knowledge_dir (curated documents) plus rag.extra_dirs.
Fetched web pages (knowledge_web/) are not indexed by default: adding them
lowered accuracy and raised latency (see NOTLAR.md, item 64).
"""
from dataclasses import dataclass
from pathlib import Path

from app.knowledge.chunking import split_sections
from app.knowledge.pdf import Progress, pdf_to_markdown

DOCUMENT_TYPES = (".md", ".txt", ".pdf")


@dataclass
class Passage:
    """One chunk to be indexed."""
    text: str
    source: str  # file name, shown as the answer's source
    url: str     # original location, from the document's "Kaynak:" line


def knowledge_dirs(cfg: dict) -> list[Path]:
    rag = cfg["rag"]
    return [Path(d) for d in [rag["knowledge_dir"], *rag.get("extra_dirs", [])]]


def list_documents(cfg: dict) -> list[Path]:
    return sorted(p for d in knowledge_dirs(cfg) if d.exists()
                  for p in d.rglob("*") if p.is_file() and p.suffix.lower() in DOCUMENT_TYPES)


def read_document(path: Path, progress: Progress | None = None) -> str:
    """The document as markdown text."""
    if path.suffix.lower() == ".pdf":
        return pdf_to_markdown(path, progress)
    return path.read_text(encoding="utf-8")


def load_passages(cfg: dict, progress: Progress | None = None) -> tuple[list[Passage], dict[str, str]]:
    """Every indexed document, chunked. Also returns each document's text
    (file name -> markdown), used for topic suggestions."""
    max_chars = cfg["rag"].get("chunk_max_chars", 900)
    documents = list_documents(cfg)
    passages, texts = [], {}
    for i, path in enumerate(documents, 1):
        if progress:
            progress(f"{path.name} ({i}/{len(documents)})")
        texts[path.name] = read_document(path, progress)
        url, chunks = split_sections(texts[path.name], max_chars)
        passages += [Passage(chunk, path.name, url) for chunk in chunks]
    return passages, texts
