"""Knowledge-base search (the "R" in RAG).

Documents are split into chunks, each chunk is turned into a vector
(embedding) and stored in Chroma. A question is embedded the same way and
the closest chunks are returned with a similarity score in [0, 1].
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

COLLECTION = "knowledge"


@dataclass
class Chunk:
    text: str
    source: str
    score: float  # cosine similarity, higher = more relevant


class Retriever:
    def __init__(self, embedding_model: str, db_path: str):
        self.model = SentenceTransformer(embedding_model, device="cpu")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    # e5 models were trained with these prefixes; leaving them out hurts accuracy.
    def _embed_queries(self, texts: list[str]):
        return self.model.encode(["query: " + t for t in texts], normalize_embeddings=True).tolist()

    def _embed_passages(self, texts: list[str]):
        return self.model.encode(["passage: " + t for t in texts], normalize_embeddings=True).tolist()

    def encode_query(self, question: str) -> list[float]:
        return self._embed_queries([question])[0]

    def search(self, question: str, top_k: int = 3, vector: list[float] | None = None) -> list[Chunk]:
        """Pass `vector` if the question was already embedded (avoids doing it twice)."""
        query = [vector or self.encode_query(question)]
        try:
            res = self.collection.query(query_embeddings=query, n_results=top_k)
        except Exception:
            # scripts.ingest rebuilds the collection; a running server still holds the
            # deleted one and every question failed (seen live). Re-open it once.
            log.warning("Knowledge collection changed on disk; reloading it")
            self.collection = self.client.get_collection(COLLECTION)
            res = self.collection.query(query_embeddings=query, n_results=top_k)
        chunks = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            # Chroma returns cosine *distance* = 1 - similarity.
            chunks.append(Chunk(text=text, source=meta["source"], score=round(1 - dist, 3)))
        return chunks

    def index_folder(self, folder: str | Path, max_chars: int = 900) -> int:
        """(Re)build the index from every .md/.txt file in `folder`."""
        self.client.delete_collection(COLLECTION)
        self.collection = self.client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

        texts, metas, ids = [], [], []
        for path in sorted(Path(folder).rglob("*")):
            if path.suffix not in (".md", ".txt"):
                continue
            url, chunks = split_sections(path.read_text(encoding="utf-8"), max_chars)
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metas.append({"source": path.name, "url": url})
                ids.append(f"{path.name}-{i}")

        if texts:
            self.collection.add(ids=ids, documents=texts, metadatas=metas,
                                embeddings=self._embed_passages(texts))
        log.info("Indexed %d chunks from %s", len(texts), folder)
        return len(texts)


def split_sections(text: str, max_chars: int) -> tuple[str, list[str]]:
    """One chunk per markdown section ("## Ada-7 ..." + its paragraphs).

    Why sections and not fixed-size pieces: packing paragraphs up to N chars
    put unrelated topics in one chunk ("Mission" + "Product range"), so the
    chunk's vector matched neither topic well and "Hangi robotlarınız var?"
    missed the product list. A section is one topic.

    Each chunk starts with "<document title> - <section title>" so it is
    understandable on its own. A section longer than max_chars is split at
    paragraph boundaries, repeating the title. The "Kaynak: <url>" line is
    returned separately as metadata instead of being embedded.

    Returns (source_url, chunks).
    """
    url, doc_title = "", ""
    sections: list[tuple[str, list[str]]] = []  # (section title, paragraphs)
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if para.startswith("Kaynak:") or para.startswith("Source:"):
            url = para.split(":", 1)[1].strip()
        elif para.startswith("# "):
            doc_title = para[2:].strip()
        elif para.startswith("#"):
            sections.append((para.lstrip("# ").strip(), []))
        elif sections:
            sections[-1][1].append(para)
        else:  # text before the first section (e.g. a note under the title)
            sections.append(("", [para]))

    chunks = []
    for title, paragraphs in sections:
        header = " - ".join(t for t in (doc_title, title) if t)
        current = ""
        for para in paragraphs:
            if current and len(current) + len(para) > max_chars:
                chunks.append(f"{header}\n{current}".strip())
                current = ""
            current += para + "\n"
        if current:
            chunks.append(f"{header}\n{current}".strip())
    return url, chunks
