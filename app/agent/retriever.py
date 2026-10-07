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

    def search(self, question: str, top_k: int = 3) -> list[Chunk]:
        res = self.collection.query(
            query_embeddings=self._embed_queries([question]), n_results=top_k
        )
        chunks = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            # Chroma returns cosine *distance* = 1 - similarity.
            chunks.append(Chunk(text=text, source=meta["source"], score=round(1 - dist, 3)))
        return chunks

    def index_folder(self, folder: str | Path, chunk_chars: int = 600) -> int:
        """(Re)build the index from every .md/.txt file in `folder`."""
        self.client.delete_collection(COLLECTION)
        self.collection = self.client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

        texts, metas, ids = [], [], []
        for path in sorted(Path(folder).rglob("*")):
            if path.suffix not in (".md", ".txt"):
                continue
            for i, chunk in enumerate(split_text(path.read_text(encoding="utf-8"), chunk_chars)):
                texts.append(chunk)
                metas.append({"source": path.name})
                ids.append(f"{path.name}-{i}")

        if texts:
            self.collection.add(ids=ids, documents=texts, metadatas=metas,
                                embeddings=self._embed_passages(texts))
        log.info("Indexed %d chunks from %s", len(texts), folder)
        return len(texts)


def split_text(text: str, max_chars: int) -> list[str]:
    """Split on blank lines (paragraphs), then pack paragraphs up to max_chars.

    Keeping whole paragraphs together keeps each chunk meaningful on its own.
    A markdown heading is carried into the chunk so the chunk knows its topic.
    """
    chunks, current, heading = [], "", ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if para.startswith("#"):
            heading = para.lstrip("# ").split("\n")[0]
        if current and len(current) + len(para) > max_chars:
            chunks.append(current)
            current = f"{heading}\n" if heading and not para.startswith("#") else ""
        current += para + "\n"
    if current.strip():
        chunks.append(current)
    return [c.strip() for c in chunks]
