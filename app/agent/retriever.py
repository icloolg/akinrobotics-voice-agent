"""Knowledge-base search: the retrieval step of RAG.

Chunks are embedded with a multilingual sentence-embedding model and stored
in Chroma (persistent). For search, all chunk vectors and a BM25 keyword
index are held in memory: a few hundred chunks take about 1 MB.

Search modes (rag.search):
  dense   embedding similarity
  bm25    keyword match (5-letter stems, suited to Turkish suffixes)
  hybrid  both rankings merged with Reciprocal Rank Fusion (default)

Scores: every threshold in the system (rag.min_score, router margins) is on
the embedding-similarity scale. Hybrid search only changes which chunks are
returned and in what order; SearchResult.best_score stays the best embedding
similarity over all chunks.
"""
import logging
import re
from dataclasses import dataclass, field

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

COLLECTION = "knowledge"


@dataclass
class Chunk:
    text: str
    source: str
    score: float  # embedding (cosine) similarity to the question


@dataclass
class SearchResult:
    chunks: list[Chunk] = field(default_factory=list)
    best_score: float = 0.0  # best embedding similarity over ALL chunks


def _terms(text: str) -> list[str]:
    """BM25 terms: lower-cased words cut to 5 letters, a simple stemmer that
    works well for Turkish retrieval (Can et al., 2008)."""
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return [w[:5] for w in re.findall(r"\w+", text)]


class Retriever:
    def __init__(self, embedding_model: str, db_path: str,
                 query_prefix: str = "query: ", passage_prefix: str = "passage: ",
                 mode: str = "hybrid", rrf_k: int = 60, candidates: int = 30):
        self.mode, self.rrf_k, self.candidates = mode, rrf_k, candidates
        self.model = SentenceTransformer(embedding_model, device="cpu")
        # Instruction prefixes the e5 models were trained with; "" for models without them (e.g. bge-m3).
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        self._loaded_id = None

    # ---- embeddings -------------------------------------------------------
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode([self.query_prefix + t for t in texts], normalize_embeddings=True).tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode([self.passage_prefix + t for t in texts], normalize_embeddings=True).tolist()

    def encode_query(self, question: str) -> list[float]:
        return self.embed_queries([question])[0]

    # ---- index --------------------------------------------------------------
    def replace(self, texts: list[str], sources: list[str], urls: list[str],
                embeddings: list[list[float]]) -> int:
        """Replace the whole index with these chunks (embedded by the caller,
        so the slow part can run while the old index keeps serving)."""
        self.client.delete_collection(COLLECTION)
        self.collection = self.client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        if texts:
            ids = [f"{source}-{i}" for i, source in enumerate(sources)]
            metas = [{"source": s, "url": u} for s, u in zip(sources, urls)]
            self.collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
        log.info("Index replaced: %d chunks", len(texts))
        return len(texts)

    def reload_if_changed(self) -> None:
        """Reload the in-memory copy when the index was rebuilt (by another
        process or the admin panel). A rebuild creates a new collection id."""
        self.collection = self.client.get_collection(COLLECTION)
        if self._loaded_id != self.collection.id:
            data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            self._texts = data["documents"] or []
            self._sources = [m["source"] for m in data["metadatas"] or []]
            self._vectors = np.array(data["embeddings"]) if self._texts else np.zeros((0, 1))
            self._bm25 = BM25Okapi([_terms(t) for t in self._texts]) if self._texts else None
            self._loaded_id = self.collection.id
            log.info("Knowledge index loaded: %d chunks", len(self._texts))

    def mentions_known_terms(self, text: str, ignore: set[str] = frozenset()) -> bool:
        """True if a word of the text (as a BM25 stem) occurs in the knowledge base."""
        self.reload_if_changed()
        words = [w for w in _terms(text) if w not in ignore]
        return bool(self._bm25) and any(w in self._bm25.idf for w in words)

    def chunk_counts(self) -> dict[str, int]:
        """Number of indexed chunks per source document."""
        self.reload_if_changed()
        counts: dict[str, int] = {}
        for source in self._sources:
            counts[source] = counts.get(source, 0) + 1
        return counts

    # ---- search -------------------------------------------------------------
    def search(self, question: str, top_k: int = 3, vector: list[float] | None = None,
               mode: str | None = None) -> SearchResult:
        """Best chunks for the question; pass `vector` if it is already embedded."""
        self.reload_if_changed()
        if not self._texts:
            return SearchResult()
        dense = self._vectors @ np.array(vector or self.encode_query(question))
        mode = mode or self.mode
        if mode == "dense":
            order = list(np.argsort(-dense)[:top_k])
        else:
            keyword = self._bm25.get_scores(_terms(question))
            if mode == "bm25":
                order = list(np.argsort(-keyword)[:top_k])
            else:
                order = self._fuse(np.argsort(-dense)[:self.candidates],
                                   [i for i in np.argsort(-keyword)[:self.candidates] if keyword[i] > 0])[:top_k]
        chunks = [Chunk(self._texts[i], self._sources[i], round(float(dense[i]), 3)) for i in order]
        return SearchResult(chunks, round(float(dense.max()), 3))

    def _fuse(self, *rankings) -> list[int]:
        """Reciprocal Rank Fusion: score(i) = sum over rankings of 1 / (k + rank)."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)
