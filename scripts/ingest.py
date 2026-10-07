"""Build the vector index from the knowledge/ folder.

    python -m scripts.ingest
Run again whenever a document is added or changed.
"""
from app.config import load_config
from app.factory import build_retriever
from app.logging_setup import setup_logging

if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])
    n = build_retriever(cfg).index_folder(cfg["rag"]["knowledge_dir"])
    print(f"Done: {n} chunks indexed.")
