"""Build the vector/keyword index from the knowledge folders, and the robot specs database.

    python -m scripts.ingest
Run again whenever a document is added or changed (or use the admin panel: /admin).
"""
from app.config import load_config
from app.factory import build_retriever
from app.knowledge import rebuild
from app.logging_setup import setup_logging

if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])
    result = rebuild(cfg, build_retriever(cfg))
    print(f"Done: {result['documents']} documents, {result['chunks']} chunks indexed.")
    if "robots" in result:
        print(f"Done: {result['robots']} robots in the specs database.")
