"""Build the vector index from the knowledge/ folder, and the robot specs database.

    python -m scripts.ingest
Run again whenever a document is added or changed.
"""
from app.config import load_config
from app.factory import build_retriever
from app.logging_setup import setup_logging

if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])
    n = build_retriever(cfg).index_folder(cfg["rag"]["knowledge_dir"],
                                          cfg["rag"].get("chunk_max_chars", 900))
    print(f"Done: {n} chunks indexed.")

    specs = cfg.get("tools", {}).get("robot_specs")
    if specs:
        from app.tools.robot_specs import build_database
        print(f"Done: {build_database(specs['csv_path'], specs['db_path'])} robots in {specs['db_path']}.")
