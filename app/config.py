import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Environment variables win over the file (used by Docker).
    if url := os.getenv("OLLAMA_BASE_URL"):
        cfg["llm"]["base_url"] = url
    if model := os.getenv("LLM_MODEL"):
        cfg["llm"]["model"] = model

    return cfg
