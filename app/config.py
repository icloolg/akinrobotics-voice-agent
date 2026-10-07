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
    if device := os.getenv("STT_DEVICE"):          # "cpu" on machines without an NVIDIA GPU
        cfg["stt"]["device"] = device
    if compute_type := os.getenv("STT_COMPUTE_TYPE"):
        cfg["stt"]["compute_type"] = compute_type
    if url := os.getenv("ROBOT_API_URL"):
        cfg["tools"]["robot_status"]["url"] = url

    return cfg
