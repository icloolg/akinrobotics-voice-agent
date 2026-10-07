"""Download the models named in config.yaml.

    python -m scripts.download_models          # Piper voices -> models/piper (needed before first run)
    python -m scripts.download_models --all    # also Whisper + embedding model into the Hugging Face
                                               # cache (used by the Docker build, so the container
                                               # starts without internet access)

Files that already exist are skipped. The LLM is pulled by Ollama itself:
    ollama pull qwen2.5:3b
"""
import argparse
from pathlib import Path

import requests

from app.config import load_config

PIPER_REPO = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def piper_url(voice: str) -> str:
    """tr_TR-dfki-medium -> .../tr/tr_TR/dfki/medium/tr_TR-dfki-medium"""
    locale, name, quality = voice.split("-")
    return f"{PIPER_REPO}/{locale.split('_')[0]}/{locale}/{name}/{quality}/{voice}"


def download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"  var      {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")  # a broken download never looks complete
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    tmp.replace(target)
    print(f"  indirildi {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="also cache the Whisper and embedding models")
    args = parser.parse_args()
    cfg = load_config()

    print("Piper sesleri:")
    for voice in cfg["tts"]["voices"].values():
        for ext in (".onnx", ".onnx.json"):
            download(piper_url(voice) + ext, Path(cfg["tts"]["voices_dir"]) / (voice + ext))

    if args.all:
        print(f"Whisper: {cfg['stt']['model']}")
        from faster_whisper import download_model
        download_model(cfg["stt"]["model"])
        print(f"Embedding: {cfg['rag']['embedding_model']}")
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(cfg["rag"]["embedding_model"], device="cpu")
        from silero_vad import load_silero_vad
        load_silero_vad(onnx=True)  # bundled with the package; this only checks it loads
    print("Tamam.")


if __name__ == "__main__":
    main()
