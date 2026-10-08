"""Download the models named in config.yaml.

    python -m scripts.download_models          # Piper voices -> models/piper (needed before first run)
    python -m scripts.download_models --all    # also Whisper, embedding, OCR and the NLI answer checker
                                               # (exported to ONNX); used by the Docker build, so the
                                               # container starts without internet access

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


NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def export_nli(cfg: dict) -> None:
    """The answer checker's NLI model as ONNX (app/agent/verify.py).
    In PyTorch it took 1.6 s per check on our CPU, in ONNX Runtime ~140 ms."""
    target = Path(cfg.get("verify", {}).get("model_dir", "models/nli"))
    if (target / "model.onnx").exists():
        print(f"  var      {target / 'model.onnx'}")
        return
    print(f"NLI: {NLI_MODEL} -> ONNX")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
    target.mkdir(parents=True, exist_ok=True)
    sample = tokenizer("öncül", "hipotez", return_tensors="pt")
    torch.onnx.export(model, (sample["input_ids"], sample["attention_mask"]), str(target / "model.onnx"),
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                    "logits": {0: "b"}},
                      opset_version=17, dynamo=False)
    tokenizer.save_pretrained(target)
    model.config.save_pretrained(target)
    print(f"  yazıldı  {target / 'model.onnx'}")


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
        export_nli(cfg)
        print("OCR (EasyOCR tr+en, for image-only PDF pages)")
        import easyocr
        easyocr.Reader(["tr", "en"], gpu=False, verbose=False)
    print("Tamam.")


if __name__ == "__main__":
    main()
