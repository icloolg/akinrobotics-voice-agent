"""Builds components from config.yaml.

To add a new provider: write the class, then add one line to the matching
registry below. Imports are lazy so unused libraries are never loaded.
"""


def build_stt(cfg: dict):
    c = cfg["stt"]
    if c["provider"] == "faster_whisper":
        from app.providers.stt.faster_whisper import FasterWhisperSTT
        return FasterWhisperSTT(c["model"], c["device"], c["compute_type"],
                                c["languages"], c.get("hotwords"))
    raise ValueError(f"Unknown STT provider: {c['provider']}")


def build_llm(cfg: dict):
    c = cfg["llm"]
    if c["provider"] == "ollama":
        from app.providers.llm.ollama import OllamaLLM
        return OllamaLLM(c["model"], c["base_url"], c["temperature"])
    raise ValueError(f"Unknown LLM provider: {c['provider']}")


def build_tts(cfg: dict):
    c = cfg["tts"]
    if c["provider"] == "piper":
        from app.providers.tts.piper import PiperTTS
        return PiperTTS(c["voices_dir"], c["voices"], c.get("length_scale", 1.0))
    raise ValueError(f"Unknown TTS provider: {c['provider']}")


def build_retriever(cfg: dict):
    from app.agent.retriever import Retriever
    return Retriever(cfg["rag"]["embedding_model"], cfg["rag"]["db_path"])


def build_agent(cfg: dict, llm=None, retriever=None):
    from app.agent.agent import Agent
    return Agent(
        llm or build_llm(cfg),
        retriever or build_retriever(cfg),
        top_k=cfg["rag"]["top_k"],
        min_score=cfg["rag"]["min_score"],
        history_turns=cfg["llm"]["history_turns"],
    )
