"""Builds every component from config.yaml (registry-based factory).

Each component type has a registry: provider name in config.yaml -> builder.
Adding an implementation = one class + one registry entry; nothing else
changes. Imports happen inside the builders, so libraries of providers that
are not selected are never loaded.
"""
import os
from typing import Callable


def _faster_whisper(c: dict):
    from app.providers.stt.faster_whisper import FasterWhisperSTT
    return FasterWhisperSTT(c["model"], c["device"], c["compute_type"], c["languages"],
                            c.get("hotwords"), c.get("default_language", "tr"),
                            c.get("switch_min_prob", 0.3), c.get("switch_ratio", 2.0))


def _ollama(c: dict):
    from app.providers.llm.ollama import OllamaLLM
    return OllamaLLM(c["model"], c["base_url"], c["temperature"], c.get("max_tokens"), c.get("num_ctx"))


def _openai_compat(c: dict):  # llama.cpp server, vLLM, LM Studio, ...
    from app.providers.llm.openai_compat import OpenAICompatLLM
    return OpenAICompatLLM(c["model"], c["base_url"], c["temperature"], c.get("max_tokens"),
                           os.getenv(c.get("api_key_env") or "LLM_API_KEY"))


def _piper(c: dict):
    from app.providers.tts.piper import PiperTTS
    return PiperTTS(c["voices_dir"], c["voices"], c.get("length_scale", 1.0))


def _datetime_tool(c: dict):
    from app.tools.datetime_tool import DateTimeTool
    return DateTimeTool(c.get("utc_offset_hours", 3))


def _robot_status(c: dict):
    from app.tools.robot_status import RobotStatusTool
    return RobotStatusTool(c["url"])


def _robot_specs(c: dict):
    from app.tools.robot_specs import RobotSpecsTool
    return RobotSpecsTool(c["db_path"])


STT: dict[str, Callable] = {"faster_whisper": _faster_whisper}
LLM: dict[str, Callable] = {"ollama": _ollama, "openai_compat": _openai_compat}
TTS: dict[str, Callable] = {"piper": _piper}
TOOLS: dict[str, Callable] = {"datetime": _datetime_tool, "robot_status": _robot_status,
                              "robot_specs": _robot_specs}


def _build(registry: dict[str, Callable], kind: str, c: dict):
    try:
        builder = registry[c["provider"]]
    except KeyError:
        raise ValueError(f"Unknown {kind} provider {c['provider']!r}; available: {', '.join(registry)}") from None
    return builder(c)


def build_stt(cfg: dict):
    return _build(STT, "STT", cfg["stt"])


def build_llm(cfg: dict):
    return _build(LLM, "LLM", cfg["llm"])


def build_tts(cfg: dict):
    return _build(TTS, "TTS", cfg["tts"])


def build_tools(cfg: dict) -> dict:
    tools_cfg = cfg.get("tools", {})
    return {name: TOOLS[name](tools_cfg.get(name, {})) for name in tools_cfg.get("enabled", [])}


def build_retriever(cfg: dict):
    from app.agent.retriever import Retriever
    c = cfg["rag"]
    return Retriever(c["embedding_model"], c["db_path"],
                     c.get("query_prefix", "query: "), c.get("passage_prefix", "passage: "),
                     c.get("search", "hybrid"), c.get("rrf_k", 60), c.get("candidates", 30))


def build_verifier(cfg: dict):
    """NLI sentence verification, or None when disabled."""
    c = cfg.get("verify", {})
    if not c.get("enabled"):
        return None
    from app.agent.verify import Verifier
    return Verifier(c["model_dir"], c.get("threshold", 0.10))


def build_agent(cfg: dict, llm=None, retriever=None):
    from app.agent.agent import Agent
    from app.agent.router import Router
    from app.agent.smalltalk import ALL_EXAMPLES, NOT_SMALLTALK, SmallTalk
    from app.agent.topics import TopicTracker
    from app.knowledge.topics import active_topics

    retriever = retriever or build_retriever(cfg)
    tools = build_tools(cfg)

    # The router compares the question to example sentences with the retriever's embedding model.
    router = Router(retriever.embed_queries)
    router.add("chitchat", ALL_EXAMPLES, **cfg["router"], exclude=NOT_SMALLTALK)
    for name, tool in tools.items():
        t = cfg["tools"][name]
        router.add(name, tool.examples, t["min_score"], t["margin"], tool.keywords)

    followup = cfg.get("followup")
    return Agent(
        llm or build_llm(cfg), retriever, router, tools,
        top_k=cfg["rag"]["top_k"],
        min_score=cfg["rag"]["min_score"],
        history_turns=cfg["llm"]["history_turns"],
        topics=TopicTracker(active_topics(cfg)) if followup else None,
        followup_max_score=(followup or {}).get("max_score", 0.85),
        smalltalk=SmallTalk(retriever.embed_queries, cfg.get("smalltalk", {}).get("min_score", 0.88)),
        allow_combining=cfg.get("verify", {}).get("enabled", False),
    )
