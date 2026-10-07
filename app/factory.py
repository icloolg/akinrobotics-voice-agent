"""Builds components from config.yaml.

To add a new provider: write the class, then add one line to the matching
registry below. Imports are lazy so unused libraries are never loaded.
"""


def build_stt(cfg: dict):
    c = cfg["stt"]
    if c["provider"] == "faster_whisper":
        from app.providers.stt.faster_whisper import FasterWhisperSTT
        return FasterWhisperSTT(c["model"], c["device"], c["compute_type"], c["languages"],
                                c.get("hotwords"), c.get("default_language", "tr"),
                                c.get("switch_min_prob", 0.3), c.get("switch_ratio", 2.0))
    raise ValueError(f"Unknown STT provider: {c['provider']}")


def build_llm(cfg: dict):
    c = cfg["llm"]
    if c["provider"] == "ollama":
        from app.providers.llm.ollama import OllamaLLM
        return OllamaLLM(c["model"], c["base_url"], c["temperature"], c.get("max_tokens"),
                         c.get("num_ctx"))
    if c["provider"] == "openai_compat":  # llama.cpp server, vLLM, LM Studio, ...
        import os
        from app.providers.llm.openai_compat import OpenAICompatLLM
        return OpenAICompatLLM(c["model"], c["base_url"], c["temperature"], c.get("max_tokens"),
                               os.getenv(c.get("api_key_env") or "LLM_API_KEY"))
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


def build_tools(cfg: dict) -> dict:
    """Tool registry: name in config.yaml -> how to build it.
    New tool = one entry here + its settings under tools: in config.yaml."""
    registry = {
        "datetime": lambda c: _tool("app.tools.datetime_tool", "DateTimeTool")(c.get("utc_offset_hours", 3)),
        "robot_status": lambda c: _tool("app.tools.robot_status", "RobotStatusTool")(c["url"]),
        "robot_specs": lambda c: _tool("app.tools.robot_specs", "RobotSpecsTool")(c["db_path"]),
    }
    tools_cfg = cfg.get("tools", {})
    return {name: registry[name](tools_cfg.get(name, {})) for name in tools_cfg.get("enabled", [])}


def _tool(module: str, cls: str):
    import importlib
    return getattr(importlib.import_module(module), cls)


def build_agent(cfg: dict, llm=None, retriever=None):
    from app.agent.agent import Agent
    from app.agent.router import CHITCHAT_EXAMPLES, Router
    from app.agent.topics import TopicTracker
    retriever = retriever or build_retriever(cfg)
    tools = build_tools(cfg)

    router = Router(retriever._embed_queries)
    router.add("chitchat", CHITCHAT_EXAMPLES, **cfg["router"])
    for name, tool in tools.items():
        t = cfg["tools"][name]
        router.add(name, tool.examples, t["min_score"], t["margin"], tool.keywords)

    return Agent(
        llm or build_llm(cfg),
        retriever,
        router,
        tools,
        top_k=cfg["rag"]["top_k"],
        min_score=cfg["rag"]["min_score"],
        history_turns=cfg["llm"]["history_turns"],
        topics=TopicTracker(cfg["followup"]["topics"]) if cfg.get("followup") else None,
        followup_max_score=cfg.get("followup", {}).get("max_score", 0.85),
    )
