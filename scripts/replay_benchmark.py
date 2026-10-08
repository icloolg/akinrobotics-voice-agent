"""Reproducible latency benchmark: the recorded voice test set through the full pipeline.

    python -m scripts.replay_benchmark                 # writes logs/replay.jsonl
    python -m scripts.benchmark --file logs/replay.jsonl --markdown

Each recording (tests/voice/recordings) goes through exactly what the server
runs for a voice turn: STT -> agent (routing, retrieval, LLM) -> sentence
verification -> TTS. Turns are timed and logged like live turns, so the same
summary script reads them. Unlike live sessions the input is fixed, so the
numbers can be reproduced on another machine. Requires Ollama.
"""
import app.metrics as metrics
from app.config import load_config
from app.factory import build_agent, build_stt, build_tts, build_verifier
from app.logging_setup import setup_logging
from app.pipeline import VoicePipeline
from scripts.eval_agent import ensure_mock_api
from scripts.eval_stt import read_wav
from scripts.record_testset import OUT, load_sentences

if __name__ == "__main__":
    cfg = load_config()
    setup_logging("WARNING")
    metrics.TURNS_FILE = metrics.TURNS_FILE.with_name("replay.jsonl")
    metrics.TURNS_FILE.unlink(missing_ok=True)

    ensure_mock_api(cfg["tools"]["robot_status"]["url"])  # the live-status tool needs the (mock) fleet API
    stt, tts, agent = build_stt(cfg), build_tts(cfg), build_agent(cfg)
    agent.llm.warmup()
    pipeline = VoicePipeline(stt, agent, tts, endpoint_ms=cfg["vad"]["min_silence_ms"], verifier=build_verifier(cfg))

    recordings = [(s, OUT / f"{s['id']}.wav") for s in load_sentences()]
    recordings = [(s, p) for s, p in recordings if p.exists()]
    stt.transcribe(read_wav(recordings[0][1]))  # warm-up, not measured
    for s, path in recordings:
        pipeline.agent.reset()  # every recording is a new conversation
        for event in pipeline.run_turn(read_wav(path)):
            if isinstance(event, dict) and event["type"] == "done":
                m = event["metrics"]
                print(f"{s['id']}  {m['route']:9s}  {m['response_ms']:6.0f} ms  {m['question']}")
    print(f"\n{len(recordings)} turns -> {metrics.TURNS_FILE}")
