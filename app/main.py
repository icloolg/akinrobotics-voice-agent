"""Voice agent server.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

WebSocket protocol (/ws):
  client -> server : binary = int16 PCM, 16 kHz mono mic audio (continuous)
                     text   = {"type": "reset"}  (forget conversation history)
  server -> client : text   = JSON events: hello, end_of_speech, transcript,
                              sentence, empty, done
                     binary = int16 PCM answer audio at hello.sample_rate
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.audio import pcm16_to_float, resample
from app.config import load_config
from app.factory import build_agent, build_stt, build_tts
from app.logging_setup import setup_logging
from app.pipeline import VoicePipeline
from app.providers.base import SAMPLE_RATE
from app.providers.vad.silero import Segmenter

cfg = load_config()
setup_logging(cfg["logging"]["level"])
log = logging.getLogger("server")
state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load every model once at startup, then warm them up so the first
    # user turn is not slowed down by lazy initialisation.
    stt, tts, agent = build_stt(cfg), build_tts(cfg), build_agent(cfg)
    agent.llm.warmup()
    sample = resample(tts.synthesize("Merhaba, nasılsın?", "tr").astype("float32") / 32768,
                      tts.sample_rate, SAMPLE_RATE)
    stt.transcribe(sample)
    state["pipeline"] = VoicePipeline(stt, agent, tts)
    log.info("Ready.")
    yield


app = FastAPI(title="AKINROBOTICS Voice Agent", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if "pipeline" in state else "loading"}


@app.websocket("/ws")
async def voice(ws: WebSocket):
    await ws.accept()
    pipeline: VoicePipeline = state["pipeline"]
    pipeline.agent.reset()
    segmenter = Segmenter(**cfg.get("vad", {}))
    await ws.send_json({"type": "hello", "sample_rate": pipeline.tts.sample_rate})
    log.info("Client connected")

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):
                if '"reset"' in msg["text"]:
                    pipeline.agent.reset()
                continue

            for utterance in segmenter.push(pcm16_to_float(msg["bytes"])):
                await ws.send_json({"type": "end_of_speech"})
                await run_turn(ws, pipeline, utterance)
                segmenter.reset()  # drop anything heard while we were answering
    except WebSocketDisconnect:
        pass
    log.info("Client disconnected")


async def run_turn(ws: WebSocket, pipeline: VoicePipeline, utterance) -> None:
    """Pull events from the blocking pipeline in a worker thread, send each one
    as soon as it exists (so audio streams out sentence by sentence)."""
    events = pipeline.run_turn(utterance)
    while (event := await asyncio.to_thread(next, events, None)) is not None:
        if isinstance(event, bytes):
            await ws.send_bytes(event)
        else:
            await ws.send_json(event)
