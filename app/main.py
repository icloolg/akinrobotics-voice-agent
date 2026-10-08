"""AkınVoice server: browser client, WebSocket voice turns, admin API.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Open http://localhost:8000 for the browser client, or run client/voice_client.py.

WebSocket protocol (/ws):
  client -> server : binary = int16 PCM, 16 kHz mono mic audio (continuous)
                     text   = {"type": "reset"}  (forget conversation history)
                              {"type": "language", "value": "auto"|"tr"|"en"}  (speech language)
  server -> client : text   = JSON events: hello, speech_start, end_of_speech,
                              transcript, sentence, empty, done
                     binary = int16 PCM answer audio at hello.sample_rate

Barge-in: the server keeps listening while it answers. If the user talks over
the answer (vad.barge_in_ms of voiced audio), it sends `speech_start`, stops
the running turn and treats the new speech as the next question. A client
that cannot cancel its own speaker echo simply does not send mic audio while
it is playing (client/voice_client.py does this); then nothing is interrupted.
"""
import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.admin import create_router as admin_router
from app.audio import pcm16_to_float, resample
from app.config import load_config
from app.factory import build_agent, build_stt, build_tts, build_verifier
from app.knowledge import Indexer
from app.logging_setup import setup_logging
from app.mock_robot_api import router as mock_robot_api
from app.pipeline import VoicePipeline
from app.providers.base import SAMPLE_RATE
from app.providers.vad.silero import Segmenter

cfg = load_config()
setup_logging(cfg["logging"]["level"])
log = logging.getLogger("server")
state: dict = {}
WEB_CLIENT = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load every model once at startup, then warm them up so the first
    # user turn is not slowed down by lazy initialisation.
    stt, tts, agent = build_stt(cfg), build_tts(cfg), build_agent(cfg)
    agent.llm.warmup()
    sample = resample(tts.synthesize("Merhaba, nasılsın?", "tr").astype("float32") / 32768,
                      tts.sample_rate, SAMPLE_RATE)
    stt.transcribe(sample)
    busy = asyncio.Lock()  # one turn at a time on the GPU, also with several clients connected
    state.update(stt=stt, tts=tts, agent=agent, verifier=build_verifier(cfg), busy=busy,
                 indexer=Indexer(cfg, agent.retriever, busy))
    log.info("Ready.")
    yield


API_DESCRIPTION = """
Sesli asistan sunucusu. Tarayıcı istemcisi: [`/`](/) · Yönetim paneli: [`/admin`](/admin)

### Sesli konuşma: WebSocket `/ws`

WebSocket uç noktaları Swagger'da listelenmez; protokol:

| Yön | Tür | İçerik |
|---|---|---|
| istemci → sunucu | binary | int16 PCM, 16 kHz mono mikrofon sesi (kesintisiz) |
| istemci → sunucu | text | `{"type": "language", "value": "auto" \\| "tr" \\| "en"}` konuşma dili · `{"type": "reset"}` sohbeti sıfırla |
| sunucu → istemci | text | JSON olaylar: `hello`, `speech_start`, `end_of_speech`, `transcript`, `sentence`, `empty`, `done` (gecikme ölçümleriyle) |
| sunucu → istemci | binary | cevap sesi, int16 PCM (`hello.sample_rate`), cümle cümle |

Kullanıcı cevap sırasında konuşursa (söz kesme) sunucu `speech_start` gönderir ve süren cevabı durdurur.

### Yönetim uç noktaları

`X-Admin-Token` başlığı gerekir (sunucuda `ADMIN_TOKEN` ortam değişkeni); tanımlı değilse yönetim kapalıdır.
"""

TAGS = [
    {"name": "Yönetim", "description": "Bilgi kaynakları, arka planda indeksleme, konu adları (`X-Admin-Token` gerekir)."},
    {"name": "Sistem", "description": "Sağlık kontrolü."},
    {"name": "Örnek filo API'si", "description": "Canlı robot durumu aracının kullandığı örnek servis; gerçek kullanımda filo yönetim sistemi."},
]

# Swagger UI is on by default (useful for review); DOCS_ENABLED=0 turns it off in production.
DOCS = os.getenv("DOCS_ENABLED", "1") != "0"

app = FastAPI(title="AkınVoice API", version="1.0.0", description=API_DESCRIPTION, openapi_tags=TAGS,
              docs_url="/docs" if DOCS else None, redoc_url=None,
              openapi_url="/openapi.json" if DOCS else None, lifespan=lifespan)
app.include_router(mock_robot_api)  # demo data source for the robot_status tool
app.include_router(admin_router(cfg, state))  # /admin: knowledge sources (needs ADMIN_TOKEN)
app.mount("/static", StaticFiles(directory=WEB_CLIENT.parent), name="static")  # robot images


@app.get("/", include_in_schema=False)
def web_client():
    return FileResponse(WEB_CLIENT)


@app.get("/health", tags=["Sistem"], summary="Sunucu hazır mı?")
def health():
    """`ok`: modeller yüklendi ve ısıtıldı; `loading`: açılış sürüyor."""
    return {"status": "ok" if "agent" in state else "loading"}


class Session:
    """One connected client: its conversation, its running answer, its VAD."""

    def __init__(self, ws: WebSocket):
        vad = dict(cfg.get("vad", {}))
        self.barge_in = vad.pop("barge_in", True)
        self.barge_in_ms = vad.pop("barge_in_ms", 250)
        self.segmenter = Segmenter(**vad)
        self.ws = ws
        # Own conversation history per client; the models are shared.
        self.pipeline = VoicePipeline(state["stt"], state["agent"].fork(), state["tts"],
                                      endpoint_ms=vad.get("min_silence_ms", 500),
                                      verifier=state["verifier"])
        self.task: asyncio.Task | None = None
        self.stop_flag = threading.Event()
        self.playback_end = 0.0   # loop time when the audio sent so far finishes playing
        self.announced = False    # speech_start already sent for the current utterance
        self.language: str | None = None  # None = detect; "tr"/"en" = chosen on the page

    @property
    def answering(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def speaking(self) -> bool:
        """The client is (probably) still playing the answer."""
        return self.answering or asyncio.get_running_loop().time() < self.playback_end

    async def on_audio(self, pcm: bytes) -> None:
        if self.answering and not self.barge_in:
            return  # half-duplex: ignore the mic while answering
        was_speaking = self.speaking
        utterances = self.segmenter.push(pcm16_to_float(pcm))

        if (self.segmenter.in_speech and not self.announced
                and self.segmenter.voiced_ms >= self.barge_in_ms):
            await self.interrupt()

        for utterance in utterances:
            if not self.announced:
                if was_speaking:
                    # Too short to be a real interruption: speaker echo, a click, "hm".
                    log.info("Ignored %.0f ms of sound during the answer", self.segmenter.last_voiced_ms)
                    continue
                await self.interrupt()
            await self.ws.send_json({"type": "end_of_speech"})
            self.stop_flag = threading.Event()
            self.task = asyncio.create_task(self.run_turn(utterance, self.stop_flag))
        if not self.segmenter.in_speech:
            self.announced = False

    async def interrupt(self) -> None:
        """The user is speaking: the client stops playing, a running answer is cancelled."""
        self.announced = True
        await self.ws.send_json({"type": "speech_start"})
        self.playback_end = 0.0
        await self.stop_turn()

    async def stop_turn(self) -> None:
        if self.task is not None:
            self.stop_flag.set()
            await asyncio.gather(self.task, return_exceptions=True)

    async def run_turn(self, utterance, stop: threading.Event) -> None:
        """Pull events from the blocking pipeline in a worker thread, send each one
        as soon as it exists (so audio streams out sentence by sentence)."""
        loop = asyncio.get_running_loop()
        async with state["busy"]:
            events = self.pipeline.run_turn(utterance, self.language)
            try:
                while not stop.is_set():
                    event = await asyncio.to_thread(next, events, None)
                    if event is None or stop.is_set():
                        break
                    if isinstance(event, bytes):
                        seconds = len(event) / 2 / self.pipeline.tts.sample_rate
                        self.playback_end = max(self.playback_end, loop.time()) + seconds
                        await self.ws.send_bytes(event)
                    else:
                        await self.ws.send_json(event)
            except (WebSocketDisconnect, RuntimeError):  # client went away mid-answer
                pass
            except Exception:
                log.exception("Turn failed")
                await _send_quietly(self.ws, {"type": "empty"})  # lets the client listen again
            finally:
                # Closing the generator also closes the LLM's HTTP stream.
                await asyncio.to_thread(events.close)
                if stop.is_set():
                    log.info("Turn interrupted by the user")


async def _send_quietly(ws: WebSocket, event: dict) -> None:
    try:
        await ws.send_json(event)
    except Exception:
        pass


@app.websocket("/ws")
async def voice(ws: WebSocket):
    await ws.accept()
    session = Session(ws)
    await ws.send_json({"type": "hello", "sample_rate": session.pipeline.tts.sample_rate,
                        "barge_in": session.barge_in})
    log.info("Client connected")
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):
                kind, data = _message(msg["text"])
                if kind == "reset":
                    await session.stop_turn()
                    session.pipeline.agent.reset()
                elif kind == "language":  # speech language chosen on the page
                    session.language = data.get("value") if data.get("value") in ("tr", "en") else None
            elif msg.get("bytes"):
                await session.on_audio(msg["bytes"])
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await session.stop_turn()
    log.info("Client disconnected")


def _message(text: str) -> tuple[str, dict]:
    try:
        data = json.loads(text)
        return data.get("type", ""), data
    except (ValueError, AttributeError):
        return "", {}
