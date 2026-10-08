"""WebSocket server logic with fake models (no GPU, no downloads needed).

    pytest tests/test_server.py

Checks turn-taking: a normal turn, barge-in (talking over the answer), short
noises during an answer, half-duplex mode and per-client conversation history.

The fakes: "speech" is loud noise, the VAD is a loudness threshold, STT/LLM/TTS
return fixed data after a short sleep.
"""
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as server
from app.providers.base import SAMPLE_RATE, Transcript

rng = np.random.default_rng(0)


def speech(ms: int) -> bytes:
    return (rng.uniform(-0.5, 0.5, SAMPLE_RATE * ms // 1000) * 32767).astype(np.int16).tobytes()


def silence(ms: int) -> bytes:
    return np.zeros(SAMPLE_RATE * ms // 1000, dtype=np.int16).tobytes()


class FakeVadModel:
    def __call__(self, x, _sr):
        return np.float64(float(abs(x).mean()) > 0.05)

    def reset_states(self):
        pass


class FakeSTT:
    def transcribe(self, audio, language=None):
        time.sleep(0.05)
        return Transcript("Ada-7 kaç kilogram?", "tr", len(audio) / SAMPLE_RATE, 0.05)


class FakeTTS:
    sample_rate = 22050

    def synthesize(self, text, language):
        return np.zeros(int(self.sample_rate * 0.3), dtype=np.int16)


class FakeLLM:
    def warmup(self):
        pass


class FakeAgent:
    """Answers in `sentences` sentences, one every `delay` seconds."""
    sentences, delay = 2, 0.02
    retriever = None  # only handed to the (unused) background indexer

    def __init__(self):
        self.llm = FakeLLM()
        self.history = []

    def fork(self):
        return FakeAgent()

    def reset(self):
        self.history.clear()

    def answer(self, question, language, result):
        result.route, result.grounded = "rag", True
        for i in range(self.sentences):
            time.sleep(self.delay)
            yield f"Bu {i + 1}. cevap cümlesidir. "
        result.answer = "tamam"
        self.history.append(question)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "build_stt", lambda cfg: FakeSTT())
    monkeypatch.setattr(server, "build_tts", lambda cfg: FakeTTS())
    monkeypatch.setattr(server, "build_agent", lambda cfg: FakeAgent())
    monkeypatch.setattr("app.providers.vad.silero.load_silero_vad", lambda onnx=True: FakeVadModel())
    monkeypatch.setattr("app.metrics.TURNS_FILE", tmp_path / "turns.jsonl")
    monkeypatch.setitem(server.cfg, "vad", {"min_silence_ms": 500, "barge_in": True, "barge_in_ms": 250})
    monkeypatch.setattr(FakeAgent, "sentences", 2)
    monkeypatch.setattr(FakeAgent, "delay", 0.02)
    with TestClient(server.app) as c:
        yield c


def read_until(ws, kind: str) -> list:
    """Collect messages until a JSON event of type `kind` arrives."""
    seen = []
    while True:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            seen.append(msg["bytes"])
            continue
        import json
        event = json.loads(msg["text"])
        seen.append(event)
        if event["type"] == kind:
            return seen


def types(messages: list) -> list[str]:
    return ["audio" if isinstance(m, bytes) else m["type"] for m in messages]


def test_web_client_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "AKINROBOTICS" in client.get("/").text


def test_one_turn(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_bytes(speech(600) + silence(800))
        seen = read_until(ws, "done")
    assert types(seen) == ["speech_start", "end_of_speech", "transcript",
                           "sentence", "audio", "sentence", "audio", "done"]
    metrics = seen[-1]["metrics"]
    # Reported latency = VAD silence wait + time to the first answer audio.
    assert metrics["response_ms"] == pytest.approx(500 + metrics["ms"]["first_audio"], abs=0.2)


def test_barge_in_stops_the_answer(client, monkeypatch):
    monkeypatch.setattr(FakeAgent, "sentences", 50)
    monkeypatch.setattr(FakeAgent, "delay", 0.05)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(speech(600) + silence(800))
        read_until(ws, "sentence")                 # the long answer has started

        ws.send_bytes(speech(400))                 # the user talks over it
        seen = read_until(ws, "speech_start")
        assert "done" not in types(seen)           # the first answer never finished

        monkeypatch.setattr(FakeAgent, "sentences", 1)
        ws.send_bytes(speech(300) + silence(800))  # ... and finishes the new question
        seen = read_until(ws, "done")
    # Nothing of the old answer is sent after speech_start.
    assert types(seen) == ["end_of_speech", "transcript", "sentence", "audio", "done"]


def test_short_noise_during_answer_is_ignored(client, monkeypatch):
    monkeypatch.setattr(FakeAgent, "sentences", 6)
    monkeypatch.setattr(FakeAgent, "delay", 0.05)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(speech(600) + silence(800))
        read_until(ws, "sentence")
        ws.send_bytes(speech(150) + silence(800))  # shorter than barge_in_ms: echo, a click
        seen = read_until(ws, "done")
    assert "speech_start" not in types(seen) and "transcript" not in types(seen)
    assert types(seen).count("sentence") == 5      # the answer went on to the end


def test_half_duplex_ignores_mic_while_answering(client, monkeypatch):
    monkeypatch.setitem(server.cfg["vad"], "barge_in", False)
    monkeypatch.setattr(FakeAgent, "sentences", 6)
    monkeypatch.setattr(FakeAgent, "delay", 0.05)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["barge_in"] is False
        ws.send_bytes(speech(600) + silence(800))
        read_until(ws, "sentence")
        ws.send_bytes(speech(600) + silence(800))
        seen = read_until(ws, "done")
    assert "speech_start" not in types(seen)


def test_each_client_has_its_own_history(client):
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        a.receive_json(), b.receive_json()
        a.send_bytes(speech(600) + silence(800))
        read_until(a, "done")
        b.send_bytes(speech(600) + silence(800))
        read_until(b, "done")
    assert server.state["agent"].history == []     # the shared agent is only a template
