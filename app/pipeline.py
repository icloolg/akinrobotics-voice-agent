"""One conversation turn: utterance audio in -> events + audio out.

    STT -> Agent (RAG + LLM stream) -> SentenceSplitter -> TTS

Every sentence is spoken as soon as it is complete, so the user hears the
first sentence while the LLM is still writing the rest. This is the main
latency optimisation of the project.

run_turn() is a plain (blocking) generator; the server pulls events from
it one by one. It yields:
    dict  -> JSON event for the client (transcript, sentence, done, ...)
    bytes -> int16 PCM audio at tts.sample_rate
"""
import logging
import time
from typing import Iterator

import numpy as np

from app.agent.agent import Agent, AgentResult
from app.metrics import TurnTimer, rtf
from app.providers.base import STTProvider, TTSProvider
from app.text import SentenceSplitter

log = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(self, stt: STTProvider, agent: Agent, tts: TTSProvider):
        self.stt = stt
        self.agent = agent
        self.tts = tts
        self.turns = 0

    def run_turn(self, audio: np.ndarray) -> Iterator[dict | bytes]:
        self.turns += 1
        timer = TurnTimer(self.turns)  # t=0: VAD decided the user stopped talking

        # 1. Speech -> text
        transcript = self.stt.transcribe(audio)
        timer.mark("stt_done")
        if not transcript.text:
            yield {"type": "empty"}
            return
        lang = transcript.language
        log.info("User (%s): %s", lang, transcript.text)
        yield {"type": "transcript", "text": transcript.text, "language": lang}

        # 2. Text -> answer stream -> 3. sentences -> 4. speech
        result = AgentResult()
        splitter = SentenceSplitter()
        tts_seconds = audio_seconds = 0.0

        def speak(sentence: str):
            nonlocal tts_seconds, audio_seconds
            start = time.perf_counter()
            pcm = self.tts.synthesize(sentence, lang)
            tts_seconds += time.perf_counter() - start
            timer.mark("first_audio")  # only the first call is recorded
            audio_seconds += len(pcm) / self.tts.sample_rate
            yield {"type": "sentence", "text": sentence}
            yield pcm.tobytes()

        for token in self.agent.answer(transcript.text, lang, result):
            timer.mark("llm_first_token")
            for sentence in splitter.push(token):
                yield from speak(sentence)
        for sentence in splitter.flush():
            yield from speak(sentence)
        timer.mark("done")

        timer.extra = {
            "language": lang,
            "question": transcript.text,
            "answer": result.answer,
            "route": result.route,
            "grounded": result.grounded,
            "sources": sorted({c.source for c in result.chunks}) if result.grounded else [],
            "retrieval_ms": result.retrieval_ms,
            "stt_rtf": rtf(transcript.processing_seconds, transcript.audio_seconds),
            "tts_rtf": rtf(tts_seconds, audio_seconds),
            "user_audio_s": round(transcript.audio_seconds, 2),
        }
        timer.log()
        yield {"type": "done", "metrics": timer.summary()}
