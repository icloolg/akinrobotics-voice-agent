import json
import logging
from typing import Iterator

import requests

from app.providers.base import LLMProvider

log = logging.getLogger(__name__)


class OllamaLLM(LLMProvider):
    """Talks to Ollama's REST API directly (POST /api/chat, stream=true).

    Ollama returns one JSON object per line; each has a small piece of text
    in message.content and "done": true on the last line.
    """

    def __init__(self, model: str, base_url: str, temperature: float = 0.2):
        self.model = model
        self.url = base_url.rstrip("/") + "/api/chat"
        self.temperature = temperature

    def stream(self, messages: list[dict]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature},
            "keep_alive": "30m",  # keep the model in GPU memory between turns
        }
        with requests.post(self.url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if piece := chunk.get("message", {}).get("content"):
                    yield piece
                if chunk.get("done"):
                    break

    def warmup(self) -> None:
        """First request loads the model into memory; do it before users talk."""
        for _ in self.stream([{"role": "user", "content": "hi"}]):
            pass
        log.info("LLM warmed up: %s", self.model)
