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

    def __init__(self, model: str, base_url: str, temperature: float = 0.2,
                 max_tokens: int | None = None, num_ctx: int | None = None):
        self.model = model
        self.url = base_url.rstrip("/") + "/api/chat"
        self.options = {"temperature": temperature}
        if max_tokens:
            # Upper limit for one answer. Answers are 1-3 spoken sentences; the
            # limit only stops a model that starts to ramble or repeat itself.
            self.options["num_predict"] = max_tokens
        if num_ctx:
            # Context window. Ollama's default is larger than our prompts need
            # (system + 2 chunks + 1 past turn is ~800 tokens) and costs VRAM.
            self.options["num_ctx"] = num_ctx

    def stream(self, messages: list[dict]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self.options,
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
