import json
import logging
from typing import Iterator

import requests

from app.providers.base import LLMProvider

log = logging.getLogger(__name__)


class OpenAICompatLLM(LLMProvider):
    """Any server that speaks the OpenAI chat API (POST /v1/chat/completions, stream=true).

    That is the common interface of open-source model servers: llama.cpp
    server, vLLM, LM Studio, LocalAI, text-generation-inference, and Ollama
    itself (http://host:11434/v1). Hosted services use it too, so the LLM can
    be moved to another machine or service by editing config.yaml only:

        llm:
          provider: openai_compat
          base_url: http://127.0.0.1:8080/v1
          model: qwen2.5-3b-instruct
          api_key_env: LLM_API_KEY      # optional; name of the env variable holding the key

    The stream is server-sent events: lines "data: {json}", ended by "data: [DONE]".
    """

    def __init__(self, model: str, base_url: str, temperature: float = 0.0,
                 max_tokens: int | None = None, api_key: str | None = None):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def stream(self, messages: list[dict]) -> Iterator[str]:
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "temperature": self.temperature}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        with requests.post(self.url, json=payload, headers=self.headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                choices = json.loads(data).get("choices") or [{}]
                if piece := (choices[0].get("delta") or {}).get("content"):
                    yield piece

    def warmup(self) -> None:
        for _ in self.stream([{"role": "user", "content": "hi"}]):
            pass
        log.info("LLM warmed up: %s", self.model)
