from __future__ import annotations

import openai
from threading import Lock
from typing import Any

from llm4ad.base import LLM


class OpenAIAPI(LLM):
    def __init__(self, base_url: str, api_key: str, model: str, timeout=60, **kwargs):
        super().__init__()
        self._model = model
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)
        self._usage_lock = Lock()
        self._token_usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @staticmethod
    def _usage_value(usage: Any, name: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return int(usage.get(name) or 0)
        return int(getattr(usage, name, 0) or 0)

    def _record_usage(self, usage: Any) -> None:
        with self._usage_lock:
            self._token_usage["requests"] += 1
            self._token_usage["prompt_tokens"] += self._usage_value(usage, "prompt_tokens")
            self._token_usage["completion_tokens"] += self._usage_value(usage, "completion_tokens")
            self._token_usage["total_tokens"] += self._usage_value(usage, "total_tokens")

    def token_usage(self) -> dict[str, int]:
        with self._usage_lock:
            return dict(self._token_usage)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=prompt,
            stream=False,
            **kwargs,
        )
        self._record_usage(getattr(response, "usage", None))
        return response.choices[0].message.content
