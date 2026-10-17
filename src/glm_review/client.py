"""Small, dependency-free client for an OpenAI-compatible model server."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClientConfig:
    base_url: str = "http://127.0.0.1:8000"
    model: str = "glm-5.3"
    api_key: str | None = None
    timeout: int = 1800

    @classmethod
    def from_env(cls) -> "ClientConfig":
        return cls(
            base_url=os.getenv("GLM_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("GLM_MODEL", cls.model),
            api_key=os.getenv("GLM_API_KEY"),
            timeout=int(os.getenv("GLM_TIMEOUT", str(cls.timeout))),
        )


class ModelServerError(RuntimeError):
    """Raised when the model server returns a malformed or failed response."""


class GLMClient:
    def __init__(self, config: ClientConfig | None = None):
        self.config = config or ClientConfig.from_env()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        n: int = 1,
        temperature: float = 0.3,
        max_tokens: int = 16_384,
        reasoning_effort: str = "max",
        enable_thinking: bool = True,
    ) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
                "reasoning_effort": reasoning_effort,
            },
        }

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> list[str]:
        payload = self.build_payload(messages, **kwargs)
        request = urllib.request.Request(
            f"{self.config.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise ModelServerError(f"model request failed: {exc}") from exc

        try:
            return [choice["message"]["content"] or "" for choice in body["choices"]]
        except (KeyError, TypeError) as exc:
            raise ModelServerError("response did not contain OpenAI-compatible choices") from exc

    def is_ready(self) -> bool:
        request = urllib.request.Request(
            f"{self.config.base_url}/v1/models", headers=self._headers()
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def wait_until_ready(self, *, poll_seconds: int = 30, max_wait: int = 3600) -> None:
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            if self.is_ready():
                return
            time.sleep(poll_seconds)
        raise TimeoutError(f"model server was not ready after {max_wait} seconds")
