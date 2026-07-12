from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx


def ollama_root(api_base: str) -> str:
    """Normalize an Ollama/OpenAI-style base URL to the Ollama server root."""
    root = api_base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def is_local_ollama(api_base: str) -> bool:
    host = (urlparse(api_base).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


@dataclass
class OllamaEmbeddingClient:
    api_base: str = "http://127.0.0.1:11434"
    model: str = "qwen3-embedding:0.6b"
    timeout: float = 60.0
    api_key: str = ""
    num_gpu: int = 0
    keep_alive: str = "30m"

    @property
    def url(self) -> str:
        return f"{ollama_root(self.api_base)}/api/embed"

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = httpx.post(
            self.url,
            headers=self.headers,
            json={"model": self.model, "input": texts,
                  "keep_alive": self.keep_alive,
                  "options": {"num_gpu": self.num_gpu}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama /api/embed 返回了无效的 embeddings 数组")
        return embeddings

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.url,
                headers=self.headers,
                json={"model": self.model, "input": texts,
                      "keep_alive": self.keep_alive,
                      "options": {"num_gpu": self.num_gpu}},
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama /api/embed 返回了无效的 embeddings 数组")
        return embeddings


class EmbeddingCircuitBreaker:
    """Avoid retrying a stopped Ollama instance on every incoming message."""

    def __init__(self, cooldown: float = 60.0):
        self.cooldown = cooldown
        self.retry_after = 0.0

    def available(self) -> bool:
        return time.monotonic() >= self.retry_after

    def success(self) -> None:
        self.retry_after = 0.0

    def failure(self) -> None:
        self.retry_after = time.monotonic() + self.cooldown
