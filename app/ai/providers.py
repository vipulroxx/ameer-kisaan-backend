"""LLM and embedding providers behind one small interface.

Swap providers with LLM_PROVIDER / EMBEDDINGS_PROVIDER: "ollama" runs models
locally in docker-compose; "openai" and "azure_openai" are for cloud
deployments; "fake" is deterministic and used by the tests.
"""
import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx

from app.core.config import Settings, get_settings


class ProviderUnavailable(RuntimeError):
    pass


class ProviderRateLimited(ProviderUnavailable):
    """HTTP 429 from the provider; retry_after is its suggested wait in seconds."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class LLM(ABC):
    name: str

    @abstractmethod
    def chat(self, system: str, user: str) -> str: ...


class Embedder(ABC):
    name: str

    @property
    def cache_id(self) -> str:
        """Identifies the vector space: cached vectors are only reused for the same id."""
        return self.name

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _post(url: str, payload: dict, headers: dict | None = None, timeout: float = 60.0) -> dict:
    try:
        r = httpx.post(url, json=payload, headers=headers or {}, timeout=timeout)
        if r.status_code == 429:
            raise ProviderRateLimited("AI provider rate limit reached", _retry_after(r))
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        raise ProviderUnavailable(f"AI provider request failed: {type(exc).__name__}") from exc


def _retry_after(r: httpx.Response) -> float:
    """Seconds to wait, from Retry-After or Google's RetryInfo ("retryDelay": "17s")."""
    try:
        return min(float(r.headers["retry-after"]), 120.0)
    except (KeyError, ValueError):
        pass
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', r.text or "")
    return min(float(m.group(1)), 120.0) if m else 20.0


# --- Ollama -----------------------------------------------------------------

class OllamaLLM(LLM):
    name = "ollama"

    def __init__(self, s: Settings):
        self.s = s

    def chat(self, system: str, user: str) -> str:
        data = _post(f"{self.s.ollama_base_url}/api/chat", {
            "model": self.s.ollama_model, "stream": False, "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, timeout=self.s.llm_timeout_seconds)
        return data["message"]["content"]


class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(self, s: Settings):
        self.s = s

    @property
    def cache_id(self) -> str:
        return f"ollama:{self.s.ollama_embed_model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = _post(f"{self.s.ollama_base_url}/api/embed", {"model": self.s.ollama_embed_model, "input": texts},
                     timeout=self.s.llm_timeout_seconds)
        return data["embeddings"]


# --- OpenAI / Azure OpenAI --------------------------------------------------

class OpenAILLM(LLM):
    name = "openai"

    def __init__(self, s: Settings, azure: bool = False):
        self.s, self.azure = s, azure
        if azure and not (s.azure_openai_endpoint and s.azure_openai_api_key and s.azure_openai_chat_deployment):
            raise ProviderUnavailable("Azure OpenAI is not configured")
        if not azure and not s.openai_api_key:
            raise ProviderUnavailable("OpenAI is not configured")

    def chat(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.azure:
            url = (f"{self.s.azure_openai_endpoint.rstrip('/')}/openai/deployments/"
                   f"{self.s.azure_openai_chat_deployment}/chat/completions?api-version={self.s.azure_openai_api_version}")
            data = _post(url, {"messages": msgs, "temperature": 0.2}, {"api-key": self.s.azure_openai_api_key},
                         self.s.llm_timeout_seconds)
        else:
            data = _post(f"{self.s.openai_base_url}/chat/completions",
                         {"model": self.s.openai_model, "messages": msgs, "temperature": 0.2},
                         {"Authorization": f"Bearer {self.s.openai_api_key}"}, self.s.llm_timeout_seconds)
        return data["choices"][0]["message"]["content"]


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, s: Settings, azure: bool = False):
        self.s, self.azure = s, azure
        if azure and not (s.azure_openai_endpoint and s.azure_openai_api_key and s.azure_openai_embed_deployment):
            raise ProviderUnavailable("Azure OpenAI embeddings are not configured")
        self.base_url = (s.openai_embed_base_url or s.openai_base_url).rstrip("/")
        self._cache_id = (f"azure:{s.azure_openai_endpoint}:{s.azure_openai_embed_deployment}" if azure
                          else f"openai:{self.base_url}:{s.openai_embed_model}")
        self.api_key = s.openai_embed_api_key or s.openai_api_key
        if not azure and not self.api_key:
            raise ProviderUnavailable("OpenAI is not configured")

    @property
    def cache_id(self) -> str:
        return self._cache_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.azure:
            url = (f"{self.s.azure_openai_endpoint.rstrip('/')}/openai/deployments/"
                   f"{self.s.azure_openai_embed_deployment}/embeddings?api-version={self.s.azure_openai_api_version}")
            data = _post(url, {"input": texts}, {"api-key": self.s.azure_openai_api_key}, self.s.llm_timeout_seconds)
        else:
            data = _post(f"{self.base_url}/embeddings", {"model": self.s.openai_embed_model, "input": texts},
                         {"Authorization": f"Bearer {self.api_key}"}, self.s.llm_timeout_seconds)
        # OpenAI returns an "index" per item; some compatible APIs (Gemini) omit it and keep input order.
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0))]


# --- Fake (tests / offline) -------------------------------------------------

class FakeEmbedder(Embedder):
    """Hashed bag-of-words vectors: deterministic, offline, good enough for tests."""
    name = "fake"
    dim = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z]{3,}", t.lower()):
                v[int(hashlib.md5(w.encode(), usedforsecurity=False).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FakeLLM(LLM):
    name = "fake"

    def chat(self, system: str, user: str) -> str:
        first = re.search(r"<source id=\"1\"[^>]*>\s*(.+?)\s*</source>", user, re.S)
        return f"Based on the guides [1]: {first.group(1)[:200]}" if first else "I don't know based on the available guides."


@lru_cache
def get_llm() -> LLM:
    s = get_settings()
    return {"ollama": lambda: OllamaLLM(s), "openai": lambda: OpenAILLM(s),
            "azure_openai": lambda: OpenAILLM(s, azure=True), "fake": FakeLLM}[s.llm_provider]()


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    return {"ollama": lambda: OllamaEmbedder(s), "openai": lambda: OpenAIEmbedder(s),
            "azure_openai": lambda: OpenAIEmbedder(s, azure=True), "fake": FakeEmbedder}[s.embeddings_provider]()
