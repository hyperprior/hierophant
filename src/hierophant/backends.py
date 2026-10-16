"""LLM backends. One method to implement; everything else is optional.

Pick one with a spec string, ``"provider:model"``:

    ollama:qwen3:8b             local ollama (http://localhost:11434)
    llamacpp:any                local llama.cpp server (http://localhost:8080)
    lmstudio:qwen3-8b           local LM Studio (http://localhost:1234)
    openai:gpt-5-mini           OpenAI (OPENAI_API_KEY)
    openrouter:some/model       OpenRouter (OPENROUTER_API_KEY)
    anthropic:claude-haiku-4-5  Claude (ANTHROPIC_API_KEY)

or pass any object with a ``complete(messages) -> Response`` method.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class Response:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Caps:
    native_tools: bool = False
    json_mode: bool = False


@runtime_checkable
class Backend(Protocol):
    name: str
    caps: Caps

    def complete(self, messages: list[Message], *, json: bool = False) -> Response: ...


# Base URL and API-key env var for each OpenAI-compatible provider.
_OPENAI_COMPAT = {
    "ollama": ("http://localhost:11434/v1", None),
    "llamacpp": ("http://localhost:8080/v1", None),
    "lmstudio": ("http://localhost:1234/v1", None),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


class OpenAICompat:
    """Anything that speaks POST /chat/completions."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        *,
        temperature: float = 0.0,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ):
        self.name = f"openai-compat:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.caps = Caps(native_tools=True, json_mode=True)
        self.temperature = temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout, headers=headers)

    def complete(self, messages: list[Message], *, json: bool = False) -> Response:
        body = {
            "model": self.model,
            "messages": [asdict(m) for m in messages],
            "temperature": self.temperature,
        }
        if json:
            body["response_format"] = {"type": "json_object"}
        r = self.client.post(f"{self.base_url}/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        return Response(
            text=data["choices"][0]["message"]["content"] or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            raw=data,
        )


class Anthropic:
    """Claude through the Messages API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ):
        self.name = f"anthropic:{model}"
        self.model = model
        self.caps = Caps(native_tools=True, json_mode=False)
        self.max_tokens = max_tokens
        self.temperature = temperature
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = client or httpx.Client(
            timeout=timeout,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )

    def complete(self, messages: list[Message], *, json: bool = False) -> Response:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [asdict(m) for m in messages if m.role != "system"],
        }
        if system:
            body["system"] = system
        r = self.client.post("https://api.anthropic.com/v1/messages", json=body)
        r.raise_for_status()
        data = r.json()
        text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        usage = data.get("usage") or {}
        return Response(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            raw=data,
        )


class Replay:
    """Hands back recorded responses in order. For tests and re-scoring old runs."""

    def __init__(self, responses: list[str | Response] | str | Path):
        if isinstance(responses, str | Path):
            lines = Path(responses).read_text().splitlines()
            responses = [Response(**json.loads(line)) for line in lines if line.strip()]
        self._responses = [r if isinstance(r, Response) else Response(text=r) for r in responses]
        self.name = "replay"
        self.caps = Caps()
        self.seen: list[list[Message]] = []

    def complete(self, messages: list[Message], *, json: bool = False) -> Response:
        self.seen.append(messages)
        if not self._responses:
            raise RuntimeError("replay backend ran out of recorded responses")
        return self._responses.pop(0)


class Recorder:
    """Wraps a backend and appends every response to a JSONL file Replay can read."""

    def __init__(self, inner: Backend, path: str | Path):
        self.inner = inner
        self.path = Path(path)
        self.name = inner.name
        self.caps = inner.caps

    def complete(self, messages: list[Message], *, json: bool = False) -> Response:
        r = self.inner.complete(messages, json=json)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(
                _json_line(
                    {
                        "text": r.text,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                    }
                )
            )
        return r


def backend(spec: str | Backend) -> Backend:
    """Turn ``"provider:model"`` into a backend; pass real backends through."""
    if not isinstance(spec, str):
        return spec
    provider, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"backend spec must look like provider:model, got {spec!r}")
    if provider == "anthropic":
        return Anthropic(model)
    if provider in _OPENAI_COMPAT:
        url, key_env = _OPENAI_COMPAT[provider]
        url = os.environ.get(f"HIEROPHANT_{provider.upper()}_URL", url)
        b = OpenAICompat(model, url, os.environ.get(key_env) if key_env else None)
        b.name = spec
        return b
    raise ValueError(
        f"unknown provider {provider!r}; use one of anthropic, {', '.join(_OPENAI_COMPAT)}"
    )


def _json_line(d: dict) -> str:
    return json.dumps(d) + "\n"
