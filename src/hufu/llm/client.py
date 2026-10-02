"""OpenAI-compatible chat client shared by all HuFu LLM-assisted features.

Configuration is entirely environment-driven; nothing is created at import
time and no global client instance is kept:

- ``HUFU_API_KEY`` / ``OPENAI_API_KEY``: API key (required on first call)
- ``HUFU_BASE_URL`` / ``OPENAI_BASE_URL``: endpoint, defaults to OpenAI;
  point it at a local vLLM/Ollama server for offline use
- ``HUFU_MODEL``: default model name, defaults to ``gpt-4o-mini``
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """One chat-completion response."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _resolve(name: str, fallback: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or os.environ.get(fallback) or default


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ChatResponse:
    """Call an OpenAI-compatible Chat Completions endpoint once.

    The ``openai`` package is imported lazily so that HuFu's rule/graph
    engine remains usable without any LLM dependency installed.
    """

    from openai import OpenAI  # lazy: optional dependency

    client = OpenAI(
        api_key=api_key or _resolve("HUFU_API_KEY", "OPENAI_API_KEY"),
        base_url=base_url or _resolve("HUFU_BASE_URL", "OPENAI_BASE_URL"),
    )
    model_name = model or _resolve("HUFU_MODEL", "OPENAI_MODEL", "gpt-4o-mini")
    kwargs: dict = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    usage = response.usage
    return ChatResponse(
        content=response.choices[0].message.content or "",
        model=model_name,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
    )


def complete(prompt: str, **kwargs) -> str:
    """Send a single user-turn prompt and return the response text."""

    return chat([{"role": "user", "content": prompt}], **kwargs).content
