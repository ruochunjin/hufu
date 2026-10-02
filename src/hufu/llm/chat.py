"""Lazy LangChain-compatible helpers for LLM-assisted graph building.

These functions accept any LangChain chat model but import LangChain only
when called, keeping the dependency optional.  New code should prefer
``hufu.llm.client`` (plain OpenAI-compatible client) instead.
"""

from __future__ import annotations

from typing import Any


def get_generate(prompts: list[str], model: Any) -> list[str]:
    """Generate one response for each prompt in a single model batch."""

    from langchain_core.messages import HumanMessage  # lazy: optional dependency

    messages_list = [[HumanMessage(content=prompt)] for prompt in prompts]
    responses = model.generate(messages_list)
    return [generation[0].text for generation in responses.generations]


def get_invoke(model: Any, text: str, **args) -> str:
    """Invoke a chat model once and return its message content."""

    response = model.invoke(text, **args)
    return response.content


def get_stream(model: Any, text: str, **args):
    """Return the model's streaming response iterator for one prompt."""

    return model.stream(text, **args)
