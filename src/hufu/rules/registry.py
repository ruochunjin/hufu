"""Predicate registry: the pluggable vocabulary of the rule language.

A predicate scores a pair of data items.  Predicates are plain callables
registered by name; rule conditions reference them through the registry, so
users can extend the language without touching the engine::

    from hufu.rules import predicate

    @predicate("same_city")
    def same_city(a, b, params, services):
        return 1.0 if a.attrs.get("city") == b.attrs.get("city") else 0.0

``Item`` is the unit of data; ``Services`` gives predicates lazy access to
neural models (embeddings, NLI) without importing them at module load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class Item:
    """One data record under inspection.

    ``attrs`` carries structured fields (object, action, timestamp, numeric
    values, ...); ``embedding`` is an optional precomputed vector.  Rules
    stay agnostic to where these came from (a vector DB, a document store,
    a training set loader).
    """

    text: str
    attrs: Mapping[str, Any] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None


class Services(Protocol):
    """Lazy neural-model access handed to predicates by the engine."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode texts into L2-normalized dense vectors."""
        ...

    def nli_label(self, premise: str, hypothesis: str) -> str:
        """Return 'entailment' | 'contradiction' | 'neutral'."""
        ...

    def nli_score(self, premise: str, hypothesis: str, label: str) -> float:
        """Return the raw NLI score for one label."""
        ...


PredicateFn = Callable[[Item, Item, Mapping[str, Any], Services], float]


class PredicateRegistry:
    """Name → predicate mapping with duplicate-registration protection."""

    def __init__(self) -> None:
        self._predicates: dict[str, PredicateFn] = {}

    def register(self, name: str, fn: PredicateFn, *, replace: bool = False) -> None:
        if not name:
            raise ValueError("predicate name must not be empty")
        if name in self._predicates and not replace:
            raise ValueError(f"predicate {name!r} is already registered")
        self._predicates[name] = fn

    def get(self, name: str) -> PredicateFn:
        try:
            return self._predicates[name]
        except KeyError:
            raise KeyError(
                f"unknown predicate {name!r}; available: {sorted(self._predicates)}"
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._predicates))

    def __contains__(self, name: str) -> bool:
        return name in self._predicates

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


# The process-wide default registry, pre-populated by hufu.rules.predicates.
DEFAULT_REGISTRY = PredicateRegistry()


def predicate(name: str, *, registry: PredicateRegistry = DEFAULT_REGISTRY):
    """Decorator registering a predicate function under ``name``."""

    def wrap(fn: PredicateFn) -> PredicateFn:
        registry.register(name, fn)
        return fn

    return wrap


__all__ = [
    "DEFAULT_REGISTRY",
    "Item",
    "PredicateFn",
    "PredicateRegistry",
    "Services",
    "predicate",
]
