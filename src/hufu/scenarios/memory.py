"""Scenario 1: conflict detection for vector-store memories (CDR-style).

Memories in a vector bank (embodied-agent experience, RAG memory, user
profile facts) can be geometrically adjacent yet semantically conflicting —
"turn on the light" vs. "turn off the light" are close in embedding space
but prescribe opposite actions.  This scenario applies the
``memory_conflicts`` rule set to all memory pairs and reports conflicts
with full explanations.

Attribute enrichment (extracting ``object`` / ``action`` / ``state`` /
``value`` from raw text) is pluggable:

- supply attributes yourself (``attrs={"object": ..., "action": ...}``), or
- let an LLM extract them (needs the ``llm`` extra and an endpoint), or
- rely on the unstructured NLI fallback rule, which needs no attributes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from hufu.rules import Item, RuleEngine, RuleSet

_RULESET_PATH = Path(__file__).parent / "rulesets" / "memory_conflicts.json"


def default_ruleset() -> RuleSet:
    """Load the bundled CDR-style memory-conflict rule set."""

    return RuleSet.load(_RULESET_PATH)


@dataclass(frozen=True, slots=True)
class MemoryConflict:
    """One conflicting pair of memories with rule explanations."""

    index_a: int
    index_b: int
    text_a: str
    text_b: str
    labels: tuple[str, ...]
    explanations: tuple[str, ...]


def extract_attributes_with_llm(texts: Sequence[str]) -> list[dict[str, Any]]:
    """Extract (object, action, state, value) attributes from memory texts.

    Uses the unified OpenAI-compatible client (``HUFU_API_KEY``,
    ``HUFU_BASE_URL``, ``HUFU_MODEL``).  Returns one attribute dict per
    text; unparseable responses degrade to empty dicts (the NLI fallback
    rule still applies).
    """

    from hufu.llm.client import chat

    prompt = (
        "For each memory sentence below, extract structured attributes as a "
        "JSON array with one object per sentence, using the keys "
        '"object", "action", "state", "value". Use null when absent. '
        "Answer with the JSON array only.\n\n"
        + "\n".join(f"{i + 1}. {text}" for i, text in enumerate(texts))
    )
    response = chat([{"role": "user", "content": prompt}])
    try:
        parsed = json.loads(response.content)
        if isinstance(parsed, list):
            return [dict(item) if isinstance(item, dict) else {} for item in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return [{} for _ in texts]


def detect_memory_conflicts(
    memories: Sequence[str | tuple[str, Mapping[str, Any]]],
    *,
    rules: RuleSet | None = None,
    engine: RuleEngine | None = None,
    enrich_with_llm: bool = False,
) -> list[MemoryConflict]:
    """Check all memory pairs for rule-confirmed conflicts.

    Each memory is either raw text or ``(text, attrs)``.  With
    ``enrich_with_llm=True``, missing attributes are extracted via the
    configured LLM endpoint before rule evaluation.
    """

    rule_set = rules if rules is not None else default_ruleset()
    rule_engine = engine if engine is not None else RuleEngine()

    texts = [m if isinstance(m, str) else m[0] for m in memories]
    attrs: list[dict[str, Any]] = [
        {} if isinstance(m, str) else dict(m[1]) for m in memories
    ]
    if enrich_with_llm and any(not attr for attr in attrs):
        extracted = extract_attributes_with_llm(texts)
        attrs = [
            {**found, **given}  # user-supplied attributes win
            for found, given in zip(extracted, attrs)
        ]

    items = [Item(text=text, attrs=attr) for text, attr in zip(texts, attrs)]
    report = rule_engine.check_all_pairs(rule_set, items)

    conflicts: list[MemoryConflict] = []
    for (i, j), matches in sorted(report.items()):
        conflicts.append(
            MemoryConflict(
                index_a=i,
                index_b=j,
                text_a=items[i].text,
                text_b=items[j].text,
                labels=tuple(sorted({match.verdict.label for match in matches})),
                explanations=tuple(match.explanation for match in matches),
            )
        )
    return conflicts


__all__ = [
    "MemoryConflict",
    "default_ruleset",
    "detect_memory_conflicts",
    "extract_attributes_with_llm",
]
