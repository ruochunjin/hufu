"""Scenario 3: contradiction detection within and across documents (RICO-style).

Documents are split into statements; every statement pair (intra-document
and/or cross-document) is checked against the ``doc_contradictions`` rule
set.  Each finding carries **evidence provenance**: the two source
statements, their document/sentence positions, and the per-condition rule
trace — so a contradiction can always be traced back to its origin text.

Statement attributes (``entity``, ``value``, ``polarity``, ``state``) can be
supplied by the caller or extracted with an LLM; without attributes the NLI
fallback rule still works on raw text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from hufu.rules import Item, RuleEngine, RuleSet

_RULESET_PATH = Path(__file__).parent / "rulesets" / "doc_contradictions.json"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def default_ruleset() -> RuleSet:
    """Load the bundled RICO-style contradiction rule set."""

    return RuleSet.load(_RULESET_PATH)


@dataclass(frozen=True, slots=True)
class Statement:
    """One statement with its provenance."""

    text: str
    doc_id: int
    sent_idx: int
    attrs: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Contradiction:
    """A detected contradiction with full evidence provenance."""

    statement_a: Statement
    statement_b: Statement
    scope: str  # "intra" | "inter"
    labels: tuple[str, ...]
    explanations: tuple[str, ...]


def split_statements(
    documents: Sequence[str],
    *,
    attrs: Sequence[Sequence[Mapping[str, Any]]] | None = None,
) -> list[Statement]:
    """Split documents into provenance-tracked statements.

    ``attrs[d][s]`` supplies structured attributes for statement ``s`` of
    document ``d`` (entity, value, polarity, state); lengths must match the
    sentence split.  Callers that need LLM extraction can build these dicts
    with ``hufu.llm.client`` before calling.
    """

    statements: list[Statement] = []
    for doc_id, document in enumerate(documents):
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(document) if s.strip()]
        for sent_idx, sentence in enumerate(sentences):
            attr: Mapping[str, Any] = {}
            if attrs is not None:
                attr = attrs[doc_id][sent_idx]
            statements.append(Statement(sentence, doc_id, sent_idx, attr))
    return statements


def detect_contradictions(
    documents: Sequence[str],
    *,
    attrs: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    rules: RuleSet | None = None,
    engine: RuleEngine | None = None,
    scope: str = "both",
) -> list[Contradiction]:
    """Detect contradictions among statements of the given documents.

    ``scope``: ``"intra"`` (within documents only), ``"inter"`` (across
    documents only), or ``"both"``.
    """

    if scope not in {"intra", "inter", "both"}:
        raise ValueError(f"unknown scope: {scope!r}")
    rule_set = rules if rules is not None else default_ruleset()
    rule_engine = engine if engine is not None else RuleEngine()
    statements = split_statements(documents, attrs=attrs)

    findings: list[Contradiction] = []
    for i in range(len(statements)):
        for j in range(i + 1, len(statements)):
            a, b = statements[i], statements[j]
            pair_scope = "intra" if a.doc_id == b.doc_id else "inter"
            if scope != "both" and pair_scope != scope:
                continue
            matches = rule_engine.check_pair(
                rule_set,
                Item(text=a.text, attrs=a.attrs),
                Item(text=b.text, attrs=b.attrs),
            )
            if matches:
                findings.append(
                    Contradiction(
                        statement_a=a,
                        statement_b=b,
                        scope=pair_scope,
                        labels=tuple(sorted({m.verdict.label for m in matches})),
                        explanations=tuple(m.explanation for m in matches),
                    )
                )
    return findings


__all__ = [
    "Contradiction",
    "Statement",
    "default_ruleset",
    "detect_contradictions",
    "split_statements",
]
