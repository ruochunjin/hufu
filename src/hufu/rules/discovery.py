"""LLM-guided rule discovery (Learner).

Instead of Apriori-style enumeration, an LLM proposes candidate rules in the
unified DSL from a sample of *labeled* item pairs; candidates are then
validated on the full labeled set with two thresholds (in the spirit of the
DocChecker Learner):

- **coverage**: fraction of positive examples the rule fires on;
- **accuracy**: precision of the rule's verdict over all labeled examples.

Only candidates passing both thresholds are returned, tagged
``source='discovered:llm'``.  The LLM sees predicate names, not raw model
calls — discovery is offline; runtime checking needs no LLM.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from hufu.rules.engine import RuleEngine
from hufu.rules.model import Rule, RuleSet, Verdict
from hufu.rules.registry import DEFAULT_REGISTRY, Item, PredicateRegistry

PROMPT_TEMPLATE = """You are designing data-quality rules in a JSON DSL.

Available predicates (each scores an ordered pair of items):
{predicates}

Items look like: {{"text": "...", "attrs": {{...}}}}.

A rule has the form:
{{"id": "...", "if": [{{"pred": "<name>", "threshold": 0.5, ...params}}],
  "then": {{"label": "{label}"}}}}

Here are labeled example pairs (label 1 = {label}):
{examples}

Propose up to {max_candidates} rules that discriminate label-1 pairs from
label-0 pairs. Answer with a JSON array of rules only."""


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    """Outcome of one discovery run."""

    ruleset: RuleSet
    candidates_total: int
    candidates_accepted: int
    per_rule: tuple[tuple[str, float, float], ...]  # (id, coverage, accuracy)


def _evaluate_candidate(
    engine: RuleEngine,
    rule: Rule,
    positives: Sequence[tuple[Item, Item]],
    negatives: Sequence[tuple[Item, Item]],
) -> tuple[float, float]:
    """Return (coverage on positives, accuracy over all labeled pairs)."""

    def fires(pair: tuple[Item, Item]) -> bool:
        return engine.check_rule(rule, pair[0], pair[1]) is not None

    if not positives:
        return 0.0, 0.0
    tp = sum(1 for pair in positives if fires(pair))
    fp = sum(1 for pair in negatives if fires(pair))
    coverage = tp / len(positives)
    accuracy = tp / (tp + fp) if (tp + fp) else 0.0
    return coverage, accuracy


def discover_rules(
    positives: Sequence[tuple[Item, Item]],
    negatives: Sequence[tuple[Item, Item]],
    *,
    label: str = "conflict",
    registry: PredicateRegistry = DEFAULT_REGISTRY,
    engine: RuleEngine | None = None,
    chat_fn: Callable[[str], str] | None = None,
    min_coverage: float = 0.2,
    min_accuracy: float = 0.8,
    max_candidates: int = 5,
    sample_size: int = 10,
    seed: int = 42,
    name: str = "discovered",
) -> DiscoveryReport:
    """Discover a rule set separating positive from negative pairs.

    ``chat_fn`` receives the prompt and returns the LLM's raw text; the
    default uses ``hufu.llm.client.chat`` (env-configured endpoint).  Pass a
    stub for testing or offline candidate generation.
    """

    if chat_fn is None:
        from hufu.llm.client import chat

        def chat_fn(prompt: str) -> str:  # type: ignore[misc]
            return chat([{"role": "user", "content": prompt}]).content

    rng = random.Random(seed)
    sample_pos = rng.sample(list(positives), min(sample_size, len(positives)))
    sample_neg = rng.sample(list(negatives), min(sample_size, len(negatives)))

    def render(pair: tuple[Item, Item], y: int) -> str:
        a, b = pair
        return json.dumps(
            {"a": {"text": a.text, "attrs": dict(a.attrs)},
             "b": {"text": b.text, "attrs": dict(b.attrs)}, "label": y},
            ensure_ascii=False,
        )

    examples = "\n".join(
        [render(pair, 1) for pair in sample_pos]
        + [render(pair, 0) for pair in sample_neg]
    )
    prompt = PROMPT_TEMPLATE.format(
        predicates="\n".join(f"- {name}" for name in registry.names()),
        label=label,
        examples=examples,
        max_candidates=max_candidates,
    )
    raw = chat_fn(prompt)

    try:
        proposals = json.loads(raw)
    except json.JSONDecodeError:
        # Tolerate prose around the JSON array.
        start, end = raw.find("["), raw.rfind("]")
        proposals = json.loads(raw[start : end + 1]) if start >= 0 else []

    rule_engine = engine if engine is not None else RuleEngine(registry=registry)
    accepted: list[Rule] = []
    stats: list[tuple[str, float, float]] = []
    for i, proposal in enumerate(proposals[:max_candidates]):
        try:
            candidate = Rule.from_dict(
                {**proposal, "then": {**proposal.get("then", {}), "label": label}}
            )
        except (KeyError, TypeError, ValueError):
            continue
        candidate = Rule(
            id=f"{name}.{i}:{candidate.id}",
            if_=candidate.if_,
            then=Verdict(label, candidate.then.confidence),
            source="discovered:llm",
            description=candidate.description,
        )
        try:
            coverage, accuracy = _evaluate_candidate(
                rule_engine, candidate, positives, negatives
            )
        except KeyError:
            continue  # unknown predicate names are rejected here
        if coverage >= min_coverage and accuracy >= min_accuracy:
            accepted.append(candidate)
            stats.append((candidate.id, coverage, accuracy))

    ruleset = RuleSet(
        name=name,
        rules=tuple(accepted),
        metadata=(
            ("discovery", "llm"),
            ("min_coverage", min_coverage),
            ("min_accuracy", min_accuracy),
            ("positives", len(positives)),
            ("negatives", len(negatives)),
        ),
    )
    return DiscoveryReport(
        ruleset=ruleset,
        candidates_total=len(proposals),
        candidates_accepted=len(accepted),
        per_rule=tuple(stats),
    )


__all__ = ["DiscoveryReport", "discover_rules"]
