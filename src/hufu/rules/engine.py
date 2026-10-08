"""Rule evaluation engine.

The engine is deliberately small and deterministic: every condition is
evaluated once per item pair, every fired rule returns a full trace (each
condition's score and pass/fail), so verdicts are explainable by
construction.  No rule ordering, no inference surprises — rules are
declarative checks over pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from hufu.rules.model import Condition, Rule, RuleSet, Verdict
from hufu.rules.registry import DEFAULT_REGISTRY, Item, PredicateRegistry, Services

DEFAULT_THRESHOLD = 0.5


class DefaultServices:
    """``Services`` implementation over HuFu's lazy model helpers.

    Models load on first use (``HUFU_EMBED_MODEL`` / ``HUFU_NLI_MODEL``);
    nothing is imported eagerly.
    """

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        from hufu.engine.embedding import get_embedding_batch

        return [vector.tolist() for vector in get_embedding_batch(list(texts))]

    def nli_label(self, premise: str, hypothesis: str) -> str:
        from hufu.engine.nli import get_nli_label

        return get_nli_label(premise, hypothesis)

    def nli_score(self, premise: str, hypothesis: str, label: str) -> float:
        # Softmax probabilities, NOT raw logits: rule thresholds (e.g. 0.8)
        # are probabilities; cross-encoder logits (3.5+) pass any threshold
        # and made the NLI fallback fire on nearly everything (golden-mem
        # attribution, 2026-10-08).
        from hufu.engine.nli import (
            get_nli_contradiction_probs_batch,
            get_nli_entailment_probs_batch,
        )

        if label == "entailment":
            return float(get_nli_entailment_probs_batch([(premise, hypothesis)])[0])
        if label == "contradiction":
            return float(get_nli_contradiction_probs_batch([(premise, hypothesis)])[0])
        raise ValueError(f"unsupported NLI label score: {label!r}")


@dataclass(frozen=True, slots=True)
class ConditionTrace:
    """How one condition evaluated for a pair."""

    condition: Condition
    score: float
    passed: bool


@dataclass(frozen=True, slots=True)
class RuleMatch:
    """A fired rule with its full evaluation trace."""

    rule: Rule
    verdict: Verdict
    traces: tuple[ConditionTrace, ...]

    @property
    def explanation(self) -> str:
        parts = [
            f"{trace.condition.predicate}={trace.score:.3f}"
            f"({'pass' if trace.passed else 'fail'})"
            for trace in self.traces
        ]
        return f"{self.rule.id} → {self.verdict.label} [{', '.join(parts)}]"


class RuleEngine:
    """Evaluate rule sets over pairs of :class:`Item`."""

    def __init__(
        self,
        registry: PredicateRegistry = DEFAULT_REGISTRY,
        services: Services | None = None,
    ) -> None:
        self._registry = registry
        self._services = services if services is not None else DefaultServices()

    def evaluate_condition(
        self, condition: Condition, a: Item, b: Item
    ) -> ConditionTrace:
        fn = self._registry.get(condition.predicate)
        score = float(fn(a, b, condition.params_dict(), self._services))
        threshold = condition.threshold if condition.threshold is not None else DEFAULT_THRESHOLD
        passed = score >= threshold
        if condition.negate:
            passed = not passed
        return ConditionTrace(condition, score, passed)

    def check_rule(self, rule: Rule, a: Item, b: Item) -> RuleMatch | None:
        """Return a match when every condition of ``rule`` fires, else None."""

        traces = tuple(self.evaluate_condition(c, a, b) for c in rule.if_)
        if not traces or not all(trace.passed for trace in traces):
            return None
        return RuleMatch(rule, rule.then, traces)

    def check_pair(
        self, rules: RuleSet | Iterable[Rule], a: Item, b: Item
    ) -> list[RuleMatch]:
        """Evaluate all rules against one ordered pair."""

        rule_list = rules.rules if isinstance(rules, RuleSet) else tuple(rules)
        matches = []
        for rule in rule_list:
            match = self.check_rule(rule, a, b)
            if match is not None:
                matches.append(match)
        return matches

    def check_all_pairs(
        self, rules: RuleSet | Iterable[Rule], items: Sequence[Item]
    ) -> dict[tuple[int, int], list[RuleMatch]]:
        """Evaluate rules on every ordered pair ``(i, j)`` with ``i < j``.

        Quality rules over pairs are symmetric in intent (conflict,
        contradiction); evaluating ``i < j`` avoids duplicate reports.
        """

        report: dict[tuple[int, int], list[RuleMatch]] = {}
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                matches = self.check_pair(rules, items[i], items[j])
                if matches:
                    report[(i, j)] = matches
        return report


__all__ = [
    "ConditionTrace",
    "DefaultServices",
    "RuleEngine",
    "RuleMatch",
]
