"""Unified quality-rule model for HuFu.

A *quality rule* has the shape::

    IF <condition> AND <condition> AND ... THEN <verdict>

where each condition applies a registered predicate (logical, statistical,
ML-based, or similarity-based) to a pair of data items.  This schema is the
common core of the three rule systems HuFu integrates:

- **CDR** (conflict detection rules for vector memories):
  ``object_same(s1,s2) ∧ action_conflict(s1,s2) ∧ vec_sim(s1,s2) ≥ ε → conflict``
- **RICO** (document contradiction rules): NLP/temporal/numeric predicates
  over statement pairs → ``contradiction``
- **OMR** (outlier management rules): logical + ML-detector + loss-function
  predicates over training records → ``outlier``

Rules are data (JSON), not code: they can be hand-written, versioned,
shared, or produced by a rule-discovery algorithm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Condition:
    """One predicate application in a rule antecedent.

    ``threshold`` compares the predicate score (default: score >= 0.5).
    ``negate`` flips the boolean outcome.  ``params`` carries
    predicate-specific options (e.g. attribute names).
    """

    predicate: str
    threshold: float | None = None
    negate: bool = False
    params: tuple[tuple[str, Any], ...] = ()

    def params_dict(self) -> dict[str, Any]:
        return dict(self.params)


@dataclass(frozen=True, slots=True)
class Verdict:
    """A rule conclusion: a quality label plus optional confidence."""

    label: str
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class Rule:
    """A complete quality rule with provenance metadata."""

    id: str
    if_: tuple[Condition, ...]
    then: Verdict
    source: str = "manual"  # manual | discovered:<algorithm>
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "if": [
                {
                    "pred": condition.predicate,
                    **({"threshold": condition.threshold} if condition.threshold is not None else {}),
                    **({"negate": True} if condition.negate else {}),
                    **condition.params_dict(),
                }
                for condition in self.if_
            ],
            "then": {"label": self.then.label, "confidence": self.then.confidence},
            "source": self.source,
            "description": self.description,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Rule":
        """Parse one rule from its JSON form (see ``RuleSet``)."""

        conditions = []
        for raw in data["if"]:
            raw = dict(raw)
            predicate = raw.pop("pred")
            threshold = raw.pop("threshold", None)
            negate = bool(raw.pop("negate", False))
            conditions.append(
                Condition(predicate, threshold, negate, tuple(sorted(raw.items())))
            )
        then = data["then"]
        return Rule(
            id=data["id"],
            if_=tuple(conditions),
            then=Verdict(then["label"], float(then.get("confidence", 1.0))),
            source=data.get("source", "manual"),
            description=data.get("description", ""),
        )


@dataclass(frozen=True, slots=True)
class RuleSet:
    """An ordered, versioned collection of rules."""

    name: str
    rules: tuple[Rule, ...]
    version: str = "1"
    metadata: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "metadata": dict(self.metadata),
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RuleSet":
        return RuleSet(
            name=data["name"],
            version=str(data.get("version", "1")),
            metadata=tuple(sorted(data.get("metadata", {}).items())),
            rules=tuple(Rule.from_dict(raw) for raw in data["rules"]),
        )

    @staticmethod
    def load(path: str | Path) -> "RuleSet":
        """Load a rule set from a JSON file."""

        with open(path, encoding="utf-8") as handle:
            return RuleSet.from_dict(json.load(handle))

    def dump(self, path: str | Path) -> None:
        """Write the rule set as JSON."""

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)


__all__ = ["Condition", "Rule", "RuleSet", "Verdict"]
