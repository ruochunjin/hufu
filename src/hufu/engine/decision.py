"""Explainable decision chain for consistency judgments.

Every verdict the engine produces carries a ``DecisionChain``: the ordered
steps from graph construction through rule firing to the final call, each
with a human-readable summary and structured detail.  Users can ask the tool
*why* a document was judged contradictory (or clean) and get the full
decision path, not just a boolean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DecisionStep:
    """One stage of the decision path."""

    stage: str  # "build" | "stars" | "rule" | "confirm" | "verdict"
    summary: str
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionChain:
    """Ordered, immutable record of how a verdict was reached."""

    steps: tuple[DecisionStep, ...]

    def explain(self, *, lang: str = "en") -> str:
        """Render the chain as a numbered, human-readable explanation."""

        labels = {
            "en": {
                "build": "Build document graph",
                "stars": "Find entity stars",
                "rule": "Rule candidates",
                "confirm": "Confirmation",
                "verdict": "Verdict",
            },
            "zh": {
                "build": "构建文档图",
                "stars": "提取实体星型结构",
                "rule": "规则候选",
                "confirm": "确认判定",
                "verdict": "最终判定",
            },
        }[lang]
        lines = []
        for index, step in enumerate(self.steps, start=1):
            lines.append(f"{index}. [{labels.get(step.stage, step.stage)}] {step.summary}")
        return "\n".join(lines)


class ChainBuilder:
    """Mutable collector frozen into a ``DecisionChain`` at the end."""

    def __init__(self) -> None:
        self._steps: list[DecisionStep] = []

    def add(self, stage: str, summary: str, **detail: Any) -> None:
        self._steps.append(DecisionStep(stage, summary, detail))

    def freeze(self) -> DecisionChain:
        return DecisionChain(tuple(self._steps))


__all__ = ["ChainBuilder", "DecisionChain", "DecisionStep"]
