"""Threshold calibration for similarity predicates (CDRs ROC/Youden port).

The CDR work learns the VecSim threshold offline from labeled positive and
negative pair scores by maximizing Youden's J (tpr - fpr).  This module is
the stdlib rewrite: no sklearn, no numpy — scores in, threshold out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ThresholdReport:
    """Outcome of Youden-J threshold learning."""

    threshold: float
    tpr: float
    fpr: float
    j_statistic: float
    n_positive: int
    n_negative: int


def learn_threshold_youden(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> ThresholdReport:
    """Learn the similarity threshold separating positive from negative pairs.

    A candidate threshold θ labels a pair positive when ``score > θ``
    (CDRs' VecSim semantics).  Candidates are midpoints between adjacent
    unique scores (plus the open ends); the winner maximizes Youden's J.
    Ties break toward the *larger* θ (precision-leaning: fewer false
    conflicts).
    """

    pos = sorted(float(s) for s in positive_scores)
    neg = sorted(float(s) for s in negative_scores)
    if not pos or not neg:
        raise ValueError("need at least one positive and one negative score")

    candidates = sorted(set(pos + neg))
    midpoints = [
        (lo + hi) / 2 for lo, hi in zip(candidates, candidates[1:])
    ]
    sweep = [candidates[0] - 1e-9, *midpoints, candidates[-1] + 1e-9]

    best: ThresholdReport | None = None
    for theta in sweep:
        tp = sum(1 for s in pos if s > theta)
        fp = sum(1 for s in neg if s > theta)
        tpr = tp / len(pos)
        fpr = fp / len(neg)
        j = tpr - fpr
        if best is None or j > best.j_statistic or (
            j == best.j_statistic and theta > best.threshold
        ):
            best = ThresholdReport(theta, tpr, fpr, j, len(pos), len(neg))
    assert best is not None  # sweep is never empty
    return best


__all__ = ["ThresholdReport", "learn_threshold_youden"]
