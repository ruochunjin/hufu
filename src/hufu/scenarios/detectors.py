"""Outlier detector protocol and stdlib implementations (OHunt port).

The OHunt rule predicates need an ``outlier(D, R, t.A, θ)`` detector.  The
paper's core definition is **gap-based isolation**: after sorting a column,
a value isolated from BOTH nearest neighbors by at least θ is an outlier
(endpoints check one side).  This module provides that detector plus the
robust MAD z-score, behind a small protocol so sklearn/PyOD/DeepOD-style
detectors (``fit``/``decision_function``) can plug in via an adapter.

Pure stdlib; no third-party imports anywhere in this module.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

Record = Mapping[str, Any]


def numeric_features(records: Sequence[Record]) -> list[str]:
    """Numeric feature keys, in sorted order for determinism."""

    if not records:
        return []
    return sorted(
        key
        for key, value in records[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


@runtime_checkable
class OutlierDetector(Protocol):
    """Detector protocol: score every record; larger = more outlying."""

    def score(self, records: Sequence[Record]) -> list[float]:
        ...

    def flag(self, records: Sequence[Record]) -> list[bool]:
        ...


class GapIsolationDetector:
    """OHunt's gap-isolation predicate with gap-of-gaps normalization.

    For each numeric feature, a record's isolation is the smaller of its
    gaps to the left/right sorted neighbors (single-sided at endpoints).
    The column scale is the MEDIAN consecutive gap — not the column span or
    MAD, which cluster structure inflates (bimodal columns would mask every
    genuine outlier; golden test `test_gap_isolation_flags_distant_point`).
    Score = isolation / median-gap; records scoring at least ``theta`` are
    flagged.  Note the documented boundary: a tight *cluster* of outliers is
    internally dense and invisible to gap isolation.
    """

    def __init__(self, theta: float = 10.0) -> None:
        if theta <= 0:
            raise ValueError("theta must be positive")
        self.theta = theta

    def score(self, records: Sequence[Record]) -> list[float]:
        features = numeric_features(records)
        if not features:
            return [0.0 for _ in records]
        scores = [0.0] * len(records)
        for key in features:
            column = [float(r[key]) for r in records]
            ordered = sorted(column)
            consecutive = [
                hi - lo for lo, hi in zip(ordered, ordered[1:]) if hi > lo
            ]
            if not consecutive:
                continue
            consecutive.sort()
            mid = len(consecutive) // 2
            median_gap = (
                consecutive[mid]
                if len(consecutive) % 2
                else (consecutive[mid - 1] + consecutive[mid]) / 2
            )
            if median_gap <= 0.0:
                continue
            order = sorted(range(len(column)), key=lambda i: column[i])
            for rank, i in enumerate(order):
                gaps = []
                if rank > 0:
                    gaps.append(column[i] - column[order[rank - 1]])
                if rank < len(order) - 1:
                    gaps.append(column[order[rank + 1]] - column[i])
                if gaps:
                    isolation = min(gaps) / median_gap
                    scores[i] = max(scores[i], isolation)
        return scores

    def flag(self, records: Sequence[Record]) -> list[bool]:
        return [s >= self.theta for s in self.score(records)]


class MADZScoreDetector:
    """Robust median/MAD z-score detector (HuFu's v0.4 default, rehomed).

    Score = max |modified z| over numeric features (MAD scaled to stddev
    units; MAD-0 columns fall back to population stddev).  Records scoring
    at least ``z_threshold`` are flagged.
    """

    def __init__(self, z_threshold: float = 3.0) -> None:
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        self.z_threshold = z_threshold

    def score(self, records: Sequence[Record]) -> list[float]:
        # Reuse the battle-tested implementation rather than duplicating.
        from hufu.scenarios.outliers import outlier_scores

        return outlier_scores(records, z_threshold=0.0)

    def flag(self, records: Sequence[Record]) -> list[bool]:
        return [s >= self.z_threshold for s in self.score(records)]


class SklearnDetectorAdapter:
    """Wrap an sklearn/PyOD-style detector into the protocol.

    The wrapped object must offer ``fit(X)`` and ``decision_function(X)``;
    ``threshold`` gives the flag cut on the (possibly negated) decision
    scores.  X is the numeric feature matrix in sorted-key order.
    """

    def __init__(self, estimator: Any, threshold: float) -> None:
        self.estimator = estimator
        self.threshold = threshold

    @staticmethod
    def _matrix(records: Sequence[Record]) -> list[list[float]]:
        features = numeric_features(records)
        return [[float(r[key]) for key in features] for r in records]

    def score(self, records: Sequence[Record]) -> list[float]:
        matrix = self._matrix(records)
        self.estimator.fit(matrix)
        return [float(s) for s in self.estimator.decision_function(matrix)]

    def flag(self, records: Sequence[Record]) -> list[bool]:
        return [s >= self.threshold for s in self.score(records)]


__all__ = [
    "GapIsolationDetector",
    "MADZScoreDetector",
    "OutlierDetector",
    "SklearnDetectorAdapter",
    "numeric_features",
]
