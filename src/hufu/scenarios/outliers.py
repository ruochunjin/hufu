"""Scenario 4: training-data outlier management — the good, the bad, the ugly.

Not every outlier is harmful (the OHunt insight):

- **good** — new-distribution data; removing/repairing it *hurts* the model;
- **bad** — erroneous but harmless; the model barely notices;
- **ugly** — erroneous and damaging; repairing it measurably helps.

HuFu classifies flagged outliers by their measured effect on a downstream
model, instead of deleting everything that looks unusual.  The downstream
model is injected by the caller (any classifier); only records whose repair
changes accuracy beyond ``gain_threshold`` are acted on.  Ugly outliers are
*repaired* (median imputation per feature), never silently deleted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

Record = Mapping[str, Any]
Evaluator = Callable[[Sequence[Record], Sequence[Any]], float]


@dataclass(frozen=True, slots=True)
class OutlierReport:
    """Classification of one record."""

    index: int
    category: str  # "normal" | "good" | "bad" | "ugly"
    outlier_score: float
    accuracy_delta: float | None  # acc(repaired) - acc(baseline); None if unevaluated
    repaired: Record | None


def _numeric_features(records: Sequence[Record]) -> list[str]:
    if not records:
        return []
    return [
        key
        for key, value in records[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def outlier_scores(
    records: Sequence[Record], *, z_threshold: float = 3.0
) -> list[float]:
    """Robust per-record outlier score: max |z| over numeric features.

    Uses median/MAD (robust to the outliers themselves); a record scoring
    above ``z_threshold`` is considered an outlier.  Discrete features often
    have MAD 0 — those fall back to the (population) standard deviation, so
    binary columns never produce infinite scores.  Pure stdlib.
    """

    features = _numeric_features(records)
    if not features:
        return [0.0 for _ in records]
    stats: dict[str, tuple[float, float]] = {}
    for key in features:
        values = sorted(float(r[key]) for r in records)
        mid = len(values) // 2
        median = (
            values[mid]
            if len(values) % 2
            else (values[mid - 1] + values[mid]) / 2
        )
        deviations = sorted(abs(v - median) for v in values)
        mad = (
            deviations[mid]
            if len(deviations) % 2
            else (deviations[mid - 1] + deviations[mid]) / 2
        )
        if mad > 0.0:
            # 0.6745 scales MAD to standard-deviation units.
            scale = 0.6745 * mad
        else:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            scale = math.sqrt(variance)
        stats[key] = (median, scale)
    scores = []
    for record in records:
        worst = 0.0
        for key in features:
            median, scale = stats[key]
            if scale > 0.0:
                z = abs(float(record[key]) - median) / scale
                worst = max(worst, z)
        scores.append(worst)
    return scores


def _median_repair(records: Sequence[Record], index: int) -> dict[str, Any]:
    """Replace numeric features of one record with column medians."""

    features = _numeric_features(records)
    repaired = dict(records[index])
    for key in features:
        values = sorted(
            float(r[key]) for i, r in enumerate(records) if i != index
        )
        mid = len(values) // 2
        median = (
            values[mid]
            if len(values) % 2
            else (values[mid - 1] + values[mid]) / 2
        )
        repaired[key] = median
    return repaired


def classify_outliers(
    records: Sequence[Record],
    labels: Sequence[Any],
    evaluator: Evaluator,
    *,
    z_threshold: float = 3.0,
    gain_threshold: float = 0.01,
    max_evals: int | None = None,
) -> list[OutlierReport]:
    """Classify outlying records as good / bad / ugly by measured impact.

    ``evaluator(train_records, train_labels) -> accuracy`` trains and scores
    the downstream model; it is called once on the full data (baseline) and
    once per flagged outlier with that record repaired.  Deltas:

    - ``delta >= gain_threshold`` → **ugly**: repair helped → repaired copy
      attached to the report;
    - ``delta <= -gain_threshold`` → **good**: repair hurt → keep as-is;
    - otherwise → **bad**: negligible impact → leave alone.

    ``max_evals`` caps evaluator calls (largest outlier scores first).
    """

    if len(records) != len(labels):
        raise ValueError("records and labels must have the same length")
    scores = outlier_scores(records, z_threshold=z_threshold)
    baseline = evaluator(records, labels)

    flagged = [i for i, s in enumerate(scores) if s >= z_threshold]
    flagged.sort(key=lambda i: scores[i], reverse=True)
    if max_evals is not None:
        flagged = flagged[:max_evals]
    flagged_set = set(flagged)

    reports: list[OutlierReport] = []
    for i, record in enumerate(records):
        if i not in flagged_set:
            reports.append(OutlierReport(i, "normal", scores[i], None, None))
            continue
        repaired = _median_repair(records, i)
        candidate = [
            repaired if j == i else records[j] for j in range(len(records))
        ]
        delta = evaluator(candidate, labels) - baseline
        if delta >= gain_threshold:
            category, fixed = "ugly", repaired
        elif delta <= -gain_threshold:
            category, fixed = "good", None
        else:
            category, fixed = "bad", None
        reports.append(OutlierReport(i, category, scores[i], delta, fixed))
    return reports


def apply_repairs(
    records: Sequence[Record], reports: Sequence[OutlierReport]
) -> list[Record]:
    """Materialize the repaired training set (only ugly records change)."""

    repaired: list[Record] = list(records)
    for report in reports:
        if report.repaired is not None:
            repaired[report.index] = report.repaired
    return repaired


__all__ = [
    "OutlierReport",
    "apply_repairs",
    "classify_outliers",
    "outlier_scores",
]
