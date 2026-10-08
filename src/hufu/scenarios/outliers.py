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


# ---------------------------------------------------------------------------
# OHunt port: hinge-loss three-way classification (paper-exact semantics)
# ---------------------------------------------------------------------------


def _decision_scores(estimator: Any, matrix: list[list[float]]) -> list[list[float]]:
    """Per-class decision scores as an (n, k) nested list.

    Binary estimators return one score per row; it is expanded to
    ``[-f, f]`` so the multiclass hinge formula applies uniformly.
    """

    raw = estimator.decision_function(matrix)
    rows: list[list[float]] = []
    for entry in raw:
        try:
            row = [float(v) for v in entry]  # (k,) row
        except TypeError:
            row = [-float(entry), float(entry)]  # scalar (binary)
        rows.append(row)
    return rows


def _hinge_losses(scores: list[list[float]], labels: Sequence[Any]) -> list[float]:
    """Multiclass hinge per sample: max_j≠y max(0, 1 - f_y + f_j)."""

    classes = sorted({str(label) for label in labels})
    losses = []
    for row, label in zip(scores, labels):
        y = classes.index(str(label))
        f_y = row[y]
        loss = 0.0
        for j, f_j in enumerate(row):
            if j != y:
                loss = max(loss, 1.0 - f_y + f_j)
        losses.append(loss)
    return losses


def classify_outliers_hinge(
    records: Sequence[Record],
    labels: Sequence[Any],
    *,
    detector: Any | None = None,
    estimator_factory: Callable[[], Any] | None = None,
) -> list[OutlierReport]:
    """OHunt-exact good/bad/ugly classification (hinge three-way ∩ detector).

    Trains the downstream classifier (default: linear SVM, injected via
    ``estimator_factory`` for any sklearn-compatible estimator) and computes
    each record's multiclass hinge loss.  Records the detector flags are:

    - hinge == 0  → **good** (outlying but classified with full margin);
    - 0 < hinge ≤ 1 → **ugly** (inside the margin; repairing helps);
    - hinge > 1 → **bad** (misclassified; erroneous and damaging).

    Unflagged records are ``normal``.  This differs from
    :func:`classify_outliers` (measured accuracy impact): the hinge variant
    is the paper's rule predicate `loss(M,D,t)` with λ=1, computed without
    retraining per record — much cheaper, but tied to the margin geometry of
    one fitted model.
    """

    if len(records) != len(labels):
        raise ValueError("records and labels must have the same length")
    if detector is None:
        from hufu.scenarios.detectors import MADZScoreDetector

        detector = MADZScoreDetector()
    if estimator_factory is None:
        def estimator_factory() -> Any:  # noqa: B023 - lazy sklearn
            from sklearn.svm import SVC

            return SVC(kernel="linear")

    from hufu.scenarios.detectors import numeric_features

    features = numeric_features(records)
    matrix = [[float(r[key]) for key in features] for r in records]
    estimator = estimator_factory()
    estimator.fit(matrix, [str(label) for label in labels])
    losses = _hinge_losses(_decision_scores(estimator, matrix), labels)

    scores = detector.score(records)
    flagged = detector.flag(records)
    reports: list[OutlierReport] = []
    eps = 1e-9
    for i, record in enumerate(records):
        if not flagged[i]:
            reports.append(OutlierReport(i, "normal", scores[i], None, None))
            continue
        hinge = losses[i]
        if hinge <= eps:
            category = "good"
        elif hinge <= 1.0 + eps:
            category = "ugly"
        else:
            category = "bad"
        reports.append(OutlierReport(i, category, scores[i], hinge, None))
    return reports


# ---------------------------------------------------------------------------
# OHunt port: repair family (KNN label repair won the paper's experiments)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    """Repaired dataset plus what changed."""

    records: tuple[Record, ...]
    labels: tuple[Any, ...]
    repaired_labels: tuple[int, ...]  # indices whose label changed
    repaired_values: tuple[int, ...]  # indices whose features changed
    deleted: tuple[int, ...]
    method: str


def repair_outliers(
    records: Sequence[Record],
    labels: Sequence[Any],
    targets: Sequence[int],
    *,
    method: str = "knn_label",
    k: int = 5,
    estimator_factory: Callable[[], Any] | None = None,
) -> RepairOutcome:
    """Repair (or delete) the target records.

    Methods (OHunt's repair family, ranked by their measured accuracy gain):

    - ``knn_label`` — re-label targets with a KNN classifier trained on the
      remaining records (the paper's best-performing repair);
    - ``rf_label`` — same idea with a random forest (needs
      ``estimator_factory`` or sklearn);
    - ``mean`` / ``median`` — replace target numeric features with the
      column statistic of the remaining records;
    - ``knn_features`` — impute target features from the k nearest remaining
      records (uniform vote per feature, stdlib);
    - ``delete`` — drop the targets entirely.

    ``targets`` are row indices (e.g. the ugly+bad records from
    classification).  Returns the full dataset with repairs applied.
    """

    if len(records) != len(labels):
        raise ValueError("records and labels must have the same length")
    target_set = set(targets)
    if not target_set:
        return RepairOutcome(
            tuple(records), tuple(labels), (), (), (), method
        )
    for i in target_set:
        if not 0 <= i < len(records):
            raise IndexError(f"target index {i} out of range for {len(records)} records")

    from hufu.scenarios.detectors import numeric_features

    features = numeric_features(records)
    keep = [i for i in range(len(records)) if i not in target_set]
    keep_matrix = [[float(records[i][key]) for key in features] for i in keep]
    keep_labels = [str(labels[i]) for i in keep]

    new_records: list[Record] = [dict(r) for r in records]
    new_labels: list[Any] = list(labels)
    relabeled: list[int] = []
    revalued: list[int] = []
    deleted: list[int] = []

    if method == "delete":
        deleted = sorted(target_set)
        new_records = [dict(records[i]) for i in keep]
        new_labels = [labels[i] for i in keep]
        return RepairOutcome(
            tuple(new_records), tuple(new_labels), (), (), tuple(deleted), method
        )

    if method in {"knn_label", "rf_label"}:
        if estimator_factory is not None:
            estimator = estimator_factory()
        elif method == "knn_label":
            from sklearn.neighbors import KNeighborsClassifier

            estimator = KNeighborsClassifier(n_neighbors=max(1, min(k, len(keep))))
        else:
            from sklearn.ensemble import RandomForestClassifier

            estimator = RandomForestClassifier(n_estimators=100, random_state=0)
        estimator.fit(keep_matrix, keep_labels)
        for i in sorted(target_set):
            row = [[float(records[i][key]) for key in features]]
            predicted = estimator.predict(row)[0]
            if str(labels[i]) != str(predicted):
                relabeled.append(i)
            new_labels[i] = predicted
        return RepairOutcome(
            tuple(new_records), tuple(new_labels),
            tuple(relabeled), (), (), method,
        )

    if method in {"mean", "median"}:
        stats: dict[str, float] = {}
        for key in features:
            column = sorted(float(records[i][key]) for i in keep)
            mid = len(column) // 2
            if method == "mean":
                stats[key] = sum(column) / len(column)
            else:
                stats[key] = (
                    column[mid]
                    if len(column) % 2
                    else (column[mid - 1] + column[mid]) / 2
                )
        for i in sorted(target_set):
            updated = dict(new_records[i])
            for key in features:
                updated[key] = stats[key]
            new_records[i] = updated
            revalued.append(i)
        return RepairOutcome(
            tuple(new_records), tuple(labels), (), tuple(revalued), (), method
        )

    if method == "knn_features":
        for i in sorted(target_set):
            row = [float(records[i][key]) for key in features]
            neighbors = sorted(
                keep,
                key=lambda j: sum(
                    (float(records[j][key]) - row[k]) ** 2
                    for k, key in enumerate(features)
                ),
            )[: max(1, k)]
            updated = dict(new_records[i])
            for k_idx, key in enumerate(features):
                updated[key] = sum(
                    float(records[j][key]) for j in neighbors
                ) / len(neighbors)
            new_records[i] = updated
            revalued.append(i)
        return RepairOutcome(
            tuple(new_records), tuple(labels), (), tuple(revalued), (), method
        )

    raise ValueError(f"unknown repair method: {method!r}")


__all__ = [
    "OutlierReport",
    "RepairOutcome",
    "apply_repairs",
    "classify_outliers",
    "classify_outliers_hinge",
    "outlier_scores",
    "repair_outliers",
]
