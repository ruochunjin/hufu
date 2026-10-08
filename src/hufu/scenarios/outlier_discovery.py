"""OLearner-lite: data-driven search over outlier detect→repair pipelines.

OHunt's OLearner runs Bayesian optimization (Optuna TPE) over predicate
switches and thresholds, with the downstream metric as the objective.  The
port keeps the *objective* (measured downstream accuracy after detect →
classify → repair) but replaces TPE with a deterministic grid — the search
space is small (detector θ × repair method), and a grid is reproducible
without an Optuna dependency.  Swap in Optuna later if the space grows.

Everything is injected: the evaluator (train→accuracy callable), the
classifier factory, even the search grid.  Nothing loads sklearn unless the
default estimator is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from hufu.scenarios.detectors import GapIsolationDetector, MADZScoreDetector
from hufu.scenarios.outliers import Record, classify_outliers_hinge, repair_outliers

Evaluator = Callable[[Sequence[Record], Sequence[Any]], float]


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """One detect→repair pipeline configuration."""

    detector: str  # "gap" | "mad"
    theta: float
    repair: str  # repair_outliers method


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Best pipeline plus the full leaderboard."""

    best: PipelineConfig
    best_accuracy: float
    baseline_accuracy: float  # no detection, no repair
    leaderboard: tuple[tuple[PipelineConfig, float], ...]  # best first


def default_grid() -> list[PipelineConfig]:
    """The default search grid (detector families × θ × repair methods)."""

    configs: list[PipelineConfig] = []
    for theta in (5.0, 10.0, 20.0):
        for repair in ("knn_label", "median", "delete"):
            configs.append(PipelineConfig("gap", theta, repair))
    for theta in (2.5, 3.5):
        for repair in ("knn_label", "median", "delete"):
            configs.append(PipelineConfig("mad", theta, repair))
    return configs


def discover_outlier_pipeline(
    records: Sequence[Record],
    labels: Sequence[Any],
    evaluator: Evaluator,
    *,
    grid: Sequence[PipelineConfig] | None = None,
    estimator_factory: Callable[[], Any] | None = None,
) -> DiscoveryResult:
    """Grid-search the detect→classify→repair pipeline by measured accuracy.

    ``evaluator(train_records, train_labels) -> accuracy`` is called once
    for the baseline and once per grid configuration.  For each config the
    pipeline is: detector flags → hinge three-way classification → repair
    the union of ugly+bad records with the configured method → evaluate.
    """

    configs = list(grid) if grid is not None else default_grid()
    baseline = evaluator(records, labels)

    scored: list[tuple[PipelineConfig, float]] = []
    for config in configs:
        if config.detector == "gap":
            detector = GapIsolationDetector(theta=config.theta)
        elif config.detector == "mad":
            detector = MADZScoreDetector(z_threshold=config.theta)
        else:
            raise ValueError(f"unknown detector: {config.detector!r}")
        reports = classify_outliers_hinge(
            records, labels,
            detector=detector, estimator_factory=estimator_factory,
        )
        targets = [r.index for r in reports if r.category in {"ugly", "bad"}]
        outcome = repair_outliers(
            records, labels, targets,
            method=config.repair, estimator_factory=estimator_factory
            if config.repair == "rf_label" else None,
        )
        accuracy = evaluator(list(outcome.records), list(outcome.labels))
        scored.append((config, accuracy))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    best_config, best_accuracy = scored[0]
    return DiscoveryResult(
        best=best_config,
        best_accuracy=best_accuracy,
        baseline_accuracy=baseline,
        leaderboard=tuple(scored),
    )


__all__ = [
    "DiscoveryResult",
    "PipelineConfig",
    "default_grid",
    "discover_outlier_pipeline",
]
