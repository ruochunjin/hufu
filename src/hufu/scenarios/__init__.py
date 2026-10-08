"""Ready-made data-quality scenarios built on the HuFu rule layer."""

from hufu.scenarios.detectors import (
    GapIsolationDetector,
    MADZScoreDetector,
    OutlierDetector,
    SklearnDetectorAdapter,
)
from hufu.scenarios.documents import (
    Contradiction,
    Statement,
    detect_contradictions,
    split_statements,
)
from hufu.scenarios.memory import (
    MemoryConflict,
    default_ruleset,
    detect_memory_conflicts,
)
from hufu.scenarios.outlier_discovery import (
    DiscoveryResult,
    PipelineConfig,
    discover_outlier_pipeline,
)
from hufu.scenarios.outliers import (
    OutlierReport,
    RepairOutcome,
    apply_repairs,
    classify_outliers,
    classify_outliers_hinge,
    outlier_scores,
    repair_outliers,
)
from hufu.scenarios.trim import (
    TrimDecision,
    TrimResult,
    infer_conflicts_from_judgments,
    trim_evidence,
    trim_hnsw_edges,
)

__all__ = [
    "Contradiction",
    "DiscoveryResult",
    "GapIsolationDetector",
    "MADZScoreDetector",
    "MemoryConflict",
    "OutlierDetector",
    "OutlierReport",
    "PipelineConfig",
    "RepairOutcome",
    "SklearnDetectorAdapter",
    "Statement",
    "TrimDecision",
    "TrimResult",
    "apply_repairs",
    "classify_outliers",
    "classify_outliers_hinge",
    "default_ruleset",
    "detect_contradictions",
    "detect_memory_conflicts",
    "discover_outlier_pipeline",
    "infer_conflicts_from_judgments",
    "outlier_scores",
    "repair_outliers",
    "split_statements",
    "trim_evidence",
    "trim_hnsw_edges",
]
