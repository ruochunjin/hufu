"""Ready-made data-quality scenarios built on the HuFu rule layer."""

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
from hufu.scenarios.outliers import (
    OutlierReport,
    apply_repairs,
    classify_outliers,
    outlier_scores,
)
from hufu.scenarios.trim import (
    TrimDecision,
    TrimResult,
    trim_evidence,
    trim_hnsw_edges,
)

__all__ = [
    "Contradiction",
    "MemoryConflict",
    "OutlierReport",
    "Statement",
    "TrimDecision",
    "TrimResult",
    "apply_repairs",
    "classify_outliers",
    "default_ruleset",
    "detect_contradictions",
    "detect_memory_conflicts",
    "outlier_scores",
    "split_statements",
    "trim_evidence",
    "trim_hnsw_edges",
]
