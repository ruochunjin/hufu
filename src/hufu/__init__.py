"""HuFu (虎符): neuro-symbolic data-quality verification for the LLM era.

Named after the ancient Chinese tiger tally — two halves that must match to
verify authenticity — HuFu checks whether pieces of data *fit together*:
retrieved passages vs. queries, memories vs. memories, statements vs.
documents.

The package pairs symbolic structure (linguistic hypergraphs, graph
simulation, quality rules) with neural models used on demand (embeddings,
NLI, optional LLMs), giving explainable, low-cost consistency verdicts.
"""

from hufu.api import (
    FANCY_CONFIG,
    CallableEmbeddingBackend,
    CallableRoleCellScorer,
    Cluster,
    ConsistencyVerdict,
    EmbeddingBackend,
    FancyHyperMatchConfig,
    Hyperedge,
    Hypergraph,
    HyperMatchOutput,
    QueryInstance,
    Role,
    RoleScoringResult,
    Vertex,
    check_pair,
    check_retrieval,
    run_hypermatch,
)
from hufu.backends import SentenceTransformerBackend

__version__ = "0.1.0"

__all__ = [
    "CallableEmbeddingBackend",
    "CallableRoleCellScorer",
    "Cluster",
    "ConsistencyVerdict",
    "EmbeddingBackend",
    "FANCY_CONFIG",
    "FancyHyperMatchConfig",
    "HyperMatchOutput",
    "Hyperedge",
    "Hypergraph",
    "QueryInstance",
    "Role",
    "RoleScoringResult",
    "SentenceTransformerBackend",
    "Vertex",
    "__version__",
    "check_pair",
    "check_retrieval",
    "run_hypermatch",
]
