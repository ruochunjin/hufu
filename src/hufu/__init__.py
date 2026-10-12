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
from hufu.rules import (
    Condition,
    Item,
    PredicateRegistry,
    Rule,
    RuleEngine,
    RuleMatch,
    RuleSet,
    Verdict,
    predicate,
)
from hufu.scenarios import detect_memory_conflicts

__version__ = "0.5.4"

__all__ = [
    "CallableEmbeddingBackend",
    "CallableRoleCellScorer",
    "Cluster",
    "Condition",
    "ConsistencyVerdict",
    "EmbeddingBackend",
    "FANCY_CONFIG",
    "FancyHyperMatchConfig",
    "HyperMatchOutput",
    "Hyperedge",
    "Hypergraph",
    "Item",
    "PredicateRegistry",
    "QueryInstance",
    "Role",
    "RoleScoringResult",
    "Rule",
    "RuleEngine",
    "RuleMatch",
    "RuleSet",
    "SentenceTransformerBackend",
    "Verdict",
    "Vertex",
    "__version__",
    "check_pair",
    "check_retrieval",
    "detect_memory_conflicts",
    "predicate",
    "run_hypermatch",
]
