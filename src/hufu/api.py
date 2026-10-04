"""HuFu public API.

Two usage levels are provided:

1. **Reference core (dependency-free)** — ``run_hypermatch`` over typed
   ``contracts`` hypergraphs with caller-injected backends.  This is the
   verified, deterministic pipeline; see ``tests/test_engine_core.py`` for
   complete examples that need no model packages.

2. **Text-level consistency checking** — ``check_pair`` / ``check_retrieval``
   build linguistic hypergraphs from raw text (spaCy + coreference) and run
   the production solver (Rust-accelerated fixed point).  These require the
   optional ``models`` extra and the ``simulation-alg`` Rust extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hufu.core import QueryInstance
from hufu.engine.config import FANCY_CONFIG, FancyHyperMatchConfig
from hufu.engine.contracts import (
    Cluster,
    Hyperedge,
    Hypergraph,
    Role,
    Vertex,
)
from hufu.engine.d_match import CallableRoleCellScorer, RoleScoringResult
from hufu.engine.hyper_simulation import HyperMatchOutput, run_hypermatch
from hufu.engine.scoring import CallableEmbeddingBackend, EmbeddingBackend


@dataclass(frozen=True, slots=True)
class ConsistencyVerdict:
    """Outcome of checking one query–document pair."""

    query: str
    document: str
    consistent: bool
    distance: float | None
    evidence: str


def check_pair(
    query_text: str,
    document_text: str,
    *,
    distance_threshold: float = 0.25,
    use_gpu: bool = False,
    engine: str = "standard",
    query_is_question: bool = True,
) -> ConsistencyVerdict:
    """Check whether one retrieved document is consistent with the query.

    High cosine similarity does not imply semantic consistency: this runs the
    hyper-simulation structural check and returns an explainable verdict.

    ``engine='standard'`` requires the ``models`` extra (spaCy
    ``en_core_web_trf``, fastcoref, sentence-transformers) and the
    ``simulation-alg`` Rust extension.  ``engine='reference'`` runs the
    pure-Python reference core over bridged hypergraphs — no Rust toolchain
    needed — and is validated against the standard path by the L3 dual-track
    tests.

    ``query_is_question=False`` treats the query as a declarative claim
    (e.g. a sentence to be checked against the document) instead of an
    interrogative; wh-words inside relative clauses are then not mistaken
    for answer placeholders.  Reference engine only.
    """

    if engine == "reference":
        from hufu.engine.reference_check import reference_consistent_detection

        has_contradiction, distance, evidence = reference_consistent_detection(
            query_text,
            document_text,
            distance_threshold=distance_threshold,
            query_is_question=query_is_question,
        )
        return ConsistencyVerdict(
            query=query_text,
            document=document_text,
            consistent=not has_contradiction,
            distance=distance,
            evidence=evidence,
        )
    if engine != "standard":
        raise ValueError(f"unknown engine {engine!r}; expected 'standard' or 'reference'")

    from hufu.engine.build_hypergraph import get_nlp, text_to_hypergraph
    from hufu.engine.embedding import cosine_similarity, get_embedding_batch
    from hufu.engine.hyper_simulation import consistent_detection

    get_nlp(use_gpu=use_gpu)  # fail fast with a clear message if spaCy is missing

    try:
        import simulation  # noqa: F401 - Rust fixed-point extension
    except ImportError as exc:  # pragma: no cover - depends on local build
        raise ImportError(
            "check_pair requires the Rust extension 'simulation-alg'. "
            "Build it with: pip install ./rust/simulation (needs a Rust toolchain)."
        ) from exc

    query_hg = text_to_hypergraph(query_text, is_query=True)
    data_hg = text_to_hypergraph(document_text)

    instance = QueryInstance(query=query_text, data=[document_text], answers=[], ground_truth=[])
    has_contradiction, evidence = consistent_detection(
        instance,
        query_hg,
        data_hg,
        query_text,
        document_text,
        distance_threshold=distance_threshold,
    )
    q_emb, d_emb = get_embedding_batch([query_text, document_text])
    distance = 1.0 - cosine_similarity(q_emb, d_emb)
    return ConsistencyVerdict(
        query=query_text,
        document=document_text,
        consistent=not has_contradiction,
        distance=distance,
        evidence=evidence,
    )


def check_retrieval(
    query_text: str,
    documents: Sequence[str],
    *,
    distance_threshold: float = 0.25,
    use_gpu: bool = False,
    engine: str = "standard",
) -> list[ConsistencyVerdict]:
    """Check every retrieved document against the query, in input order.

    Use this as a post-checker between vector retrieval and generation in any
    RAG pipeline: documents flagged ``consistent=False`` are candidates for
    filtering, repair, or down-weighting.
    """

    verdicts: list[ConsistencyVerdict] = []
    for document in documents:
        verdicts.append(
            check_pair(
                query_text,
                document,
                distance_threshold=distance_threshold,
                use_gpu=use_gpu,
                engine=engine,
            )
        )
    return verdicts


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
    "Vertex",
    "check_pair",
    "check_retrieval",
    "run_hypermatch",
]
