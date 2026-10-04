"""Reference-core consistency checking: ``check_pair`` without the Rust extension.

Mirrors ``consistent_detection`` semantics — embedding distance gate first,
then structural coverage of the query's *critical* vertices — but executes
the pure-Python reference core (``run_hypermatch``) over bridged hypergraphs.
This is the L2/L3 workhorse: it runs anywhere the ``models`` extra is
installed, with no Rust toolchain.

Coverage quantifies over the production critical-vertex policy (query
placeholders, entity-typed vertices, core grammatical roles, nouns/proper
nouns), not over every matchable vertex: parser-merge artifacts would
otherwise inflate the required set (golden100 calibration, 2026-10-04).
"""

from __future__ import annotations

from typing import Literal

from hufu.core.log import getLogger

logger = getLogger(__name__)

_CRITICAL_DEPS = {"nsubj", "nsubjpass", "dobj", "iobj", "pobj"}


def is_critical_local_vertex(vertex) -> bool:
    """Rust-free copy of the production ``is_critical_vertex`` policy.

    (The canonical copy in ``hyper_simulation`` is wrapped to lazy-load the
    Rust standard path on call, so the reference engine keeps its own.)
    """

    if vertex.is_query():
        return True
    if any(e.name != "NOT_ENTITY" for e in vertex.ents):
        return True
    if vertex.dep().name in _CRITICAL_DEPS:
        return True
    from hufu.hypergraph.linguistic import Pos

    return vertex.pos_equal(Pos.NOUN) or vertex.pos_equal(Pos.PROPN)


def reference_consistent_detection(
    query_text: str,
    data_text: str,
    *,
    distance_threshold: float = 0.25,
    hv_strategy: Literal["typed", "domain", "denial"] = "typed",
    hc_backend=None,
    role_scorer=None,
    query_local=None,
    data_local=None,
    query_is_question: bool = True,
) -> tuple[bool, float, str]:
    """Return ``(has_contradiction, distance, evidence)`` for one text pair.

    The contract matches ``consistent_detection`` except that the embedding
    distance is also returned (the standard path recomputes it in the API
    layer; here one computation serves both the gate and the verdict).
    ``query_local`` / ``data_local`` optionally supply pre-built production
    hypergraphs (e.g. from the benchmark pickle cache) so long documents do
    not have to be re-parsed on every evaluation run.

    ``query_is_question=False`` builds the query side as a declarative claim:
    wh-words inside relative clauses ("the nursery where ...") are then NOT
    mistaken for answer placeholders (finding F8 — declarative refs in the
    ContraDoc harness hit exactly this).
    """

    from hufu.backends import SentenceTransformerBackend

    from .bridge import bridge_allowed_pairs, bridge_hypergraph
    from .build_hypergraph import text_to_hypergraph
    from .embedding import cosine_similarity, get_embedding_batch
    from .hyper_simulation import run_hypermatch

    # Distance is computed here rather than via hyper_simulation.get_distance,
    # which is wrapped to lazy-load the Rust standard path on call.
    query_embedding, data_embedding = get_embedding_batch([query_text, data_text])
    distance = 1.0 - cosine_similarity(query_embedding, data_embedding)
    logger.info(f"Compute cosine distance: {distance:.4f}, threshold={distance_threshold}")
    if distance <= distance_threshold:
        return (
            False,
            distance,
            f"[CONSISTENT] Distance={distance:.4f} ≤ threshold={distance_threshold}",
        )

    query_local = (
        query_local
        if query_local is not None
        else text_to_hypergraph(query_text, is_query=query_is_question)
    )
    data_local = data_local if data_local is not None else text_to_hypergraph(data_text)
    query_core = bridge_hypergraph(query_local, side="query").hypergraph
    data_core = bridge_hypergraph(data_local, side="data").hypergraph

    # Coverage follows the production critical-vertex policy, restricted to
    # vertices that remain matchable in contracts space (a verb can be
    # "critical" in production yet is structurally represented as a
    # predicate here).
    matchable_ids = {v.id for v in query_core.vertices if v.matchable}
    critical = [
        v
        for v in query_local.vertices
        if is_critical_local_vertex(v) and f"v{v.id}" in matchable_ids
    ]
    if not query_core.hyperedges or not critical:
        return (
            False,
            distance,
            f"[CONSISTENT] No critical vertices to cover (distance={distance:.4f})",
        )

    hv_pairs = bridge_allowed_pairs(query_local, data_local, strategy=hv_strategy)
    logger.info(f"h_v allowed pairs ({hv_strategy}): {len(hv_pairs)}")

    # Restrict HC enumeration to whole-component query clusters: 28× fewer
    # candidates on nested queries (measured 426→15), same verdicts on the
    # golden suites.  Falls back to free enumeration if components cannot
    # be formed (disconnected multi-sentence queries degrade gracefully).
    from .bridge import bridge_type_compatible, whole_query_clusters

    try:
        query_clusters = whole_query_clusters(query_core) or None
    except ValueError:
        query_clusters = None

    backend = hc_backend or SentenceTransformerBackend()
    if role_scorer is None:
        from hufu.backends import EmbeddingRoleCellScorer

        # Default stays with the bare-target baseline: on the golden suites it
        # outperforms the production two-view scorer (whose masked structural
        # view is *too* strict for title-laden queries — see working record
        # F9/scorer-calibration).  Pass role_scorer=EmbeddingRoleScorer(backend)
        # to opt into the two-view variant.
        role_scorer = EmbeddingRoleCellScorer(backend)
    from .fixed_point import FixedPointMode

    output = run_hypermatch(
        query_core,
        data_core,
        hv_allowed_pairs=hv_pairs,
        hc_backend=backend,
        role_scorer=role_scorer,
        query_clusters=query_clusters,
        type_compatible=bridge_type_compatible,
        # Existential witness semantics: rival alignments on similar graphs
        # must not annihilate a correct coverage witness (F9).
        fixed_point_mode=FixedPointMode.WITNESS_OR,
    )
    relation = output.fixed_point.relation
    uncovered = [
        vertex for vertex in critical if not any(q == f"v{vertex.id}" for q, _ in relation)
    ]
    if not uncovered:
        if not query_is_question:
            # Declarative claims need a polarity channel: vertex coverage
            # cannot see negation/antonymy ("recognizes" vs "does not
            # recognize").  Cross-encoder NLI over the two claim texts adds
            # it; questions are OOD for the NLI model and skip this gate.
            from .nli import get_nli_label

            nli_label = get_nli_label(query_text, data_text)
            logger.info(f"declarative-pair NLI label: {nli_label}")
            if nli_label == "contradiction":
                evidence = (
                    f"[CONTRADICTION] Distance={distance:.4f} > threshold; "
                    "structural coverage satisfied but NLI polarity conflict "
                    "between the claims"
                )
                logger.info(evidence)
                return True, distance, evidence
        evidence = (
            f"[CONSISTENT] Distance={distance:.4f} > threshold but structural "
            f"coverage satisfied\n  ✓ All {len(critical)} critical Q vertices "
            "matched in D via reference HyperMatch"
        )
        logger.info(evidence)
        return False, distance, evidence

    lines = [
        f"Q vertex unmatched in D: '{vertex.text()}' (ID={vertex.id})"
        for vertex in uncovered
    ]
    evidence = (
        f"[CONTRADICTION] Distance={distance:.4f} > threshold={distance_threshold}\n"
        + "\n".join(f"  • {line}" for line in lines)
    )
    logger.info(evidence)
    return True, distance, evidence
