"""Reference-core consistency checking: ``check_pair`` without the Rust extension.

Mirrors ``consistent_detection`` semantics — embedding distance gate first,
then structural coverage of the query's matchable vertices — but executes the
pure-Python reference core (``run_hypermatch``) over bridged hypergraphs.
This is the L2/L3 workhorse: it runs anywhere the ``models`` extra is
installed, with no Rust toolchain.
"""

from __future__ import annotations

from typing import Literal

from hufu.core.log import getLogger

logger = getLogger(__name__)


def reference_consistent_detection(
    query_text: str,
    data_text: str,
    *,
    distance_threshold: float = 0.25,
    hv_strategy: Literal["typed", "domain", "denial"] = "typed",
    hc_backend=None,
    role_scorer=None,
) -> tuple[bool, float, str]:
    """Return ``(has_contradiction, distance, evidence)`` for one text pair.

    The contract matches ``consistent_detection`` except that the embedding
    distance is also returned (the standard path recomputes it in the API
    layer; here one computation serves both the gate and the verdict).
    """

    from hufu.backends import EmbeddingRoleCellScorer, SentenceTransformerBackend

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

    query_local = text_to_hypergraph(query_text, is_query=True)
    data_local = text_to_hypergraph(data_text)
    query_core = bridge_hypergraph(query_local, side="query").hypergraph
    data_core = bridge_hypergraph(data_local, side="data").hypergraph

    required = tuple(v for v in query_core.vertices if v.matchable)
    if not query_core.hyperedges or not required:
        return (
            False,
            distance,
            f"[CONSISTENT] No structural query requirements (distance={distance:.4f})",
        )

    hv_pairs = bridge_allowed_pairs(query_local, data_local, strategy=hv_strategy)
    logger.info(f"h_v allowed pairs ({hv_strategy}): {len(hv_pairs)}")

    output = run_hypermatch(
        query_core,
        data_core,
        hv_allowed_pairs=hv_pairs,
        hc_backend=hc_backend or SentenceTransformerBackend(),
        role_scorer=role_scorer or EmbeddingRoleCellScorer(),
    )
    consistency = output.query_consistency
    if consistency.consistent:
        evidence = (
            f"[CONSISTENT] Distance={distance:.4f} > threshold but structural "
            f"coverage satisfied\n  ✓ All {len(consistency.covered_query_vertex_ids)} "
            "matchable Q vertices covered via reference HyperMatch"
        )
        logger.info(evidence)
        return False, distance, evidence

    lines = []
    for vertex_id in consistency.uncovered_query_vertex_ids:
        vertex = query_core.vertex(vertex_id)
        lines.append(f"Q vertex unmatched in D: '{vertex.text}' (ID={vertex_id})")
    evidence = (
        f"[CONTRADICTION] Distance={distance:.4f} > threshold={distance_threshold}\n"
        + "\n".join(f"  • {line}" for line in lines)
    )
    logger.info(evidence)
    return True, distance, evidence
