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

import re
from typing import Literal

from hufu.core.log import getLogger

logger = getLogger(__name__)

_CRITICAL_DEPS = {"nsubj", "nsubjpass", "dobj", "iobj", "pobj"}

# (value, unit) extraction for the declarative numeric rule: first number in
# the text plus the 1-2 word token(s) right after it.
_NUM_UNIT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([A-Za-zÀ-ÿ]+)?")
# SI-style unit families with conversion factors to the base unit.
_UNIT_TABLE = {
    "kilometers": ("m", 1000.0), "kilometer": ("m", 1000.0), "km": ("m", 1000.0),
    "meters": ("m", 1.0), "meter": ("m", 1.0), "m": ("m", 1.0),
    "centimeters": ("m", 0.01), "centimeter": ("m", 0.01), "cm": ("m", 0.01),
    "millimeters": ("m", 0.001), "millimeter": ("m", 0.001), "mm": ("m", 0.001),
    "kilograms": ("g", 1000.0), "kilogram": ("g", 1000.0), "kg": ("g", 1000.0),
    "grams": ("g", 1.0), "gram": ("g", 1.0), "g": ("g", 1.0),
    "milligrams": ("g", 0.001), "milligram": ("g", 0.001), "mg": ("g", 0.001),
    "hours": ("s", 3600.0), "hour": ("s", 3600.0), "h": ("s", 3600.0),
    "minutes": ("s", 60.0), "minute": ("s", 60.0), "min": ("s", 60.0),
    "seconds": ("s", 1.0), "second": ("s", 1.0), "s": ("s", 1.0),
}


def _first_number_unit(text: str) -> tuple[float, str | None] | None:
    """Extract the first (value, unit) pair from a claim text."""

    match = _NUM_UNIT.search(text)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    unit = match.group(2).lower() if match.group(2) else None
    return value, unit


def numeric_conflict(query_text: str, data_text: str) -> bool:
    """Rule-skeleton numeric check for declarative pairs.

    Weak NLI models miss magnitude conflicts (42.195 vs 42195 km) and flag
    unit-converted truths (42.195 km == 42195 m).  When both claims state a
    number with units in the same dimension, compare after conversion with a
    small tolerance; different values mean a genuine numeric conflict.
    Returns False when either side lacks a comparable number+unit (the NLI
    verdict then stands alone).
    """

    left = _first_number_unit(query_text)
    right = _first_number_unit(data_text)
    if left is None or right is None:
        return False
    (lv, lu), (rv, ru) = left, right
    if not lu or not ru:
        return False
    left_dim = _UNIT_TABLE.get(lu)
    right_dim = _UNIT_TABLE.get(ru)
    if left_dim is None or right_dim is None or left_dim[0] != right_dim[0]:
        return False
    lv_base = lv * left_dim[1]
    rv_base = rv * right_dim[1]
    magnitude = max(abs(lv_base), abs(rv_base), 1e-9)
    return abs(lv_base - rv_base) / magnitude > 1e-3


def _numeric_equal(query_text: str, data_text: str) -> bool:
    """True when both claims carry same-dimension numbers that agree after
    unit conversion (e.g. 42.195 km == 42195 m) — NLI's contradiction label
    on such pairs is a false positive the rule overrides (golden250 M46)."""

    left = _first_number_unit(query_text)
    right = _first_number_unit(data_text)
    if left is None or right is None:
        return False
    (lv, lu), (rv, ru) = left, right
    if not lu or not ru:
        return False
    left_dim = _UNIT_TABLE.get(lu)
    right_dim = _UNIT_TABLE.get(ru)
    if left_dim is None or right_dim is None or left_dim[0] != right_dim[0]:
        return False
    lv_base = lv * left_dim[1]
    rv_base = rv * right_dim[1]
    magnitude = max(abs(lv_base), abs(rv_base), 1e-9)
    return abs(lv_base - rv_base) / magnitude <= 1e-3


def declarative_polarity_verdict(
    query_text: str, data_text: str, distance: float
) -> tuple[bool, str] | None:
    """Polarity channel for declarative claim pairs (numeric rule + NLI gate).

    Vertex coverage cannot see negation/antonymy ("recognizes" vs "does not
    recognize").  The numeric rule runs first: it catches magnitude
    conflicts weak NLI models miss and forgives unit-converted truths they
    flag.  Returns ``(has_contradiction, evidence)``, or None when neither
    channel has a verdict (caller keeps its structural verdict).
    """

    if numeric_conflict(query_text, data_text):
        return (
            True,
            f"[CONTRADICTION] Distance={distance:.4f}; "
            "numeric conflict between the claims (rule check)",
        )
    if _numeric_equal(query_text, data_text):
        return (
            False,
            f"[CONSISTENT] Distance={distance:.4f}; "
            "numbers agree after unit conversion (rule check)",
        )
    from .nli import get_nli_label

    nli_label = get_nli_label(query_text, data_text)
    logger.info(f"declarative-pair NLI label: {nli_label}")
    if nli_label == "contradiction":
        return (
            True,
            f"[CONTRADICTION] Distance={distance:.4f}; "
            "structural coverage satisfied but NLI polarity conflict "
            "between the claims",
        )
    return None


def is_critical_local_vertex(vertex) -> bool:
    """Rust-free copy of the production ``is_critical_vertex`` policy.

    (The canonical copy in ``hyper_simulation`` is wrapped to lazy-load the
    Rust standard path on call, so the reference engine keeps its own.)
    """

    if vertex.is_query():
        return True
    if any(e.name != "NOT_ENTITY" for e in vertex.ents):
        return True
    from hufu.hypergraph.linguistic import Pos

    if vertex.pos_equal(Pos.PRON):
        # Pronouns carry no critical constraint of their own: when
        # coreference resolves them, the referred entity vertex is critical
        # in its own right; when it does not, an unmatchable pronoun surface
        # form only fabricates coverage failures (DocNLI id 103: "It had a
        # population ..." flagged solely on 'it').
        return False
    if vertex.dep().name in _CRITICAL_DEPS:
        return True

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
        # No early "consistent" return: conflicts hide in the MOST similar
        # pairs (same frame, one value differs — "weighs 2.3 kg" vs "weighs
        # 4.1 kg" has distance 0.01).  The golden suites pin the structural
        # path with threshold=-1 for exactly this reason; the public gate
        # must not short-circuit it (golden-rag200, 2026-10-08).  Distance
        # still goes into the evidence for callers who pre-filter.
        logger.info("distance within gate; structural check still runs")

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
    # predicate here; likewise merged-span entities may be represented only
    # inside edge structure).  Unmatchable does NOT mean uncoverable —
    # golden100 I01 (identical texts) proved that hard-failing on
    # unmatchable critical vertices mass-regresses the pinned suites.
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
            verdict = declarative_polarity_verdict(query_text, data_text, distance)
            if verdict is not None:
                logger.info(verdict[1])
                return verdict[0], distance, verdict[1]
        evidence = (
            f"[CONSISTENT] Distance={distance:.4f}; structural "
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
