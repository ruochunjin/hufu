"""Bridge production LocalHypergraph into the typed contracts.Hypergraph.

The reference core (``run_hypermatch``) consumes typed, dependency-free
contracts, while the production build pipeline produces spaCy-backed
``LocalHypergraph`` objects.  This module is the only sanctioned conversion
point between the two worlds; keeping it isolated lets the L3 dual-track
tests prove that both execution paths reach identical verdicts.

Mapping decisions follow the original author's clarification (2026-10-03):
vertex labels come from the ``Vertex.is_*`` / ``Vertex.type()`` method family,
``kind='query'`` comes from ``Vertex.is_query()``, and hyperedge roles are
the dependency labels of the participant nodes — no invented vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hufu.core.log import getLogger
from hufu.hypergraph.hypergraph import Hypergraph as LocalHypergraph
from hufu.hypergraph.hypergraph import Vertex as LocalVertex

from .contracts import Hyperedge, Hypergraph, Role, Vertex

logger = getLogger(__name__)

Side = Literal["query", "data"]

# Entity types (ENT names) that classify a vertex as a concrete value rather
# than a referential entity.
_VALUE_TYPES = {"NUMBER", "TEMPORAL"}
_EVENT_TYPES = {"EVENT"}


def local_vertex_kind(vertex: LocalVertex, *, is_edge_root: bool = False) -> str:
    """Map a production vertex onto a contracts VertexKind.

    Priority follows the author's method-family rule.  ``is_edge_root`` forces
    ``predicate`` because contracts require every hyperedge predicate to be a
    predicate-kind vertex even when the root is nominal (copular clauses).
    """

    if is_edge_root:
        return "predicate"
    if vertex.is_query():
        return "query"
    if vertex.is_verb():
        return "predicate"
    if vertex.is_adjective():
        return "adjective"
    if vertex.is_adverb():
        return "adverb"
    ent_type = vertex.type()
    if ent_type is not None:
        if ent_type.name in _VALUE_TYPES:
            return "value"
        if ent_type.name in _EVENT_TYPES:
            return "event"
    return "entity"


def local_semantic_type(vertex: LocalVertex) -> str:
    """Coarse type string used by feature_type / type_compatible."""

    ent_type = vertex.type()
    if ent_type is not None:
        return ent_type.name
    if vertex.poses:
        return vertex.poses[0].name
    return "UNKNOWN"


def local_expected_type(vertex: LocalVertex) -> str | None:
    """Expected answer type for query placeholders (QueryType name)."""

    query_type = vertex.query_type()
    return query_type.name if query_type is not None else None


@dataclass(frozen=True)
class BridgeResult:
    """Converted hypergraph plus diagnostic id mappings."""

    hypergraph: Hypergraph
    local_to_core_vertex: dict[int, str]
    dropped_edges: int


def bridge_hypergraph(local: LocalHypergraph, *, side: Side) -> BridgeResult:
    """Convert one production hypergraph into a validated contracts graph."""

    root_ids = {edge.root.id for edge in local.hyperedges}
    vertices: list[Vertex] = []
    local_to_core: dict[int, str] = {}
    for local_vertex in local.vertices:
        core_id = f"v{local_vertex.id}"
        local_to_core[local_vertex.id] = core_id
        vertices.append(
            Vertex(
                id=core_id,
                text=local_vertex.text(),
                semantic_type=local_semantic_type(local_vertex),
                kind=local_vertex_kind(
                    local_vertex, is_edge_root=local_vertex.id in root_ids
                ),
                expected_type=local_expected_type(local_vertex),
            )
        )

    father_index: dict[int, int] = {}
    edges: list[Hyperedge] = []
    dropped = 0
    for index, local_edge in enumerate(local.hyperedges):
        roles: list[Role] = []
        seen_incidences: set[tuple[str, str]] = set()
        for participant in local_edge.vertices:
            if participant == local_edge.root:
                continue
            node = local_edge.current_node(participant)
            incidence = (node.dep.name, local_to_core[participant.id])
            if incidence in seen_incidences:
                continue
            seen_incidences.add(incidence)
            roles.append(Role(name=incidence[0], vertex_id=incidence[1]))
        if not roles:
            # Contracts reject root-only pseudo-facts; the builder should not
            # emit them, but fused graphs can, so drop defensively.
            dropped += 1
            logger.debug(f"dropping root-only hyperedge: {local_edge.desc!r}")
            continue
        edge_index = len(edges)
        father_index[id(local_edge)] = edge_index
        context_id = f"{side}-ctx-{local_edge.hypergraph_id or 0}"
        father_id = None
        if local_edge.father is not None:
            parent = father_index.get(id(local_edge.father))
            if parent is not None:
                father_id = f"e{parent}"
        edges.append(
            Hyperedge(
                id=f"e{edge_index}",
                predicate_id=local_to_core[local_edge.root.id],
                roles=tuple(roles),
                context_id=context_id,
                father_id=father_id,
            )
        )

    core = Hypergraph(vertices=tuple(vertices), hyperedges=tuple(edges), side=side)
    return BridgeResult(
        hypergraph=core,
        local_to_core_vertex=local_to_core,
        dropped_edges=dropped,
    )


def bridge_allowed_pairs(
    query: LocalHypergraph,
    data: LocalHypergraph,
    *,
    strategy: Literal["typed", "domain", "denial"] = "typed",
) -> set[tuple[str, str]]:
    """Produce h_v candidate pairs in contracts id space.

    ``typed`` (default) mirrors the reference core's own gate: matchable
    kinds only, ``default_type_compatible`` on feature types — so query
    placeholders pair through their expected answer type (``?Where`` matches
    GPE/LOC data vertices).  ``domain`` reuses the production
    ``Vertex.is_domain``/``ent_same`` rules, whose NER-asymmetry clause
    (NER vs no-NER returns False) misses pairs like ``author`` ↔
    ``Ernest Hemingway``.  ``denial`` delegates to the production denial rule
    (NLI model fallback).  The L3 dual-track tests compare the three before
    one is frozen as the default.
    """

    query_vertices = {v.id: v for v in query.vertices}
    data_vertices = {v.id: v for v in data.vertices}
    if strategy == "denial":
        from .denial import compute_allowed_pairs

        raw = compute_allowed_pairs(query_vertices, data_vertices)
        return {(f"v{q_id}", f"v{d_id}") for q_id, d_id in raw}
    if strategy == "domain":
        return {
            (f"v{q_id}", f"v{d_id}")
            for q_id, q_vertex in query_vertices.items()
            for d_id, d_vertex in data_vertices.items()
            if q_vertex.is_domain(d_vertex) or q_vertex.ent_same(d_vertex)
        }
    from .d_match import default_type_compatible

    query_core = bridge_hypergraph(query, side="query").hypergraph
    data_core = bridge_hypergraph(data, side="data").hypergraph
    return {
        (qv.id, dv.id)
        for qv in query_core.vertices
        if qv.matchable
        for dv in data_core.vertices
        if dv.matchable and default_type_compatible(qv, dv)
    }
