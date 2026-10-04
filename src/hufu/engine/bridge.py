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

import re
from dataclasses import dataclass
from typing import Literal

from hufu.core.log import getLogger
from hufu.hypergraph.hypergraph import Hypergraph as LocalHypergraph
from hufu.hypergraph.hypergraph import Vertex as LocalVertex

from .contracts import Hyperedge, Hypergraph, Role, Vertex

logger = getLogger(__name__)

# Leading digit run ("8 planets", "1,000,000 residents", "42.") — merged
# quantity phrases lose their NUM token in retokenization, so the digit test
# on the surface text reproduces the production ``pos_range(Pos.NUM)`` rule.
_NUMBER_PREFIX = re.compile(r"^\d[\d,.\s]*")


def _contains_number(vertex: LocalVertex) -> bool:
    """Whether the vertex carries a quantity (NUM node or digit-led text)."""

    return any(p.name == "NUM" for p in vertex.poses) or bool(
        _NUMBER_PREFIX.match(vertex.text())
    )

Side = Literal["query", "data"]

# Entity types (ENT names) that classify a vertex as a concrete value rather
# than a referential entity.
_VALUE_TYPES = {"NUMBER", "TEMPORAL"}
_EVENT_TYPES = {"EVENT"}


def local_vertex_kind(vertex: LocalVertex, *, is_edge_root: bool = False) -> str:
    """Map a production vertex onto a contracts VertexKind.

    Priority follows the author's method-family rule.  On the QUERY side a
    root is forced to ``predicate`` (contracts require predicate-kind edge
    heads; and query-side nominals becoming matchable bloat the critical
    coverage set — A/B regression on golden100).  On the DATA side
    ``bridge_hypergraph`` instead synthesizes an implicit predicate for
    non-verb roots, keeping the nominal matchable: data vertices only serve
    as match targets, so there is no coverage bloat (fixes H05/L07 where the
    answer phrase was a nominal root and thus unmatchable).
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
    if _contains_number(vertex):
        return "value"
    return "entity"


def local_semantic_type(vertex: LocalVertex) -> str:
    """Coarse type string used by feature_type / type_compatible.

    A quantity-bearing vertex ("8 planets") reports NUMBER even when its head
    noun carries an entity type, mirroring the production ``pos_range(NUM)``
    acceptance in the denial rule.  TEMPORAL/NUMBER entity types win over the
    digit heuristic so dates never become quantities.
    """

    ent_type = vertex.type()
    if ent_type is not None and ent_type.name in _VALUE_TYPES:
        return ent_type.name
    if _contains_number(vertex):
        return "NUMBER"
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
    """Convert one production hypergraph into a validated contracts graph.

    Root handling is asymmetric by side (golden100 A/B, 2026-10-04): query
    roots are forced to predicate kind; on the data side a non-verb root
    instead gets a synthesized implicit predicate vertex and joins the roles
    as the head participant, so answer phrases that head nominal relations
    ("42 chapters", "ceiling of the chapel") stay matchable.
    """

    root_ids = {edge.root.id for edge in local.hyperedges}
    synthesize_root = side == "data"
    vertices: list[Vertex] = []
    local_to_core: dict[int, str] = {}
    for local_vertex in local.vertices:
        core_id = f"v{local_vertex.id}"
        local_to_core[local_vertex.id] = core_id
        is_root = local_vertex.id in root_ids
        natural_kind = local_vertex_kind(local_vertex, is_edge_root=False)
        kind = (
            "predicate"
            if is_root and not (synthesize_root and natural_kind != "predicate")
            else natural_kind
        )
        vertices.append(
            Vertex(
                id=core_id,
                text=local_vertex.text(),
                semantic_type=local_semantic_type(local_vertex),
                kind=kind,
                expected_type=local_expected_type(local_vertex),
            )
        )
    kinds = {v.id: v.kind for v in vertices}

    father_index: dict[int, int] = {}
    edges: list[Hyperedge] = []
    dropped = 0
    for index, local_edge in enumerate(local.hyperedges):
        roles: list[Role] = []
        seen_incidences: set[tuple[str, str]] = set()
        root_core_id = local_to_core[local_edge.root.id]
        predicate_id = root_core_id
        if synthesize_root and kinds[root_core_id] != "predicate":
            # Nominal data root: implicit predicate, root joins as head role.
            predicate_id = f"e{len(edges)}-pred"
            root_node = local_edge.current_node(local_edge.root)
            head_role = root_node.dep.name or "head"
            incidence = (head_role, root_core_id)
            if incidence not in seen_incidences:
                seen_incidences.add(incidence)
                roles.append(Role(name=head_role, vertex_id=root_core_id))
            vertices.append(
                Vertex(
                    id=predicate_id,
                    text=local_edge.root.text(),
                    semantic_type="PREDICATE",
                    kind="predicate",
                )
            )
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
                predicate_id=predicate_id,
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


def bridge_type_compatible(qv: Vertex, dv: Vertex) -> bool:
    """Type-compatibility gate mirroring the production denial query branch.

    ``default_type_compatible`` requires feature_type equality or a shared
    TYPE_GROUPS family, which leaves several QueryType values unreachable:
    ``WHAT``/``WHICH`` have no data-side feature_type at all, so a typed hv
    built on the default gate can never match them (golden100 E-family).
    The production ``denial_comment`` query branch defines the intended
    acceptance sets; this gate encodes them in contracts space:

    - PERSON → AGENT family, or untyped nouns (NER misses like "José
      Saramago" fall back to PROPN; the production rule then asks NLI, which
      is lenient — approximate that with acceptance)
    - BELONGS → AGENT/ORGANIZATION/LOCATION families
    - WHAT / WHICH / REASON / untyped placeholder → any matchable data vertex
    - ATTRIBUTE → adjective/adverb kinds

    Non-query pairs fall back to ``default_type_compatible``.
    """

    from .d_match import default_type_compatible
    from .scoring import TYPE_GROUPS

    family = TYPE_GROUPS.get(dv.feature_type, dv.feature_type)
    if qv.kind != "query":
        if default_type_compatible(qv, dv):
            return True
        # Exact surface identity is the strongest anchor: NER can type the
        # same mention differently across sides ("Emma" as PERSON in the
        # query but WORK_OF_ART in the document) and must not break the pair
        # (golden100 D01).
        q_text, d_text = qv.text.strip().casefold(), dv.text.strip().casefold()
        if not q_text or not d_text:
            return False
        if q_text == d_text:
            return True
        # Lexical containment: retokenization merges noun compounds into one
        # vertex ("striker score"), hiding the head noun from typed matching.
        # A word-boundary containment pair approximates the production NLI
        # fallback, which accepts such fragment pairs as entailments
        # (golden100 C12/J03).
        shorter, longer = (q_text, d_text) if len(q_text) <= len(d_text) else (d_text, q_text)
        pattern = r"\b" + re.escape(shorter) + r"\b"
        return bool(re.search(pattern, longer))
    expected = (qv.expected_type or "").upper() or qv.feature_type
    if expected in {"", "UNKNOWN", "WHAT", "WHICH", "REASON"}:
        return dv.matchable
    if expected == "LOCATION":
        # Production ent_range groups GPE/LOC/FAC/ORG/NORP together for
        # locations — ORG covers NER-mistyped places ("Kyōto" → ORG).
        return family in {"LOCATION", "ORGANIZATION"} or dv.feature_type in {"COUNTRY", "NORP"}
    if expected == "TIME":
        return family == "TEMPORAL"
    if expected == "NUMBER":
        return family == "QUANTITY" or dv.feature_type == "NUMBER"
    if expected == "PERSON":
        # "Who" is legitimately answered by organizations ("The startup
        # hired..."); PROPN stands in for the production NLI fallback when
        # NER misses a name ("José Saramago").  Abstract/common nouns are
        # NOT persons ("medicine" must not answer ?Who — golden100 D04).
        return family in {"AGENT", "ORGANIZATION"} or dv.feature_type == "PROPN"
    if expected == "BELONGS":
        return family in {"AGENT", "ORGANIZATION", "LOCATION"}
    if expected == "ATTRIBUTE":
        return dv.kind in {"adjective", "adverb"}
    return default_type_compatible(qv, dv)


def whole_query_clusters(query_core: Hypergraph) -> tuple:
    """One cluster per connected edge component of each query context.

    Passing these as ``query_clusters`` prunes HC enumeration to
    component-level alignments (measured 426→15 candidates on a 6-edge nested
    query — 28× fewer embeddings) instead of exploring every sub-cluster
    permutation.  Returns an empty tuple when the query has no edges.
    """

    from .contracts import Cluster

    clusters = []
    for context_id in query_core.context_ids:
        adjacency = query_core.edge_adjacency(context_id)
        seen: set[str] = set()
        for edge_id in sorted(adjacency):
            if edge_id in seen:
                continue
            component: list[str] = []
            stack = [edge_id]
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                component.append(current)
                stack.extend(sorted(adjacency[current] - seen))
            clusters.append(
                Cluster.from_edges(query_core, component, context_id=context_id)
            )
    return tuple(clusters)


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

    query_core = bridge_hypergraph(query, side="query").hypergraph
    data_core = bridge_hypergraph(data, side="data").hypergraph
    return {
        (qv.id, dv.id)
        for qv in query_core.vertices
        if qv.matchable
        for dv in data_core.vertices
        if dv.matchable and bridge_type_compatible(qv, dv)
    }
