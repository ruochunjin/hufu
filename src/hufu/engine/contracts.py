"""Shared, dependency-free contracts for the HyperMatch computation pipeline.

The production project stores richer spaCy and model objects on its vertices.
Those objects are deliberately absent here.  The classes below retain only the
information used by HCCalc, HC scoring, D-match, the Hyper Simulation fixed
point, and query-coverage Consistency.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal, Mapping

Pair = tuple[str, str]
VertexKind = Literal[
    "entity",
    "query",
    "event",
    "value",
    "adjective",
    "adverb",
    "predicate",
    "virtual",
]

MATCHABLE_KINDS = frozenset(
    {"entity", "query", "event", "value", "adjective", "adverb"}
)
LEXICAL_KINDS = frozenset({"predicate", "adjective", "adverb"})


def stable_id(prefix: str, *parts: object) -> str:
    """Return a content-derived identifier independent of enumeration order."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


@dataclass(frozen=True, slots=True)
class Vertex:
    """One typed hypergraph vertex.

    A query placeholder uses ``kind='query'`` and stores the expected answer
    type in ``expected_type``.  It is therefore represented in exactly the
    same type space as data vertices; strings such as ``?ocean`` never become
    a special lexical feature.
    """

    id: str
    text: str
    semantic_type: str
    kind: VertexKind = "entity"
    expected_type: str | None = None
    referent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("vertex id must not be empty")
        if self.kind not in MATCHABLE_KINDS | {"predicate", "virtual"}:
            raise ValueError(f"unsupported vertex kind: {self.kind!r}")
        if self.kind == "query" and not (self.expected_type or self.semantic_type):
            raise ValueError("a query placeholder needs an expected answer type")

    @property
    def feature_type(self) -> str:
        """Type seen by HC/D-match after query-variable refinement."""

        value = self.expected_type if self.kind == "query" else self.semantic_type
        return (value or "UNKNOWN").upper()

    @property
    def matchable(self) -> bool:
        return self.kind in MATCHABLE_KINDS


@dataclass(frozen=True, slots=True)
class Role:
    """A canonical semantic role incident to one hyperedge."""

    name: str
    vertex_id: str

    def __post_init__(self) -> None:
        if not self.name or not self.vertex_id:
            raise ValueError("role name and vertex id must not be empty")


@dataclass(frozen=True, slots=True)
class Hyperedge:
    """One provenance-preserving n-ary fact.

    ``context_id`` is attached to the fact rather than inferred from its
    vertices.  Vertices may be shared after fusion, while a fact still belongs
    to one source context.  A valid fact contains a predicate and at least one
    semantic role; root-only pseudo-hyperedges are rejected.
    """

    id: str
    predicate_id: str
    roles: tuple[Role, ...]
    context_id: str
    canonical_frame: str = ""
    voice: str = "unknown"
    polarity: str = "positive"
    modality: str = "asserted"
    time: str | None = None
    quantity: str | None = None
    father_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.predicate_id or not self.context_id:
            raise ValueError("hyperedge id, predicate, and context are required")
        if not self.roles:
            raise ValueError(f"root-only hyperedge {self.id!r} is not a fact")
        # Coordination can legitimately attach more than one participant to
        # the same canonical role.  Only an identical incidence is invalid.
        incidences = [(role.name, role.vertex_id) for role in self.roles]
        if len(incidences) != len(set(incidences)):
            raise ValueError(f"hyperedge {self.id!r} repeats a role incidence")

    @property
    def vertex_ids(self) -> tuple[str, ...]:
        return (self.predicate_id, *(role.vertex_id for role in self.roles))


@dataclass(frozen=True, slots=True)
class Hypergraph:
    """Validated query or fused-data hypergraph used by the reference core."""

    vertices: tuple[Vertex, ...]
    hyperedges: tuple[Hyperedge, ...]
    side: Literal["query", "data"]
    _vertices: Mapping[str, Vertex] = field(init=False, repr=False, compare=False)
    _hyperedges: Mapping[str, Hyperedge] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        vertex_map = {vertex.id: vertex for vertex in self.vertices}
        edge_map = {edge.id: edge for edge in self.hyperedges}
        if len(vertex_map) != len(self.vertices):
            raise ValueError("hypergraph contains duplicate vertex ids")
        if len(edge_map) != len(self.hyperedges):
            raise ValueError("hypergraph contains duplicate hyperedge ids")
        for edge in self.hyperedges:
            missing = set(edge.vertex_ids) - set(vertex_map)
            if missing:
                raise ValueError(f"hyperedge {edge.id!r} references unknown vertices: {missing}")
            if vertex_map[edge.predicate_id].kind != "predicate":
                raise ValueError(f"hyperedge {edge.id!r} predicate is not a predicate vertex")
            if edge.father_id is not None:
                father = edge_map.get(edge.father_id)
                if father is None:
                    raise ValueError(f"hyperedge {edge.id!r} has an unknown father")
                if father.context_id != edge.context_id:
                    raise ValueError("father links may not cross source contexts")
        object.__setattr__(self, "_vertices", vertex_map)
        object.__setattr__(self, "_hyperedges", edge_map)

    def vertex(self, vertex_id: str) -> Vertex:
        return self._vertices[vertex_id]

    def edge(self, edge_id: str) -> Hyperedge:
        return self._hyperedges[edge_id]

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(sorted({edge.context_id for edge in self.hyperedges}))

    def edges_in_context(self, context_id: str) -> tuple[Hyperedge, ...]:
        return tuple(
            sorted(
                (edge for edge in self.hyperedges if edge.context_id == context_id),
                key=lambda edge: edge.id,
            )
        )

    def incident_edge_ids(
        self, vertex_id: str, *, context_id: str | None = None
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                edge.id
                for edge in self.hyperedges
                if vertex_id in edge.vertex_ids
                and (context_id is None or edge.context_id == context_id)
            )
        )

    def edge_adjacency(self, context_id: str) -> dict[str, set[str]]:
        """Return the dual adjacency induced by shared incident vertices."""

        edges = self.edges_in_context(context_id)
        result = {edge.id: set() for edge in edges}
        for offset, left in enumerate(edges):
            left_vertices = set(left.vertex_ids)
            for right in edges[offset + 1 :]:
                # Father links are typed structural incidence even when a
                # parser fragment has no explicit shared role vertex.
                father_adjacent = (
                    left.father_id == right.id or right.father_id == left.id
                )
                if father_adjacent or left_vertices & set(right.vertex_ids):
                    result[left.id].add(right.id)
                    result[right.id].add(left.id)
        return result


@dataclass(frozen=True, slots=True)
class Cluster:
    """A connected, context-local set of hyperedges."""

    edge_ids: tuple[str, ...]
    context_id: str
    id: str = ""

    def __post_init__(self) -> None:
        canonical = tuple(sorted(dict.fromkeys(self.edge_ids)))
        if not canonical:
            raise ValueError("a hyperedge cluster must not be empty")
        object.__setattr__(self, "edge_ids", canonical)
        if not self.context_id:
            raise ValueError("cluster context must not be empty")
        if not self.id:
            object.__setattr__(self, "id", stable_id("cluster", self.context_id, *canonical))

    @classmethod
    def from_edges(
        cls, graph: Hypergraph, edge_ids: Iterable[str], *, context_id: str
    ) -> "Cluster":
        """Construct and validate a context-local connected edge cluster."""

        value = cls(tuple(edge_ids), context_id)
        value.validate(graph)
        return value

    def validate(self, graph: Hypergraph) -> None:
        """Require all selected edges to share a context and remain connected."""

        edges = tuple(graph.edge(edge_id) for edge_id in self.edge_ids)
        if any(edge.context_id != self.context_id for edge in edges):
            raise ValueError("an HC may not mix source contexts")
        if len(edges) == 1:
            return
        adjacency = graph.edge_adjacency(self.context_id)
        selected = set(self.edge_ids)
        visited: set[str] = set()
        pending = [min(selected)]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(sorted((adjacency[current] & selected) - visited))
        if visited != selected:
            raise ValueError("a hyperedge cluster must be connected")

    def vertices(self, graph: Hypergraph) -> tuple[Vertex, ...]:
        """Return every incident vertex in stable identifier order."""

        ids = {
            vertex_id
            for edge_id in self.edge_ids
            for vertex_id in graph.edge(edge_id).vertex_ids
        }
        return tuple(graph.vertex(vertex_id) for vertex_id in sorted(ids))


@dataclass(frozen=True, slots=True)
class HCCandidate:
    """One threshold-free HCCalc proposal with an explicit route trace."""

    query_cluster: Cluster
    data_cluster: Cluster
    routes: tuple[str, ...]
    structural_score: float
    id: str = ""

    def __post_init__(self) -> None:
        routes = tuple(sorted(dict.fromkeys(self.routes)))
        if not routes:
            raise ValueError("an HC candidate needs at least one proposal route")
        object.__setattr__(self, "routes", routes)
        if not self.id:
            object.__setattr__(
                self,
                "id",
                stable_id("hc", self.query_cluster.id, self.data_cluster.id),
            )


@dataclass(frozen=True, slots=True)
class HCScore:
    """Calibrated HC probability with optional per-view diagnostics."""

    probability: float
    view_scores: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("HC probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class DMatchDecision:
    """Semantic-role relation returned for one accepted HC pair."""

    pairs: frozenset[Pair]
    relation_conflict: bool = False
    status: str = "ok"
    pair_scores: tuple[tuple[Pair, float], ...] = ()

    def __post_init__(self) -> None:
        left = [pair[0] for pair in self.pairs]
        right = [pair[1] for pair in self.pairs]
        if len(left) != len(set(left)) or len(right) != len(set(right)):
            raise ValueError("D-match must be a partial one-to-one relation")
        score_pairs = [pair for pair, _ in self.pair_scores]
        if len(score_pairs) != len(set(score_pairs)):
            raise ValueError("D-match contains duplicate pair scores")
        if any(pair not in self.pairs for pair in score_pairs):
            raise ValueError("D-match score refers to a pair outside the decoded relation")
        if any(not 0.0 <= float(score) <= 1.0 for _, score in self.pair_scores):
            raise ValueError("D-match pair scores must be probabilities")

    @property
    def effective_pairs(self) -> frozenset[Pair]:
        return frozenset() if self.relation_conflict else self.pairs

    @property
    def inference_failed(self) -> bool:
        """Whether no semantic D-match decision was obtained.

        This state must be distinguished from a valid empty relation, which
        is destructive semantic evidence.  Failed decisions are never
        registered in the fixed point.
        """

        return self.status.startswith("inference_error")


@dataclass(frozen=True, slots=True)
class HCDependency:
    """One logical HC and all node pairs with which Delta associates it."""

    id: str
    query_cluster: Cluster
    data_cluster: Cluster
    anchor_pairs: frozenset[Pair]
    dmatch: DMatchDecision
    registration_mode: Literal["full", "support_only"]
    query_edge_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.anchor_pairs:
            raise ValueError("an HC dependency needs an id and at least one anchor")
        if self.registration_mode not in {"full", "support_only"}:
            raise ValueError("unknown HC registration mode")
        if self.dmatch.inference_failed:
            raise ValueError("an HC with failed D-match inference cannot be registered")

    @property
    def destructive(self) -> bool:
        return self.registration_mode == "full"


@dataclass(frozen=True, slots=True)
class HCFailure:
    """Why one HC dependency failed for one tentative vertex pair.

    The two condition names identify anchor-membership and recursive-closure
    failures.  Keeping one record per HC is important: a document is
    ``conflict`` only when an HC containing one of its hyperedges has a
    qualifying deletion certificate.  A bare fact that some vertex pair
    disappeared is not enough.
    """

    hc_id: str
    condition: Literal["anchor_membership", "dependency_closure", "diagnostic"]
    reason: str
    missing_pairs: tuple[Pair, ...] = ()

    def __post_init__(self) -> None:
        if not self.hc_id:
            raise ValueError("an HC failure needs an HC id")
        if self.condition not in {
            "anchor_membership",
            "dependency_closure",
            "diagnostic",
        }:
            raise ValueError(f"unknown Hyper Simulation condition: {self.condition!r}")
        missing = tuple(sorted(set(self.missing_pairs)))
        if self.condition == "dependency_closure" and not missing:
            raise ValueError(
                "a dependency-closure failure must identify a missing D-match pair"
            )
        if self.condition == "anchor_membership" and missing:
            raise ValueError(
                "an anchor-membership failure cannot carry missing closure pairs"
            )
        object.__setattr__(self, "missing_pairs", missing)


@dataclass(frozen=True, slots=True)
class Removal:
    """One fixed-point deletion and its cluster-specific cause records."""

    iteration: int
    pair: Pair
    failures: tuple[HCFailure, ...]

    def __post_init__(self) -> None:
        if self.iteration <= 0:
            raise ValueError("removal iteration must be positive")
        ordered = tuple(
            sorted(
                self.failures,
                key=lambda value: (
                    value.hc_id,
                    value.condition,
                    value.reason,
                    value.missing_pairs,
                ),
            )
        )
        if not ordered:
            raise ValueError("a removal needs at least one HC failure")
        object.__setattr__(self, "failures", ordered)

    @property
    def reason(self) -> str:
        """Stable summary retained for logs and the compact examples."""

        reasons = tuple(sorted({value.reason for value in self.failures}))
        return reasons[0] if len(reasons) == 1 else "+".join(reasons)

    @property
    def hc_ids(self) -> tuple[str, ...]:
        return tuple(sorted({value.hc_id for value in self.failures}))

    @property
    def missing_pairs(self) -> tuple[Pair, ...]:
        return tuple(
            sorted(
                {
                    pair
                    for failure in self.failures
                    for pair in failure.missing_pairs
                }
            )
        )


@dataclass(frozen=True, slots=True)
class HyperSimulationResult:
    """Immutable fixed-point relation and its traceable deletion history."""

    initial_relation: frozenset[Pair]
    relation: frozenset[Pair]
    iterations: int
    removals: tuple[Removal, ...]
    mode: str = "all_hc"

    def relation_by_query(self) -> dict[str, tuple[str, ...]]:
        """Group surviving data vertex ids by query vertex id."""

        grouped: dict[str, list[str]] = {}
        for query_id, data_id in sorted(self.relation):
            grouped.setdefault(query_id, []).append(data_id)
        return {key: tuple(values) for key, values in grouped.items()}


class HCRegistry:
    """Deduplicate logical HC definitions while indexing all Delta anchors."""

    def __init__(self) -> None:
        self._dependencies: dict[str, HCDependency] = {}
        self._by_anchor: dict[Pair, set[str]] = {}

    def register(self, dependency: HCDependency) -> str:
        """Register one logical HC and index each of its Delta anchors."""

        previous = self._dependencies.get(dependency.id)
        if previous is not None and previous != dependency:
            raise ValueError(f"conflicting definitions for logical HC {dependency.id}")
        self._dependencies[dependency.id] = dependency
        for pair in dependency.anchor_pairs:
            self._by_anchor.setdefault(pair, set()).add(dependency.id)
        return dependency.id

    @property
    def dependencies(self) -> tuple[HCDependency, ...]:
        """Return registered dependencies in stable identifier order."""

        return tuple(self._dependencies[key] for key in sorted(self._dependencies))

    def ids_for_anchor(self, pair: Pair) -> tuple[str, ...]:
        """Return logical HC ids associated with one tentative vertex pair."""

        return tuple(sorted(self._by_anchor.get(pair, set())))

    def __len__(self) -> int:
        return len(self._dependencies)


def validate_pairs_exist(
    pairs: Iterable[Pair], query: Hypergraph, data: Hypergraph
) -> frozenset[Pair]:
    """Validate pair endpoints and return a canonical immutable relation."""

    result = frozenset((str(left), str(right)) for left, right in pairs)
    for left, right in result:
        if left not in query._vertices or right not in data._vertices:
            raise ValueError(f"pair {(left, right)!r} references an unknown vertex")
    return result


def iter_cluster_roles(graph: Hypergraph, cluster: Cluster) -> Iterator[tuple[str, Role]]:
    """Yield ``(edge_id, role)`` pairs in deterministic order."""

    for edge_id in cluster.edge_ids:
        edge = graph.edge(edge_id)
        for role in sorted(edge.roles, key=lambda value: (value.name, value.vertex_id)):
            yield edge_id, role


__all__ = [
    "Cluster",
    "DMatchDecision",
    "HCCandidate",
    "HCDependency",
    "HCFailure",
    "HCRegistry",
    "HCScore",
    "HyperSimulationResult",
    "Hyperedge",
    "Hypergraph",
    "LEXICAL_KINDS",
    "MATCHABLE_KINDS",
    "Pair",
    "Removal",
    "Role",
    "Vertex",
    "iter_cluster_roles",
    "stable_id",
    "validate_pairs_exist",
]
