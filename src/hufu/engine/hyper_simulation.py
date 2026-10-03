"""Hyper Simulation entry points and model-injected orchestration.

This module is the shortest reading path through the method:

``h_v -> HCCalc -> HC scoring -> D-match -> fixed point -> Consistency``.

``run_hypermatch`` executes the typed pipeline with caller-provided model
backends.  ``compute_hyper_simulation`` preserves the compatibility interface
for production hypergraph objects.  Keeping both implementations here makes
their shared fixed-point boundary explicit.

Estimator components are frozen before the fixed point starts.  No
model, dataset, checkpoint, API client, or experiment path is imported here.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import (
    FANCY_CONFIG,
    FancyHyperMatchConfig,
)
from .consistent import (
    CompleteCoreResult,
    DocumentMark,
    MatchedHCWitness,
    QueryConsistencyResult,
    compute_complete_cores,
    compute_query_consistency,
    mark_documents,
)
from .contracts import (
    Cluster,
    HCCandidate,
    HCDependency,
    HCRegistry,
    HCScore,
    HyperSimulationResult,
    Pair,
)
from .contracts import (
    Hypergraph as CoreHypergraph,
)
from .contracts import (
    Vertex as CoreVertex,
)
from .d_match import RoleCellScorer, compute_dmatch, default_type_compatible
from .fixed_point import compute_hyper_simulation as compute_fixed_point
from .registration import HCRegistrationDecision, decide_hc_registration
from .scoring import (
    CachedEmbeddingBackend,
    EmbeddingBackend,
    score_hc_candidates,
    score_singleton_relations,
)
from .semantic_cluster import enumerate_hc_candidates


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    """HC score, registration outcome, and resulting dependency id."""

    candidate_id: str
    hc_score: HCScore
    registration: HCRegistrationDecision
    dependency_id: str | None


@dataclass(frozen=True, slots=True)
class HyperMatchOutput:
    """Complete immutable output of the model-injected pipeline."""

    config_fingerprint: str
    candidates: tuple[HCCandidate, ...]
    candidate_decisions: tuple[CandidateDecision, ...]
    dependencies: tuple[HCDependency, ...]
    fixed_point: HyperSimulationResult
    query_consistency: QueryConsistencyResult
    complete_cores: CompleteCoreResult
    document_marks: tuple[DocumentMark, ...]


def run_hypermatch(
    query: CoreHypergraph,
    data: CoreHypergraph,
    *,
    hv_allowed_pairs: Iterable[Pair],
    hc_backend: EmbeddingBackend,
    role_scorer: RoleCellScorer,
    config: FancyHyperMatchConfig = FANCY_CONFIG,
    query_clusters: Sequence[Cluster] | None = None,
    type_compatible: Callable[[CoreVertex, CoreVertex], bool] = default_type_compatible,
) -> HyperMatchOutput:
    """Execute the compact HyperMatch reference pipeline.

    The function is intentionally deterministic.  Model backends may batch on
    GPU internally, but their outputs are converted into immutable decisions
    before the monotone fixed-point calculation begins.
    """

    allowed = _validate_hv(
        query, data, hv_allowed_pairs, type_compatible=type_compatible
    )
    # Consistency quantifies over the complete query hypergraph.  Predicate
    # vertices are represented structurally by hyperedges in this schema and
    # remain outside D-match.  Complete cores therefore require all query edges
    # and all matchable semantic query vertices; callers cannot silently pass a
    # favorable subset.
    required_edges = tuple(sorted(edge.id for edge in query.hyperedges))
    required_vertices = tuple(
        sorted(vertex.id for vertex in query.vertices if vertex.matchable)
    )
    if not required_edges or not required_vertices:
        raise ValueError("HyperMatch requires query edges and comparison vertices")

    cached_backend = CachedEmbeddingBackend(hc_backend)
    singleton_scores = score_singleton_relations(
        query,
        data,
        cached_backend,
        weights=config.three_view_weights,
    )
    candidates = enumerate_hc_candidates(
        query,
        data,
        allowed,
        singleton_scores,
        config.hccalc,
        query_clusters=query_clusters,
    )
    hc_scores = score_hc_candidates(
        query,
        data,
        candidates,
        cached_backend,
        weights=config.three_view_weights,
    )

    registry = HCRegistry()
    decisions: list[CandidateDecision] = []
    for candidate in candidates:
        hc_score = hc_scores[candidate.id]
        anchors = _candidate_anchors(query, data, candidate, allowed)
        if not anchors or hc_score.probability < config.hc.rescue_threshold:
            registration = decide_hc_registration(
                probability=hc_score.probability,
                relation_conflict=False,
                dmatch_pairs=(),
                anchor_pairs=anchors,
                support_threshold=config.hc.support_threshold,
                rescue_threshold=config.hc.rescue_threshold,
                single_answer_rescue_threshold=config.hc.single_answer_rescue_threshold,
                destructive_threshold=config.hc.destructive_threshold,
            )
            decisions.append(
                CandidateDecision(candidate.id, hc_score, registration, None)
            )
            continue

        dmatch = compute_dmatch(
            query,
            data,
            candidate,
            role_scorer,
            hv_allowed_pairs=allowed,
            anchor_pairs=anchors,
            threshold=config.dmatch.ordinary_role_threshold,
            type_compatible=type_compatible,
        )
        if dmatch.inference_failed:
            # No semantic evidence was obtained.  Omitting this HC is the
            # conservative fixed-point behavior: the absence of an associated
            # cluster cannot delete any pair, and the failed result cannot be
            # promoted into a complete-core support witness.
            registration = HCRegistrationDecision(
                hc_accepted=False,
                active=False,
                mode="none",
                rescued=False,
                positive_support=False,
                destructive_use=False,
                reason_codes=("dmatch_inference_error_abstained",),
            )
            decisions.append(
                CandidateDecision(candidate.id, hc_score, registration, None)
            )
            continue
        exact = _exact_fixed_pairs(query, data, anchors)
        canonical = _canonical_role_pairs(query, data, candidate, anchors)
        low_confidence = {
            pair
            for pair, score in dmatch.pair_scores
            if score < config.dmatch.destructive_role_threshold
        }
        query_variables = {
            vertex.id
            for vertex in candidate.query_cluster.vertices(query)
            if vertex.kind == "query"
        }
        registration = decide_hc_registration(
            probability=hc_score.probability,
            relation_conflict=dmatch.relation_conflict,
            dmatch_pairs=dmatch.effective_pairs,
            anchor_pairs=anchors,
            exact_fixed_pairs=exact,
            canonical_role_pairs=canonical,
            low_confidence_role_pairs=low_confidence,
            query_variable_ids=query_variables,
            support_threshold=config.hc.support_threshold,
            rescue_threshold=config.hc.rescue_threshold,
            single_answer_rescue_threshold=config.hc.single_answer_rescue_threshold,
            destructive_threshold=config.hc.destructive_threshold,
        )
        dependency_id: str | None = None
        if registration.active:
            dependency = HCDependency(
                id=candidate.id,
                query_cluster=candidate.query_cluster,
                data_cluster=candidate.data_cluster,
                anchor_pairs=anchors,
                dmatch=dmatch,
                registration_mode=registration.mode,
                query_edge_ids=tuple(
                    edge_id
                    for edge_id in candidate.query_cluster.edge_ids
                    if edge_id in required_edges
                ),
            )
            dependency_id = registry.register(dependency)
        decisions.append(
            CandidateDecision(candidate.id, hc_score, registration, dependency_id)
        )

    dependencies = registry.dependencies
    fixed_point = compute_fixed_point(allowed, dependencies)
    witnesses = _surviving_witnesses(
        data,
        dependencies,
        fixed_point,
    )
    query_consistency = compute_query_consistency(
        required_vertices,
        fixed_point.relation,
    )
    complete_cores = compute_complete_cores(
        required_edges,
        required_vertices,
        witnesses,
    )
    if complete_cores.complete and not query_consistency.consistent:
        raise AssertionError("a complete core must imply query-vertex coverage")
    context_vertices = {
        context_id: {
            vertex_id
            for edge in data.edges_in_context(context_id)
            for vertex_id in edge.vertex_ids
        }
        for context_id in data.context_ids
    }
    document_marks = mark_documents(
        data.context_ids,
        context_vertex_ids=context_vertices,
        fixed_point=fixed_point,
        complete_cores=complete_cores,
        dependencies=dependencies,
    )
    return HyperMatchOutput(
        config.fingerprint,
        candidates,
        tuple(decisions),
        dependencies,
        fixed_point,
        query_consistency,
        complete_cores,
        document_marks,
    )


def _validate_hv(
    query: CoreHypergraph,
    data: CoreHypergraph,
    pairs: Iterable[Pair],
    *,
    type_compatible: Callable[[CoreVertex, CoreVertex], bool],
) -> frozenset[Pair]:
    result: set[Pair] = set()
    for left, right in pairs:
        query_vertex = query.vertex(str(left))
        data_vertex = data.vertex(str(right))
        if not query_vertex.matchable or not data_vertex.matchable:
            continue
        if type_compatible(query_vertex, data_vertex):
            result.add((query_vertex.id, data_vertex.id))
    return frozenset(result)


def _candidate_anchors(
    query: CoreHypergraph,
    data: CoreHypergraph,
    candidate: HCCandidate,
    allowed: frozenset[Pair],
) -> frozenset[Pair]:
    query_ids = {vertex.id for vertex in candidate.query_cluster.vertices(query)}
    data_ids = {vertex.id for vertex in candidate.data_cluster.vertices(data)}
    return frozenset(
        pair for pair in allowed if pair[0] in query_ids and pair[1] in data_ids
    )


def _exact_fixed_pairs(
    query: CoreHypergraph, data: CoreHypergraph, anchors: frozenset[Pair]
) -> frozenset[Pair]:
    return frozenset(
        (left, right)
        for left, right in anchors
        if query.vertex(left).kind != "query"
        and query.vertex(left).text.strip().casefold()
        == data.vertex(right).text.strip().casefold()
    )


def _canonical_role_pairs(
    query: CoreHypergraph,
    data: CoreHypergraph,
    candidate: HCCandidate,
    anchors: frozenset[Pair],
) -> frozenset[Pair]:
    query_roles = _role_names(query, candidate.query_cluster)
    data_roles = _role_names(data, candidate.data_cluster)
    return frozenset(
        pair
        for pair in anchors
        if query_roles.get(pair[0], set()) & data_roles.get(pair[1], set())
    )


def _role_names(graph: CoreHypergraph, cluster: Cluster) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for edge_id in cluster.edge_ids:
        for role in graph.edge(edge_id).roles:
            result.setdefault(role.vertex_id, set()).add(role.name.casefold())
    return result


def _surviving_witnesses(
    data: CoreHypergraph,
    dependencies: Sequence[HCDependency],
    fixed_point: HyperSimulationResult,
) -> tuple[MatchedHCWitness, ...]:
    result = []
    for dependency in dependencies:
        dmatch = dependency.dmatch.effective_pairs
        if (
            not dependency.query_edge_ids
            or not dmatch
            or not dmatch <= fixed_point.relation
        ):
            continue
        # Preserve concrete data vertex ids.  Core coverage takes the union of
        # D-match domains across witnesses; it does not require a cross-HC
        # referent merge or a global one-to-one relation.
        for _query_id, data_id in dmatch:
            data.vertex(data_id)  # hard endpoint existence check
        result.append(
            MatchedHCWitness(
                id=dependency.id,
                query_edge_ids=dependency.query_edge_ids,
                data_edge_ids=dependency.data_cluster.edge_ids,
                context_id=dependency.data_cluster.context_id,
                dmatch_pairs=tuple(sorted(dmatch)),
            )
        )
    return tuple(sorted(result, key=lambda value: value.id))


def compute_hyper_simulation(*args, fancy: bool = False, **kwargs):
    """Run a public call without coercing either solver signature.

    Arguments are forwarded unchanged, preserving each implementation's
    representation, estimator, and return-value contracts.
    """

    if fancy:
        return run_hypermatch(*args, **kwargs)
    _load_standard_dependencies()
    return _compute_standard_hyper_simulation(*args, **kwargs)


def run_selected(*args, fancy: bool = False, **kwargs):
    """Forward an explicitly selected public call to the canonical entry point."""

    return compute_hyper_simulation(*args, fancy=fancy, **kwargs)


__all__ = [
    "CandidateDecision",
    "HyperMatchOutput",
    "SemanticCluster",
    "build_delta_and_dmatch",
    "calc_d_match",
    "calc_d_match_batch",
    "calc_semantic_cluster_pairs",
    "compute_hyper_simulation",
    "consistent_detection",
    "convert_local_to_sim",
    "get_d_match",
    "get_semantic_cluster_pairs",
    "load_hypergraphs_for_instance",
    "query_fixup",
    "run_hypermatch",
    "run_selected",
]


# ---------------------------------------------------------------------------
# Lazily loaded dependencies for compatibility functions
# ---------------------------------------------------------------------------
_STANDARD_DEPENDENCIES_LOADED = False


def _load_standard_dependencies() -> None:
    """Load optional parser, model, and Rust dependencies on first use.

    Structural scoring and public contracts remain dependency-free at import.
    Imports are collected locally and published atomically so a failed optional
    dependency never leaves a partially initialized compatibility namespace.
    """

    global _STANDARD_DEPENDENCIES_LOADED
    if _STANDARD_DEPENDENCIES_LOADED:
        return

    import numpy as numpy_module
    from simulation import (
        Delta as StandardDelta,
    )
    from simulation import (
        DMatch as StandardDMatch,
    )
    from simulation import (
        Hyperedge as StandardSimHyperedge,
    )
    from simulation import (
        Hypergraph as StandardSimHypergraph,
    )
    from simulation import (
        Node as StandardSimNode,
    )
    from tqdm import tqdm as tqdm_function

    from hufu.core.log import (
        current_query_id as standard_current_query_id,
    )
    from hufu.core.log import (
        getLogger as standard_get_logger,
    )
    from hufu.core.query_instance import QueryInstance as StandardQueryInstance
    from hufu.engine.denial import (
        compute_allowed_pairs as standard_compute_allowed_pairs,
    )
    from hufu.engine.denial import (
        compute_allowed_pairs_batch as standard_compute_allowed_pairs_batch,
    )
    from hufu.engine.denial import (
        compute_allowed_pairs_batch_with_score as standard_compute_allowed_pairs_batch_with_score,
    )
    from hufu.engine.denial import (
        get_matched_vertices as standard_get_matched_vertices,
    )
    from hufu.engine.denial import (
        get_top_k_matched_vertices as standard_get_top_k_matched_vertices,
    )
    from hufu.engine.denial import (
        get_top_k_matched_vertices_by_scores as standard_get_top_k_matched_vertices_by_scores,
    )
    from hufu.engine.embedding import (
        cosine_similarity as standard_cosine_similarity,
    )
    from hufu.engine.embedding import (
        get_cosine_similarity_batch as standard_get_cosine_similarity_batch,
    )
    from hufu.engine.embedding import (
        get_embedding_batch as standard_get_embedding_batch,
    )
    from hufu.engine.embedding import (
        get_similarity as standard_get_similarity,
    )
    from hufu.engine.embedding import (
        get_similarity_batch as standard_get_similarity_batch,
    )
    from hufu.engine.nli import (
        get_nli_label as standard_get_nli_label,
    )
    from hufu.engine.nli import (
        get_nli_labels_batch as standard_get_nli_labels_batch,
    )
    from hufu.engine.nli import (
        get_nli_remix_score_batch as standard_get_nli_remix_score_batch,
    )
    from hufu.hypergraph.entity import ENT as StandardENT
    from hufu.hypergraph.hypergraph import (
        Hyperedge as StandardHyperedge,
    )
    from hufu.hypergraph.hypergraph import (
        Hypergraph as StandardHypergraph,
    )
    from hufu.hypergraph.hypergraph import (
        LocalDoc as StandardLocalDoc,
    )
    from hufu.hypergraph.hypergraph import (
        Node as StandardNode,
    )
    from hufu.hypergraph.hypergraph import (
        Vertex as StandardVertex,
    )
    from hufu.hypergraph.linguistic import (
        Dep as StandardDep,
    )
    from hufu.hypergraph.linguistic import (
        Entity as StandardEntity,
    )
    from hufu.hypergraph.linguistic import (
        Pos as StandardPos,
    )
    from hufu.hypergraph.linguistic import (
        QueryType as StandardQueryType,
    )
    from hufu.hypergraph.linguistic import (
        Tag as StandardTag,
    )
    from hufu.hypergraph.path import (
        find_shortest_hyperpaths as standard_find_shortest_hyperpaths,
    )
    from hufu.hypergraph.path import (
        find_shortest_hyperpaths_local as standard_find_shortest_hyperpaths_local,
    )

    globals().update(
        {
            "np": numpy_module,
            "tqdm": tqdm_function,
            "compute_allowed_pairs": standard_compute_allowed_pairs,
            "compute_allowed_pairs_batch": standard_compute_allowed_pairs_batch,
            "compute_allowed_pairs_batch_with_score": standard_compute_allowed_pairs_batch_with_score,
            "get_matched_vertices": standard_get_matched_vertices,
            "get_top_k_matched_vertices": standard_get_top_k_matched_vertices,
            "get_top_k_matched_vertices_by_scores": standard_get_top_k_matched_vertices_by_scores,
            "cosine_similarity": standard_cosine_similarity,
            "get_cosine_similarity_batch": standard_get_cosine_similarity_batch,
            "get_embedding_batch": standard_get_embedding_batch,
            "get_similarity": standard_get_similarity,
            "get_similarity_batch": standard_get_similarity_batch,
            "get_nli_label": standard_get_nli_label,
            "get_nli_labels_batch": standard_get_nli_labels_batch,
            "get_nli_remix_score_batch": standard_get_nli_remix_score_batch,
            "ENT": StandardENT,
            "Hyperedge": StandardHyperedge,
            "Hypergraph": StandardHypergraph,
            "LocalHypergraph": StandardHypergraph,
            "LocalDoc": StandardLocalDoc,
            "Node": StandardNode,
            "Vertex": StandardVertex,
            "Dep": StandardDep,
            "Entity": StandardEntity,
            "Pos": StandardPos,
            "QueryType": StandardQueryType,
            "Tag": StandardTag,
            "find_shortest_hyperpaths": standard_find_shortest_hyperpaths,
            "find_shortest_hyperpaths_local": standard_find_shortest_hyperpaths_local,
            "QueryInstance": StandardQueryInstance,
            "current_query_id": standard_current_query_id,
            "getLogger": standard_get_logger,
            "DMatch": StandardDMatch,
            "Delta": StandardDelta,
            "SimHyperedge": StandardSimHyperedge,
            "SimHypergraph": StandardSimHypergraph,
            "SimNode": StandardSimNode,
        }
    )
    _STANDARD_DEPENDENCIES_LOADED = True
# ---------------------------------------------------------------------------
# Semantic-cluster construction and cluster-local matching
# ---------------------------------------------------------------------------
def abstraction_lca(query: list[str], data: list[str]) -> tuple[str, int]:
    """Return the shared prefix node and depth of two abstraction paths."""

    if not query or not data:
        return '', -1
    if query[0] != data[0]:
        return '', -1
    lca = query[0]
    depth = 0
    min_len = min(len(query), len(data))
    for i in range(min_len):
        if query[i] == data[i]:
            lca = query[i]
            depth = i
        else:
            break
    return lca, depth
def _vertex_sort_key(vertex: Vertex) -> tuple[int, str]:
    return (vertex.id, vertex.text())
def _hyperedge_signature(hyperedge: Hyperedge) -> tuple[int, int, int, str]:
    root_id = hyperedge.root.id if hyperedge.root else -1
    return (root_id, hyperedge.start, hyperedge.end, hyperedge.desc)
def _cluster_sort_key(cluster: 'SemanticCluster') -> tuple:
    return cluster.signature()
class TarjanLCA:
    """Precompute lowest common ancestors used by the path scorer."""

    def __init__(self, edges: list[tuple[Node, Node]], queries: list[tuple[Node, Node]]) -> None:
        self.adj: dict[Node, list[Node]] = {}
        self.nodes: set[Node] = set()
        in_degree: dict[Node, int] = {}
        for a, b in edges:
            self.nodes.add(a)
            self.nodes.add(b)
            if a not in self.adj:
                self.adj[a] = []
            self.adj[a].append(b)
            if a not in in_degree: in_degree[a] = 0
            if b not in in_degree: in_degree[b] = 0
            in_degree[b] += 1
        self.queries = list(queries)
        self.query_map: dict[Node, list[tuple[Node, int]]] = {}
        for i, (u, v) in enumerate(self.queries):
            self.nodes.add(u)
            self.nodes.add(v)
            if u not in in_degree: in_degree[u] = 0
            if v not in in_degree: in_degree[v] = 0
            if u not in self.query_map: self.query_map[u] = []
            if v not in self.query_map: self.query_map[v] = []
            self.query_map[u].append((v, i))
            if u != v:
                self.query_map[v].append((u, i))
        self.uf_parent: dict[Node, Node] = {}
        self.ancestor: dict[Node, Node] = {}
        self.visited: set[Node] = set()
        self.res: list[Node | None] = [None] * len(self.queries)
        self.node_roots: dict[Node, Node] = {}
        for n in list(self.nodes):
            self.uf_parent[n] = n
            self.ancestor[n] = n
        sorted_nodes = sorted(list(self.nodes), key=lambda n: in_degree.get(n, 0))
        for n in sorted_nodes:
            if n not in self.visited:
                self.tarjan(n, None, n)
    def find(self, x):
        """Return a disjoint-set representative with path compression."""

        if x not in self.uf_parent:
            self.uf_parent[x] = x
            return x
        if self.uf_parent[x] != x:
            self.uf_parent[x] = self.find(self.uf_parent[x])
        return self.uf_parent[x]
    def union(self, x, y):
        """Merge two disjoint-set components."""

        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        self.uf_parent[ry] = rx
    def tarjan(self, u, p, root_id):
        """Run one recursive step of the offline LCA traversal."""

        self.node_roots[u] = root_id
        self.ancestor[u] = u
        for v in self.adj.get(u, []):
            if v == p:
                continue
            if v in self.visited:
                continue
            self.tarjan(v, u, root_id)
            self.union(u, v)
            self.ancestor[self.find(u)] = u
        self.visited.add(u)
        for other, qi in self.query_map.get(u, []):
            if other in self.visited:
                if self.node_roots.get(other) == root_id:
                    self.res[qi] = self.ancestor[self.find(other)]
    def lca(self) -> list[Node | None]:
        """Return LCA answers in the same order as the input queries."""

        return self.res
class SemanticCluster:
    """View one relation as one or more dependency hyperedges.

    The class caches structural paths, cluster text, and embeddings because the
    enumerator repeatedly compares the same relation fragments.
    """

    def __init__(self, hyperedges: list[Hyperedge], doc: LocalDoc, is_query: bool=False) -> None:
        _load_standard_dependencies()
        self.hyperedges = hyperedges
        self.doc = doc
        self.vertices: list[Vertex] = []
        self.contained_hyperedges: dict[Vertex, list[Hyperedge]] = {}
        self.embedding: np.ndarray | None = None
        self.text_cache: str | None = None
        self.vertices_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
        self.node_paths_cache: dict[tuple[Node, Node], tuple[str, int]] = {}
        self.is_query = is_query
        self._signature: tuple | None = None
        self.vertices_paths_within_hyperedges: dict[tuple[Node, Node, Node], tuple[str, int]] = {}
        self._hyperedge_groups: list[list[Hyperedge]] | None = None
        self._group_intersections: dict[tuple[int, int], set[Node]] | None = None
        self._hyperedge_to_group: dict[Hyperedge, int] | None = None
        self._vertex_to_hyperedges: dict[Vertex, list[Hyperedge]] | None = None
        self._node_pair_nearest_root: dict[tuple[Node, Node], Node] | None = None
    def _build_hyperedge_groups(self) -> tuple[list[list[Hyperedge]], dict[Hyperedge, int]]:
        if self._hyperedge_groups is not None and self._hyperedge_to_group is not None:
            return self._hyperedge_groups, self._hyperedge_to_group
        ultimate_root_cache: dict[Node, Node] = {}
        def get_ultimate_root(start: Node) -> Node:
            if start in ultimate_root_cache:
                return ultimate_root_cache[start]
            current = start
            visited: set[Node] = set()
            trace: list[Node] = []
            while True:
                if current in ultimate_root_cache:
                    ultimate = ultimate_root_cache[current]
                    break
                if current in visited:
                    ultimate = current
                    break
                visited.add(current)
                trace.append(current)
                head = current.head
                if head is None or head == current:
                    ultimate = current
                    break
                current = head
            for node in trace:
                ultimate_root_cache[node] = ultimate
            return ultimate
        groups_dict: dict[Node, list[Hyperedge]] = {}
        for he in self.hyperedges:
            root = he.current_node(he.root)
            if root is None:
                continue
            ultimate_root = get_ultimate_root(root)
            if ultimate_root not in groups_dict:
                groups_dict[ultimate_root] = []
            groups_dict[ultimate_root].append(he)
        groups = list(groups_dict.values())
        he_to_group: dict[Hyperedge, int] = {}
        for group_idx, group in enumerate(groups):
            for he in group:
                he_to_group[he] = group_idx
        self._hyperedge_groups = groups
        self._hyperedge_to_group = he_to_group
        node_pairs_to_roots: dict[tuple[Node, Node], Node] = {}
        for group in groups:
            node_to_roots: dict[Node, set[Node]] = {}
            group_nodes: set[Node] = set()
            for he in group:
                root = he.current_node(he.root)
                if root is None:
                    continue
                for vertex in he.vertices:
                    node = he.current_node(vertex)
                    if node is None:
                        continue
                    group_nodes.add(node)
                    if node not in node_to_roots:
                        node_to_roots[node] = set()
                    node_to_roots[node].add(root)
            root_chain_cache: dict[Node, list[Node]] = {}
            root_depth_cache: dict[Node, dict[Node, int]] = {}
            root_pair_nearest_cache: dict[tuple[Node, Node], Node | None] = {}
            def get_root_chain(root: Node) -> list[Node]:
                if root in root_chain_cache:
                    return root_chain_cache[root]
                chain: list[Node] = []
                current = root
                visited: set[Node] = set()
                while current is not None and current not in visited:
                    visited.add(current)
                    chain.append(current)
                    current = current.head
                root_chain_cache[root] = chain
                return chain
            def get_root_depth_map(root: Node) -> dict[Node, int]:
                if root in root_depth_cache:
                    return root_depth_cache[root]
                depth_map: dict[Node, int] = {}
                for depth, ancestor in enumerate(get_root_chain(root)):
                    depth_map[ancestor] = depth
                root_depth_cache[root] = depth_map
                return depth_map
            def nearest_common_root(root1: Node, root2: Node) -> Node | None:
                if root1 == root2:
                    return root1
                key = (root1, root2)
                if key in root_pair_nearest_cache:
                    return root_pair_nearest_cache[key]
                ancestors1 = get_root_depth_map(root1)
                nearest: Node | None = None
                for ancestor in get_root_chain(root2):
                    if ancestor in ancestors1:
                        nearest = ancestor
                        break
                root_pair_nearest_cache[(root1, root2)] = nearest
                root_pair_nearest_cache[(root2, root1)] = nearest
                return nearest
            group_nodes_list = list(group_nodes)
            for i in range(len(group_nodes_list)):
                for j in range(i + 1, len(group_nodes_list)):
                    node1 = group_nodes_list[i]
                    node2 = group_nodes_list[j]
                    roots1 = node_to_roots.get(node1, set())
                    roots2 = node_to_roots.get(node2, set())
                    if not roots1 or not roots2:
                        continue
                    best_root: Node | None = None
                    best_score: int | None = None
                    for root1 in roots1:
                        depth1 = get_root_depth_map(root1)
                        for root2 in roots2:
                            common = nearest_common_root(root1, root2)
                            if common is None:
                                continue
                            depth2 = get_root_depth_map(root2)
                            score = depth1.get(common, 10**9) + depth2.get(common, 10**9)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_root = common
                    if best_root is not None:
                        node_pairs_to_roots[(node1, node2)] = best_root
                        node_pairs_to_roots[(node2, node1)] = best_root
        self._node_pair_nearest_root = node_pairs_to_roots
        vertex_to_groups: dict[Vertex, set[int]] = {}
        for he in self.hyperedges:
            group_idx = he_to_group.get(he)
            if group_idx is None:
                continue
            for vertex in he.vertices:
                if vertex not in vertex_to_groups:
                    vertex_to_groups[vertex] = set()
                vertex_to_groups[vertex].add(group_idx)
        self._vertex_to_groups_cache = vertex_to_groups
        def get_group_nodes(group: list[Hyperedge]) -> set[Node]:
            nodes = set()
            for he in group:
                for vertex in he.vertices:
                    node = he.current_node(vertex)
                    if node is not None:
                        nodes.add(node)
            return nodes
        group_adjacency: dict[int, set[int]] = {}
        inter_group_bridges: dict[tuple[int, int], set[Node]] = {}
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                nodes_i = get_group_nodes(groups[i])
                nodes_j = get_group_nodes(groups[j])
                bridges = nodes_i & nodes_j
                if bridges:
                    group_adjacency.setdefault(i, set()).add(j)
                    group_adjacency.setdefault(j, set()).add(i)
                    inter_group_bridges[(i, j)] = bridges
                    inter_group_bridges[(j, i)] = bridges
        self._inter_group_bridge_cache = inter_group_bridges
        inter_group_distances: dict[tuple[int, int], list[int]] = {}
        for start_group in range(len(groups)):
            dist: dict[int, int] = {start_group: 0}
            parent: dict[int, int | None] = {start_group: None}
            queue = deque([start_group])
            while queue:
                current = queue.popleft()
                for neighbor in group_adjacency.get(current, set()):
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        parent[neighbor] = current
                        queue.append(neighbor)
            for end_group, d in dist.items():
                if end_group == start_group:
                    inter_group_distances[(start_group, end_group)] = [start_group]
                else:
                    path = []
                    current = end_group
                    while current is not None:
                        path.append(current)
                        current = parent.get(current)
                    path.reverse()
                    inter_group_distances[(start_group, end_group)] = path
        self._inter_group_distances_cache = inter_group_distances
        return groups, he_to_group
    def _find_group_intersections(self) -> dict[tuple[int, int], set[Node]]:
        if self._group_intersections is not None:
            return self._group_intersections
        groups, _ = self._build_hyperedge_groups()
        def get_group_nodes(group: list[Hyperedge]) -> set[Node]:
            nodes = set()
            for he in group:
                for vertex in he.vertices:
                    node = he.current_node(vertex)
                    if node is not None:
                        nodes.add(node)
            return nodes
        intersections: dict[tuple[int, int], set[Node]] = {}
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                nodes_i = get_group_nodes(groups[i])
                nodes_j = get_group_nodes(groups[j])
                intersection = nodes_i & nodes_j
                if intersection:
                    intersections[(i, j)] = intersection
                    intersections[(j, i)] = intersection
        self._group_intersections = intersections
        return intersections
    def _get_nearest_root_for_node_pair(self, node1: Node, node2: Node) -> Node | None:
        self._build_hyperedge_groups()
        if self._node_pair_nearest_root is not None:
            return self._node_pair_nearest_root.get((node1, node2))
        return None
    def _build_vertex_to_groups(self) -> dict[Vertex, set[int]]:
        self._build_hyperedge_groups()
        if not hasattr(self, '_vertex_to_groups_cache'):
            self._vertex_to_groups_cache = {}
        return self._vertex_to_groups_cache
    def _get_vertex_groups(self, v: Vertex) -> set[int]:
        vertex_to_groups = self._build_vertex_to_groups()
        return vertex_to_groups.get(v, set())
    def _build_inter_group_bridge_map(self) -> dict[tuple[int, int], set[Node]]:
        self._build_hyperedge_groups()
        if not hasattr(self, '_inter_group_bridge_cache'):
            self._inter_group_bridge_cache = {}
        return self._inter_group_bridge_cache
    def _find_shortest_group_path(self, g1: int, g2: int) -> list[int]:
        if g1 == g2:
            return [g1]
        self._build_hyperedge_groups()
        if hasattr(self, '_inter_group_distances_cache'):
            return self._inter_group_distances_cache.get((g1, g2), [])
        return []
    def _find_closest_group_pair(self, v1_groups: set[int], v2_groups: set[int]) -> tuple[int, int, list[int]] | None:
        if not v1_groups or not v2_groups:
            return None
        common_groups = v1_groups & v2_groups
        if common_groups:
            g = next(iter(common_groups))
            return (g, g, [g])
        self._build_hyperedge_groups()
        shortest_path: list[int] | None = None
        best_length = float('inf')
        best_pair: tuple[int, int] | None = None
        if hasattr(self, '_inter_group_distances_cache'):
            distances = self._inter_group_distances_cache
            for g1 in v1_groups:
                for g2 in v2_groups:
                    path = distances.get((g1, g2), [])
                    if path and len(path) < best_length:
                        best_length = len(path)
                        best_pair = (g1, g2)
                        shortest_path = path
        if best_pair is None or shortest_path is None:
            return None
        return (best_pair[0], best_pair[1], shortest_path)
    def get_path_node_steps(self, v1: Vertex, v2: Vertex) -> tuple[list[list[Node]], Node | None, Node | None]:
        """Return dependency-node segments connecting two cluster vertices."""

        logger = getLogger("semantic_cluster")
        if v1 is None or v2 is None:
            return [], None, None
        try:
            v1_groups = self._get_vertex_groups(v1)
            v2_groups = self._get_vertex_groups(v2)
            if not v1_groups or not v2_groups:
                logger.debug("[get_path_node_steps] v1 or v2 not in any group")
                return [], None, None
            closest_result = self._find_closest_group_pair(v1_groups, v2_groups)
            if closest_result is None:
                logger.debug(f"[get_path_node_steps] No path between v1_groups={v1_groups} and v2_groups={v2_groups}")
                return [], None, None
            g1, gn, group_path = closest_result
            logger.debug(f"[get_path_node_steps] Found path: {group_path}")
            groups, he_to_group = self._build_hyperedge_groups()
            n1 = None
            for he in self.hyperedges:
                if v1 in he.vertices and he_to_group.get(he) == g1:
                    n1 = he.current_node(v1)
                    if n1 is not None:
                        break
            if n1 is None:
                logger.debug(f"[get_path_node_steps] Cannot find node for v1 in group {g1}")
                return [], None, None
            n2 = None
            for he in self.hyperedges:
                if v2 in he.vertices and he_to_group.get(he) == gn:
                    n2 = he.current_node(v2)
                    if n2 is not None:
                        break
            if n2 is None:
                logger.debug(f"[get_path_node_steps] Cannot find node for v2 in group {gn}")
                return [], None, None
            triple_sequence: list[tuple[Node, int, Node]] = []
            bridge_map = self._build_inter_group_bridge_map()
            if g1 == gn:
                triple_sequence.append((n1, g1, n2))
            else:
                current_node = n1
                for i in range(len(group_path) - 1):
                    gi = group_path[i]
                    next_gi = group_path[i + 1]
                    bridge_key = (gi, next_gi) if (gi, next_gi) in bridge_map else (next_gi, gi)
                    bridges = bridge_map.get(bridge_key, set())
                    if not bridges:
                        logger.debug(f"[get_path_node_steps] No bridges between {gi} and {next_gi}")
                        return [], None, None
                    next_node = next(iter(bridges))
                    triple_sequence.append((current_node, gi, next_node))
                    current_node = next_node
                triple_sequence.append((current_node, gn, n2))
            result: list[list[Node]] = []
            for node_a, group_idx, node_b in triple_sequence:
                assert self._node_pair_nearest_root
                nearest_root = self._node_pair_nearest_root.get((node_a, node_b))
                if nearest_root is None:
                    nearest_root = self._node_pair_nearest_root.get((node_b, node_a))
                if nearest_root is None:
                    logger.debug(f"[get_path_node_steps] Cannot find common root for {node_a.text} and {node_b.text} in group {group_idx}")
                    return [], None, None
                path_a: list[Node] = []
                current = node_a
                visited_a: set[Node] = set()
                while current is not None and current not in visited_a:
                    visited_a.add(current)
                    path_a.append(current)
                    if current == nearest_root:
                        break
                    current = current.head
                path_b: list[Node] = []
                current = node_b
                visited_b: set[Node] = set()
                while current is not None and current not in visited_b:
                    visited_b.add(current)
                    path_b.append(current)
                    if current == nearest_root:
                        break
                    current = current.head
                if not path_a or path_a[-1] != nearest_root:
                    logger.debug(f"[get_path_node_steps] Failed to trace {node_a.text} to root {nearest_root.text}")
                    return [], None, None
                if not path_b or path_b[-1] != nearest_root:
                    logger.debug(f"[get_path_node_steps] Failed to trace {node_b.text} to root {nearest_root.text}")
                    return [], None, None
                merged_path = path_a + path_b[-2::-1]
                sorted_nodes = sorted(merged_path, key=lambda node: node.index if hasattr(node, 'index') else float('inf'))
                result.append(sorted_nodes)
            return result, n1, n2
        except Exception as e:
            logger.exception(f"[get_path_node_steps] Error finding path: {e}")
            return [], None, None
    @staticmethod
    def likely_nodes(nodes1: list[Vertex], nodes2: list[Vertex]) -> dict[Vertex, set[Vertex]]:
        """Group NLI-compatible candidates from ``nodes2`` by source vertex."""

        likely_nodes: dict[Vertex, set[Vertex]] = {}
        text_pair_to_node_pairs: dict[tuple[str, str], tuple[Vertex, Vertex]] = {}
        for node1 in nodes1:
            for node2 in nodes2:
                text_pair_to_node_pairs[(node1.text(), node2.text())] = (node1, node2)
        text_pairs = list(text_pair_to_node_pairs.keys())
        labels = get_nli_labels_batch(text_pairs)
        for i, text_pair in enumerate(text_pairs):
            node_pair = text_pair_to_node_pairs[text_pair]
            label = labels[i]
            node1, node2 = node_pair
            if label == "entailment" or (label == "neutral" and node1.is_domain(node2)):
                if node1 not in likely_nodes:
                    likely_nodes[node1] = set()
                likely_nodes[node1].add(node2)
        return likely_nodes
    def is_subset_of(self, other: 'SemanticCluster') -> bool:
        """Return whether all of this cluster's hyperedges occur in ``other``."""

        self_edge_set = set(self.hyperedges)
        other_edge_set = set(other.hyperedges)
        return self_edge_set.issubset(other_edge_set)
    def get_contained_hyperedges(self, vertex: Vertex) -> list[Hyperedge]:
        """Return and cache cluster hyperedges incident to a vertex."""

        if vertex in self.contained_hyperedges:
            return self.contained_hyperedges[vertex]
        contained_edges: list[Hyperedge] = []
        for he in self.hyperedges:
            if vertex in he.vertices:
                contained_edges.append(he)
        self.contained_hyperedges[vertex] = contained_edges
        return contained_edges
    def get_vertices(self) -> list[Vertex]:
        """Return cluster vertices in first-incidence order without duplicates."""

        if len(self.vertices) > 0:
            return self.vertices
        id_set: set[int] = set()
        ordered_vertices: list[Vertex] = []
        for he in self.hyperedges:
            for v in he.vertices:
                if v.id in id_set:
                    continue
                id_set.add(v.id)
                ordered_vertices.append(v)
        self.vertices = ordered_vertices
        return self.vertices
    def get_path_within_hyperedges(self, v1: Node, v2: Node, root: Node) -> tuple[str, int]:
        """Render and measure the dependency path between two nodes via a root."""

        key = (v1, v2, root)
        if key in self.vertices_paths_within_hyperedges:
            return self.vertices_paths_within_hyperedges[key]
        path_nodes: set[Node] = set()
        path_nodes.add(root)
        path_nodes.add(v1)
        path_nodes.add(v2)
        current = v1.head
        while current and current not in path_nodes:
            path_nodes.add(current)
            current = current.head
        current = v2.head
        while current and current not in path_nodes:
            path_nodes.add(current)
            current = current.head
        nodes = sorted(list(path_nodes), key=lambda n: n.index)
        base_desc = " ".join([f"{n.text}" for n in nodes])
        type_v1 = v1.type_str()
        type_v2 = v2.type_str()
        if not type_v1:
            type_v1 = v1.text
        if not type_v2:
            type_v2 = v2.text
        if type_v1 == type_v2:
            type_v1 = f"{type_v1}#1"
            type_v2 = f"{type_v2}#2"
        desc = base_desc.replace(v1.text, type_v1).replace(v2.text, type_v2)
        return desc, len(nodes)
    def get_path_to_root(self, node: Node, root: Node) -> tuple[str, int]:
        """Render and measure a dependency path from one node to a root."""

        path_nodes: set[Node] = set()
        path_nodes.add(root)
        path_nodes.add(node)
        current = node.head
        while current and current not in path_nodes:
            path_nodes.add(current)
            current = current.head
        nodes = sorted(list(path_nodes), key=lambda n: n.index)
        desc = " ".join([f"{n.text}" for n in nodes])
        node_type = node.type_str()
        if not node_type:
            node_type = node.text
        desc = desc.replace(node.text, node_type)
        return desc, len(nodes)
    def get_paths_between_vertices(self, v1: Vertex, v2: Vertex) -> tuple[str, int]:
        """Return the shortest cached masked path between two cluster vertices."""

        key = (v1, v2)
        if key in self.vertices_paths:
            return self.vertices_paths[key]
        logger = getLogger("semantic_cluster")
        logger.debug(f"get_paths_between_vertices called for: '{v1.text()}' ↔ '{v2.text()}'")
        node_vertex: dict[Node, Vertex] = {}
        nodes_in_vertices: set[Node] = set()
        for he in self.hyperedges:
            for v in he.vertices:
                if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
                    continue
                nodes_in_vertices.add(he.current_node(v))
                node_vertex[he.current_node(v)] = v
        nodes_in_vertices_list = list(nodes_in_vertices)
        queries: list[tuple[Node, Node]] = []
        for i in range(len(nodes_in_vertices_list) - 1):
            for j in range(i + 1, len(nodes_in_vertices_list)):
                u = nodes_in_vertices_list[i]
                v = nodes_in_vertices_list[j]
                queries.append((u, v))
        edge_between_nodes: list[tuple[Node, Node]] = []
        saved_nodes: set[Node] = set()
        for he in self.hyperedges:
            root = he.current_node(he.root)
            for i in range(1, len(he.vertices)):
                node = he.current_node(he.vertices[i])
                edge_between_nodes.append((root, node))
                saved_nodes.add(node)
            head = root.head
            current = root
            visited_in_trace = {current}
            while head:
                if head in visited_in_trace:
                    logger.warning(f"Cycle detected in head trace: '{head.text}' is already in trace. Breaking.")
                    break
                visited_in_trace.add(head)
                edge_between_nodes.append((head, current))
                if head in saved_nodes:
                    break
                current = head
                if head.head == head:
                    logger.warning(f"Detected self-loop at node '{current.text}' during v→k trace. Breaking.")
                    break
                head = head.head
            saved_nodes.add(root)
        lca_results = TarjanLCA(edge_between_nodes, queries).lca()
        lca_map: dict[tuple[Node, Node], Node] = {}
        for i, (u, v) in enumerate(queries):
            lca_node = lca_results[i]
            if lca_node:
                lca_map[(u, v)] = lca_node
        node_paths: dict[tuple[Vertex, Vertex], list[tuple[str, int]]] = {}
        for (u, v), k in lca_map.items():
            vertex_u = node_vertex[u]
            vertex_v = node_vertex[v]
            if u == k:
                text = f"#A -{v.dep.name}-> #B"
                node_paths.setdefault((vertex_u, vertex_v), []).append((text, 1))
                continue
            elif v == k:
                text = f"#A <-{u.dep.name}- #B"
                node_paths.setdefault((vertex_u, vertex_v), []).append((text, 1))
                continue
            node_cnt = 1
            path_items: list[Node] = []
            current = u
            current_trace: list[str] = [current.text]
            visited_trace: set[Node] = {current}
            while current != k:
                if current in nodes_in_vertices:
                    node_cnt += 1
                    path_items.append(current)
                if current.head is None:
                    logger.warning(f"路径追溯失败 u→k: Node '{current.text}' (index={current.index}) has no head "
                            f"while tracing to LCA '{k.text}' (index={k.index}). "
                            f"Trace: {' → '.join(current_trace)}")
                    break
                if current.head in visited_trace:
                    logger.warning(f"Cycle detected in u→k trace: '{current.head.text}' is already in trace. Breaking.")
                    break
                visited_trace.add(current.head)
                if current.head == current:
                    logger.warning(f"Detected self-loop at node '{current.text}' during u→k trace. Breaking.")
                    break
                current = current.head
                current_trace.append(current.text)
            else:
                path_items.append(k)
                rev_path_items: list[Node] = []
                current = v
                current_trace = [current.text]
                visited_trace_v: set[Node] = {current}
                while current != k:
                    if current in nodes_in_vertices:
                        node_cnt += 1
                        rev_path_items.append(current)
                    if current.head is None:
                        logger.warning(f"路径追溯失败 v→k: Node '{current.text}' (index={current.index}) has no head "
                                f"while tracing to LCA '{k.text}' (index={k.index}). "
                                f"Trace: {' → '.join(current_trace)}")
                        break
                    if current.head in visited_trace_v:
                        logger.warning(f"Cycle detected in v→k trace: '{current.head.text}' is already in trace. Breaking.")
                        break
                    visited_trace_v.add(current.head)
                    if current.head == current:
                        logger.warning(f"Detected self-loop at node '{current.text}' during v→k trace. Breaking.")
                        break
                    current = current.head
                    current_trace.append(current.text)
                else:
                    rev_path_items = rev_path_items[::-1]
                    path_items.extend(rev_path_items)
                    text = node_sequence_to_text(path_items)
                    text_inv = text.replace("#A", "#TEMP").replace("#B", "#A").replace("#TEMP", "#B")
                    node_paths.setdefault((vertex_u, vertex_v), []).append((text, node_cnt))
                    node_paths.setdefault((vertex_v, vertex_u), []).append((text_inv, node_cnt))
                    continue
        for (vertex_u, vertex_v), paths in node_paths.items():
            if paths:
                paths = sorted(paths, key=lambda x: x[1])
                self.vertices_paths[(vertex_u, vertex_v)] = paths[0]
        result = self.vertices_paths.get(key, ("", 0))
        logger.debug(f"get_paths_between_vertices result: count={result[1]}, sample='{result[0][:50]}...'")
        return result
    def text(self) -> str:
        """Render and cache the resolved source text covered by the cluster."""

        if self.text_cache is not None:
            return self.text_cache
        if not self.hyperedges:
            return ""
        logger = getLogger("semantic_cluster")
        try:
            root_ancestors = {}
            for e in self.hyperedges:
                root_node = e.current_node(e.root)
                if root_node is None:
                    logger.error(f"[text] Hyperedge {e} has invalid root node (None). Skipping.")
                    continue
                root_ancestors[root_node] = root_node
            for e in self.hyperedges:
                root = e.current_node(e.root)
                if root is None:
                    continue
                node = root
                visited = set()
                while node.head is not None:
                    if node in visited:
                        logger.warning(f"[text] Detected cycle in ancestor chain starting from {root.text}. Breaking.")
                        break
                    visited.add(node)
                    if node.head in root_ancestors:
                        root_ancestors[root] = root_ancestors[node.head]
                        break
                    node = node.head
            root_to_nodes: dict[Node, set[Node]] = {}
            for e in self.hyperedges:
                root = e.current_node(e.root)
                if root is None or root not in root_ancestors:
                    continue
                ultimate_root = root_ancestors[root]
                if ultimate_root not in root_to_nodes:
                    root_to_nodes[ultimate_root] = set()
                for vertex in e.vertices:
                    node = e.current_node(vertex)
                    if node is not None:
                        root_to_nodes[ultimate_root].add(node)
            sub_cluster_roots = set(root_ancestors.get(r, r) for r in root_to_nodes.keys())
            sub_clusters = sorted(list(sub_cluster_roots), key=lambda r: getattr(r, 'index', float('inf')))
            texts = []
            for root in sub_clusters:
                if root not in root_to_nodes:
                    continue
                nodes = list(root_to_nodes[root])
                if not nodes:
                    continue
                try:
                    start = min(getattr(node, 'index', 0) for node in nodes)
                    end = max(getattr(node, 'index', 0) for node in nodes) + 1
                except Exception as ex:
                    logger.error(f"[text] Failed to compute indices for root {root.text}: {ex}")
                    continue
                sentence_by_range = str(self.doc[start:end]) if self.doc else ""
                sentence_obj = getattr(root, 'sentence', None)
                sentence = str(sentence_obj) if sentence_obj else ""
                def calc_prefix_suffix(range_text, full_sentence):
                    start_idx = full_sentence.find(range_text)
                    if start_idx != -1:
                        prefix = full_sentence[:start_idx].strip()
                        suffix = full_sentence[start_idx + len(range_text):].strip()
                        return prefix, suffix
                    else:
                        return "", ""
                prefix, suffix = calc_prefix_suffix(sentence_by_range, sentence)
                replacement = []
                for node in nodes:
                    if node == root:
                        continue
                    resolved_text = Vertex.resolved_text(node)
                    original_text = getattr(node, 'text', '')
                    replacement.append((original_text, resolved_text))
                if prefix:
                    replacement.append((prefix, ""))
                if suffix:
                    replacement.append((suffix, ""))
                final_sentence = sentence
                for old, new in replacement:
                    if old in final_sentence:
                        final_sentence = final_sentence.replace(old, new)
                cleaned = final_sentence.strip()
                if cleaned:
                    texts.append(cleaned)
            text = " ".join(texts).strip()
            self.text_cache = text
            return text
        except Exception as e:
            logger.exception(f"[text] Unexpected error in SemanticCluster.text(): {e}")
            fallback = " ".join(
                str(e.current_node(e.root).text) for e in self.hyperedges
                if e.current_node(e.root) and hasattr(e.current_node(e.root), 'text')
            ).strip()
            self.text_cache = fallback
            return fallback
    def virtual_text(self) -> str:
        """Reserved hook for virtualized cluster rendering."""

        pass
    def _build_signature(self) -> tuple:
        if not self.hyperedges:
            return ()
        items = []
        for he in self.hyperedges:
            root_id = he.root.id if he.root else -1
            items.append((root_id, he.start, he.end, he.desc))
        items.sort()
        return tuple(items)
    def signature(self) -> tuple:
        """Return the stable structural signature used for hashing and equality."""

        if self._signature is None:
            self._signature = self._build_signature()
        return self._signature
    def to_triple(self) -> list[tuple[str, list[str]]]:
        """Project each hyperedge into a predicate and resolved argument list."""

        triples = []
        for he in self.hyperedges:
            root_text = Vertex.resolved_text(he.current_node(he.root))
            args = []
            for vertex in he.vertices:
                if vertex == he.root:
                    continue
                node = he.current_node(vertex)
                node_text = Vertex.resolved_text(node)
                args.append(node_text)
                if node.pos in {Pos.ADJ, Pos.ADV} and node.dep in {Dep.amod, Dep.advmod}:
                    head = node.head
                    if head and head.pos in {Pos.NOUN, Pos.PROPN, Pos.VERB}:
                        head_text = Vertex.resolved_text(head)
                        triples.append(("attr", [head_text, node_text]))
            triples.append((root_text, args))
        return triples
    def to_triple_text(self) -> str:
        """Render the cluster's projected triples as a compact conjunction."""

        texts = []
        for root, args in self.to_triple():
            if len(args) == 0:
                texts.append(f"{root}()")
            else:
                texts.append(f"{root}({', '.join(args)})")
        return " & ".join(texts)
    def __hash__(self) -> int:
        return hash((self.is_query, self.signature()))
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticCluster):
            return False
        if self.is_query != other.is_query:
            return False
        return self.signature() == other.signature()
def combine_hyperedges_to_cluster(hypergraph: Hypergraph) -> list[SemanticCluster]:
    """Apply ordered structural rules that group parser hyperedges into HCs."""

    clusters: list[SemanticCluster] = []
    root_to_hyperedge: dict[Node, Hyperedge] = {}
    for he in hypergraph.hyperedges:
        root_node = he.current_node(he.root)
        if root_node is None:
            continue
        root_to_hyperedge[root_node] = he
    hyperedge_visited: set[Hyperedge] = set()
    for he in hypergraph.hyperedges:
        if he in hyperedge_visited:
            continue
        root_node = he.current_node(he.root)
        if root_node is None:
            continue
        if root_node.pos in {Pos.VERB, Pos.AUX}:
            continue
        children = he.vertices[1:]
        descent_hyperedges = []
        for child_vertex in children:
            child_node = he.current_node(child_vertex)
            if child_node is None:
                continue
            if child_node.pos in {Pos.VERB, Pos.AUX}:
                child_he = root_to_hyperedge.get(child_node)
                if child_he and child_he not in hyperedge_visited:
                    descent_hyperedges.append(child_he)
                    hyperedge_visited.add(child_he)
        if not descent_hyperedges:
            continue
        cluster_hyperedges = [he] + descent_hyperedges
        clusters.append(SemanticCluster(cluster_hyperedges, hypergraph.doc, is_query=True))
        hyperedge_visited.add(he)
    for he in hypergraph.hyperedges:
        root_node = he.current_node(he.root)
        if root_node and root_node.dep == Dep.relcl and root_node.head:
            head_he = root_to_hyperedge.get(root_node.head)
            if head_he and head_he not in hyperedge_visited:
                clusters.append(SemanticCluster([he, head_he], hypergraph.doc, is_query=True))
                hyperedge_visited.add(he)
                hyperedge_visited.add(head_he)
    for he in hypergraph.hyperedges:
        root_node = he.current_node(he.root)
        if root_node and root_node.pos in {Pos.VERB, Pos.AUX} and root_node.dep in {Dep.advcl, Dep.ccomp} and root_node.head and root_node.head.pos in {Pos.VERB, Pos.AUX}:
            head_he = root_to_hyperedge.get(root_node.head)
            if head_he and head_he not in hyperedge_visited:
                clusters.append(SemanticCluster([he, head_he], hypergraph.doc, is_query=True))
                hyperedge_visited.add(he)
                hyperedge_visited.add(head_he)
    for he in hypergraph.hyperedges:
        if he not in hyperedge_visited:
            clusters.append(SemanticCluster([he], hypergraph.doc))
    return clusters
def calc_semantic_cluster_pairs(
    hypergraph_q: Hypergraph,
    hypergraph_d: Hypergraph,
    likely_map: dict[Vertex, set[Tuple[Vertex, float]]],
    cluster_sim_threshold: float = 0.5,
    branch_threshold: int = 5,
    is_multihop: bool = False,
    logger: Optional[logging.Logger] = None
) -> list[tuple[SemanticCluster, SemanticCluster, float]]:
    """Enumerate and threshold path-seeded HC candidate pairs.

    Candidate paths are seeded from top-b likely vertex matches, then scored
    with the cluster embedding estimator.  Stable intermediate scores keep the
    enumeration directly traceable.
    """

    assert logger is not None
    logger.info(f"[SemanticClusterPairs] Start: Q_edges={len(hypergraph_q.hyperedges)}, "
                   f"D_edges={len(hypergraph_d.hyperedges)}, likely_map={len(likely_map)}, "
                   f"threshold={cluster_sim_threshold}, multihop={is_multihop}")
    pairs: list[tuple[SemanticCluster, SemanticCluster, float]] = []
    clusters_q = combine_hyperedges_to_cluster(hypergraph_q)
    logger.info(f"[SemanticClusterPairs] Step1-Done: Generated {len(clusters_q)} query clusters from {len(hypergraph_q.hyperedges)} hyperedges")
    calc_embedding_for_cluster_batch(clusters_q)
    logger.info(f"[SemanticClusterPairs] Step2-Done: Computed embeddings for {len(clusters_q)} query clusters")
    K_LIKELY = branch_threshold
    matched_count = sum(len(v) for v in likely_map.values())
    logger.info(f"[SemanticClusterPairs] Step3-Done: likely_map built with {len(likely_map)} source vertices, "
                   f"{matched_count} total matches (K={K_LIKELY})")
    vertices_pairs_need_path: list[tuple[Vertex, Vertex]] = []
    vertices_pairs_to_sc: dict[tuple[Vertex, Vertex], list[SemanticCluster]] = {}
    pair_gen_start = time.time() if logger else None
    for sc_q in clusters_q:
        vertices_d_pairs: set[tuple[Vertex, Vertex]] = set()
        for u in sc_q.get_vertices():
            for u_prime in sc_q.get_vertices():
                if u == u_prime:
                    continue
                if Vertex.is_both_verb(u, u_prime):
                    continue
                for v, score_v in likely_map.get(u, set()):
                    for v_prime, score_v_prime in likely_map.get(u_prime, set()):
                        if v == v_prime:
                            continue
                        if Vertex.is_both_verb(v, v_prime):
                            continue
                        vertices_d_pairs.add((v, v_prime))
        for pair in vertices_d_pairs:
            vertices_pairs_to_sc.setdefault(pair, []).append(sc_q)
            vertices_pairs_need_path.append(pair)
    if logger and pair_gen_start is not None:
        pair_gen_time = time.time() - pair_gen_start
        unique_pairs = len(set(vertices_pairs_need_path))
        logger.info(f"[SemanticClusterPairs] Step4-Done: Generated {len(vertices_pairs_need_path)} raw pairs, "
                   f"{unique_pairs} unique pairs, {len(vertices_pairs_to_sc)} pairs mapped to clusters, "
                   f"time={pair_gen_time:.2f}s")
    vertices_pairs_need_path = list(set(vertices_pairs_need_path))
    logger.info(f"[SemanticClusterPairs] Step5-Start: Path search for {len(vertices_pairs_need_path)} unique vertex pairs "
                   f"(method={'local' if is_multihop else 'global'})")
    path_search_start = time.time()
    if is_multihop:
        path_map = find_shortest_hyperpaths_local(hypergraph_d, vertices_pairs_need_path)
    else:
        path_map = find_shortest_hyperpaths(hypergraph_d, vertices_pairs_need_path)
    path_search_time = time.time() - path_search_start
    reachable = sum(1 for p in path_map.values() if p)
    logger.info(f"[SemanticClusterPairs] Step5-Done: Path search completed in {path_search_time:.2f}s, "
                f"reachable={reachable}/{len(path_map)} pairs")
    sc_pairs_candidates: list[tuple[SemanticCluster, SemanticCluster]] = []
    sc_d_candidates: list[SemanticCluster] = []
    for (v, v_prime), scs in vertices_pairs_to_sc.items():
        path = path_map.get((v, v_prime), [])
        for sc_q in scs:
            sc_d = SemanticCluster(path, hypergraph_d.doc)
            sc_pairs_candidates.append((sc_q, sc_d))
            sc_d_candidates.append(sc_d)
    logger.info(f"[SemanticClusterPairs] Step6-Done: Built {len(sc_pairs_candidates)} candidate cluster pairs, {len(sc_d_candidates)} document clusters to embed")
    calc_embedding_for_cluster_batch(sc_d_candidates)
    logger.info(f"[SemanticClusterPairs] Step7-Done: Computed embeddings for {len(sc_d_candidates)} document clusters")
    filter_start = time.time()
    sim_embedding_pairs = [ (sc_q.embedding, sc_d.embedding) for sc_q, sc_d in sc_pairs_candidates]
    if sim_embedding_pairs:
        sim_scores = get_cosine_similarity_batch(sim_embedding_pairs, is_normalized=True)
    else:
        sim_scores = []
    passed_count = 0
    for (sc_q, sc_d), sim_score in zip(sc_pairs_candidates, sim_scores):
        assert sc_q.embedding is not None and sc_d.embedding is not None, "Embedding should have been calculated"
        if sim_score >= cluster_sim_threshold:
            pairs.append((sc_q, sc_d, sim_score))
            passed_count += 1
    filter_time = time.time() - filter_start
    logger.info(f"[SemanticClusterPairs] Step8-Done: Filtered {passed_count}/{len(sc_pairs_candidates)} pairs "
                f"by threshold={cluster_sim_threshold}, time={filter_time:.2f}s")
    logger.info(f"[SemanticClusterPairs] Return: {len(pairs)} final semantic cluster pairs")
    return pairs
def build_descendant_cluster(
    vertex: Vertex,
    hg: Hypergraph,
    max_hops: int = 2
) -> 'SemanticCluster':
    """Collect incident facts reachable through bounded dependency descendants."""

    logger = getLogger("semantic_cluster")
    node = vertex.nodes[0] if vertex.nodes else None
    if not node or not hasattr(node, 'children'):
        direct_edges = [e for e in hg.hyperedges if vertex in e.vertices]
        return SemanticCluster(direct_edges, hg.doc)
    descendant_nodes = {node}
    queue = [(node, 0)]
    while queue:
        curr_node, depth = queue.pop(0)
        if depth >= max_hops:
            continue
        for child in getattr(curr_node, 'children', []):
            if child not in descendant_nodes:
                descendant_nodes.add(child)
                queue.append((child, depth + 1))
    visited_edges = set()
    node_to_vertex = {}
    for v in hg.vertices:
        for n in v.nodes:
            node_to_vertex[n] = v
    for e in hg.hyperedges:
        for v in e.vertices:
            if any(n in descendant_nodes for n in v.nodes):
                visited_edges.add(e)
                break
    logger.debug(
        f"build_descendant_cluster: vertex='{vertex.text()}' (ID={vertex.id}) → "
        f"{len(descendant_nodes)} descendant nodes, {len(visited_edges)} hyperedges"
    )
    return SemanticCluster(list(visited_edges), hg.doc)
def calc_embedding_for_cluster_batch(clusters: list[SemanticCluster]) -> None:
    """Populate semantic clusters with normalized text embeddings."""

    texts = [sc.text() for sc in clusters]
    embeddings = get_embedding_batch(texts)
    for i, sc in enumerate(clusters):
        sc.embedding = np.array(embeddings[i])
def get_semantic_cluster_pairs(
    query_hg: Hypergraph,
    data_hg: Hypergraph,
    allowed_pairs: Set[Tuple[int, int]],
    vertex_to_sim_id_q: Dict[Vertex, int],
    vertex_to_sim_id_d: Dict[Vertex, int],
    max_hops_query: int = 1,
    max_hops_data: int = 2,
    cluster_sim_threshold: float = 0.4,
    logger: Optional[logging.Logger] = None
) -> List[Tuple[SemanticCluster, SemanticCluster, float, Vertex, Vertex]]:
    """Build and score one descendant-cluster pair per allowed vertex pair."""

    if logger is None:
        logger = getLogger("semantic_cluster")
    logger.info(f"按需构建语义簇 (Query hops={max_hops_query}, Data hops={max_hops_data})...")
    start_time = time.time()
    sim_id_to_vertex_q = {sim_id: v for v, sim_id in vertex_to_sim_id_q.items()}
    sim_id_to_vertex_d = {sim_id: v for v, sim_id in vertex_to_sim_id_d.items()}
    cluster_pairs = []
    pair_count = kept_count = 0
    for q_sim_id, d_sim_id in allowed_pairs:
        pair_count += 1
        q_vertex = sim_id_to_vertex_q.get(q_sim_id)
        d_vertex = sim_id_to_vertex_d.get(d_sim_id)
        if q_vertex is None or d_vertex is None:
            continue
        sc_q = build_descendant_cluster(q_vertex, query_hg, max_hops=max_hops_query)
        if not sc_q.hyperedges:
            continue
        sc_d = build_descendant_cluster(d_vertex, data_hg, max_hops=max_hops_data)
        if not sc_d.hyperedges:
            continue
        calc_embedding_for_cluster_batch([sc_q, sc_d])
        if sc_q.embedding is None or sc_d.embedding is None:
            continue
        sim_score = cosine_similarity(sc_q.embedding, sc_d.embedding)
        if sim_score >= cluster_sim_threshold:
            cluster_pairs.append((sc_q, sc_d, sim_score, q_vertex, d_vertex))
            kept_count += 1
            q_triples = sc_q.to_triple() or []
            d_triples = sc_d.to_triple() or []
            q_triple_repr = f"({q_triples[0][0]}, {', '.join(q_triples[0][1])})" if q_triples else "(no triple)"
            d_triple_repr = f"({d_triples[0][0]}, {', '.join(d_triples[0][1])})" if d_triples else "(no triple)"
            logger.debug(
                f"→ 采纳簇对 #{kept_count} | Δ(Q{q_vertex.id}, D{d_vertex.id}) score={sim_score:.3f}\n"
                f"  Q: text='{sc_q.text()}' | nodes={len(sc_q.get_vertices())}, edges={len(sc_q.hyperedges)}\n"
                f"     triple={q_triple_repr}\n"
                f"  D: text='{sc_d.text()}' | nodes={len(sc_d.get_vertices())}, edges={len(sc_d.hyperedges)}\n"
                f"     triple={d_triple_repr}"
            )
    cost = time.time() - start_time
    logger.info(f"语义簇构建完成: {pair_count} allowed pairs → {kept_count} high-similarity cluster pairs (cost {cost:.3f}s)")
    return cluster_pairs
def node_sequence_to_text(nodes: list[Node]) -> str:
    """Render a dependency path with masked endpoints and generic noun slots."""

    if not nodes:
        return ""
    start, end = nodes[0], nodes[-1]
    nodes = sorted(nodes, key=lambda n: n.index)
    texts = []
    for node in nodes:
        if node == start:
            texts.append("#A")
        elif node == end:
            texts.append("#B")
        elif node.pos in {Pos.ADV, Pos.ADJ, Pos.DET}:
            continue
        elif node.pos in {Pos.NOUN, Pos.PROPN, Pos.PRON}:
            texts.append("some")
        else:
            texts.append(Vertex.resolved_text(node))
    return " ".join(texts)
def _formal_text_of(root: Node, node: Node) -> str:
    match (root.pos, node.dep):
        case (Pos.AUX, Dep.nsubj) | (Pos.AUX, Dep.nsubjpass):
            text = "#A is something"
        case (Pos.AUX, Dep.iobj) | (Pos.AUX, Dep.dobj):
            text = "#A is something"
        case (Pos.VERB, Dep.nsubj) | (Pos.VERB, Dep.nsubjpass):
            text = "#A does something"
        case (Pos.VERB, Dep.iobj) | (Pos.VERB, Dep.dobj):
            text = "Someone does #A"
        case _:
            text = f"#A -{node.dep.name}-> something"
    return text
def _better_path(s1: str, s2: str, s2_inv: str) -> bool:
    nli_labels = {"entailment": 3, "neutral": 2, "contradiction": 1}
    label1 = get_nli_label(s1, s2)
    label2 = get_nli_label(s1, s2_inv)
    if nli_labels[label1] > nli_labels[label2]:
        return True
    sim1 = get_similarity(s1, s2)
    sim2 = get_similarity(s1, s2_inv)
    return sim1 > sim2
def _legal_vertices(v1: Vertex, v2: Vertex) -> bool:
    label = get_nli_label(v1.text(), v2.text())
    if not (label == "entailment" or (label == "neutral" and v1.is_domain(v2))):
        return False
    dep1 = v1.dep()
    dep2 = v2.dep()
    SUBJECT_DEPS = {Dep.nsubj, Dep.nsubjpass, Dep.csubj, Dep.agent}
    OBJECT_DEPS = {Dep.dobj, Dep.iobj, Dep.pobj, Dep.attr}
    MODIFIER_DEPS = {Dep.amod, Dep.nmod, Dep.advmod, Dep.appos}
    if (dep1 in SUBJECT_DEPS and dep2 in SUBJECT_DEPS) or (dep1 in OBJECT_DEPS and dep2 in OBJECT_DEPS) or (dep1 in MODIFIER_DEPS and dep2 in MODIFIER_DEPS):
        return True
    if {dep1, dep2} <= {Dep.nmod, Dep.dobj}:
        return True
    return False
def _path_score(s1: str, cnt1: int, s2: str, cnt2: int, path_score_cache: dict[tuple[str, str], float]) -> float:
    key = (s1, s2)
    if key in path_score_cache:
        return path_score_cache[key]
    sim = get_similarity(s1, s2)
    score = sim / (cnt1 + cnt2)
    path_score_cache[key] = score
    return score
def _get_matched_vertices(vertices1: list[Vertex], vertices2: list[Vertex]) -> dict[Vertex, set[Vertex]]:
    matched_vertices: dict[Vertex, set[Vertex]] = {}
    text_pair_to_node_pairs: dict[tuple[str, str], tuple[Vertex, Vertex]] = {}
    for node1 in vertices1:
        for node2 in vertices2:
            text_pair_to_node_pairs[(node1.text(), node2.text())] = (node1, node2)
    text_pairs = list(text_pair_to_node_pairs.keys())
    labels = get_nli_labels_batch(text_pairs)
    for i, text_pair in enumerate(text_pairs):
        node_pair = text_pair_to_node_pairs[text_pair]
        label = labels[i]
        node1, node2 = node_pair
        if label == "entailment" or node1.is_domain(node2):
            if node1 not in matched_vertices:
                matched_vertices[node1] = set()
            matched_vertices[node1].add(node2)
    return matched_vertices
def get_d_match(sc1: SemanticCluster, sc2: SemanticCluster, score_threshold: float = 0.0, force_include: Optional[Tuple[Vertex, Vertex]] = None) -> list[tuple[Vertex, Vertex, float]]:
    """Estimate role pairs inside an accepted semantic-cluster pair."""

    dm_logger = getLogger("d_match")
    sc1_vertices_all = sc1.get_vertices()
    sc1_vertices_noun = [v for v in sc1_vertices_all if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
    sc1_edges = sc1.hyperedges
    sc1_text = sc1.text()
    sc1_triples = sc1.to_triple() or []
    sc1_triple_repr = str(sc1_triples[0]) if sc1_triples else "(no triple)"
    sc2_vertices_all = sc2.get_vertices()
    sc2_vertices_noun = [v for v in sc2_vertices_all if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
    sc2_edges = sc2.hyperedges
    sc2_text = sc2.text()
    sc2_triples = sc2.to_triple() or []
    sc2_triple_repr = str(sc2_triples[0]) if sc2_triples else "(no triple)"
    dm_logger.info(
        f"=== D-Match 开始 (阈值={score_threshold}) ===\n"
        f"→ SC1:\n"
        f"   text='{sc1_text}'\n"
        f"   triple={sc1_triple_repr}\n"
        f"   nodes={len(sc1_vertices_all)} (noun={len(sc1_vertices_noun)}), edges={len(sc1_edges)}\n"
        f"→ SC2:\n"
        f"   text='{sc2_text}'\n"
        f"   triple={sc2_triple_repr}\n"
        f"   nodes={len(sc2_vertices_all)} (noun={len(sc2_vertices_noun)}), edges={len(sc2_edges)}"
    )
    matches: list[tuple[Vertex, Vertex]] = []
    sc1_vertices = list(filter(lambda v: not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX)), sc1.get_vertices()))
    sc2_vertices = list(filter(lambda v: not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX)), sc2.get_vertices()))
    index_map: dict[Vertex, int] = {}
    for e in sc1.hyperedges:
        for v in e.vertices:
            if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
                continue
            if v not in index_map:
                index_map[v] = e.current_node(v).index
    for e in sc2.hyperedges:
        for v in e.vertices:
            if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
                continue
            if v not in index_map:
                index_map[v] = e.current_node(v).index
    sc1_edges: list[tuple[Vertex, Vertex]] = []
    for he in sc1.hyperedges:
        for i in range(len(he.vertices) - 1):
            for j in range(i + 1, len(he.vertices)):
                if he.have_no_link(he.vertices[i], he.vertices[j]):
                    continue
                if he.is_sub_vertex(he.vertices[i], he.vertices[j]):
                    sc1_edges.append((he.vertices[i], he.vertices[j]))
                else:
                    sc1_edges.append((he.vertices[j], he.vertices[i]))
    sc1_pairs_set = set(sc1_edges)
    added = True
    tc_loop_count = 0
    while added:
        tc_loop_count += 1
        if tc_loop_count == 1 or tc_loop_count % 10 == 0:
            dm_logger.info(f"Transitive Closure Iteyration {tc_loop_count}: current pairs count = {len(sc1_pairs_set)}")
        added = False
        adj = {}
        for u, v in sc1_pairs_set:
            if u not in adj: adj[u] = []
            adj[u].append(v)
        new_edges = set()
        for u in adj:
            for v in adj[u]:
                if v in adj:
                    for w in adj[v]:
                        if u == w: continue
                        if (u, w) not in sc1_pairs_set and (u, w) not in new_edges:
                            new_edges.add((u, w))
                            added = True
        if new_edges:
            sc1_pairs_set.update(new_edges)
    sc1_pairs = list(sc1_pairs_set)
    def _is_pair_in_vertices(u: Vertex, v: Vertex) -> bool:
        if u.pos_equal(Pos.VERB) or u.pos_equal(Pos.AUX):
            return False
        if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
            return False
        return True
    sc1_pairs = list(filter(lambda pairs: _is_pair_in_vertices(pairs[0], pairs[1]), sc1_pairs))
    sc1_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
    for u, v in sc1_pairs:
        s, cnt = sc1.get_paths_between_vertices(u, v)
        if cnt == 0:
            continue
        sc1_paths[(u, v)] = (s, cnt)
    likely_nodes = _get_matched_vertices(sc1_vertices, sc2_vertices)
    sc2_pairs: list[tuple[Vertex, Vertex]] = []
    sc2_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
    dm_logger.info(f"Start Core Matching Logic: {len(sc1_pairs)} sc1 pairs")
    processed_count = 0
    for u, u_prime in sc1_pairs:
        processed_count += 1
        if processed_count % 10 == 0:
             dm_logger.debug(f"Processing sc1 pair {processed_count}/{len(sc1_pairs)}")
        for v, v_prime in itertools.product(likely_nodes.get(u, set()), likely_nodes.get(u_prime, set())):
            if v == v_prime:
                continue
            s1, cnt1 = sc1_paths[(u, u_prime)]
            dm_logger.debug(f"    Calling sc2.get_paths_between_vertices('{v.text()}', '{v_prime.text()}')")
            s2, cnt2 = sc2.get_paths_between_vertices(v, v_prime)
            dm_logger.debug(f"    Forward path: count={cnt2}, sample='{s2[:50]}...'")
            dm_logger.debug(f"    Calling sc2.get_paths_between_vertices('{v_prime.text()}', '{v.text()}')")
            s2_inv, cnt2_prime = sc2.get_paths_between_vertices(v_prime, v)
            dm_logger.debug(f"    Backward path: count={cnt2_prime}, sample='{s2_inv[:50]}...'")
            if cnt2 == 0 or s2 == "":
                if cnt2_prime > 0 and s2_inv:
                    sc2_pairs.append((v_prime, v))
                    sc2_paths[(v_prime, v)] = (s2_inv, cnt2_prime)
                continue
            elif cnt2_prime == 0 or s2_inv == "":
                sc2_pairs.append((v, v_prime))
                sc2_paths[(v, v_prime)] = (s2, cnt2)
                continue
            if not s2 or not s2_inv:
                dm_logger.debug(f"D-Match跳过: Empty paths for vertex pair '{v.text()}' ↔ '{v_prime.text()}' in cluster. s2='{s2}', s2_inv='{s2_inv}'")
                continue
            if _better_path(s1, s2, s2_inv):
                sc2_pairs.append((v, v_prime))
                sc2_paths[(v, v_prime)] = (s2, cnt2)
            else:
                sc2_pairs.append((v_prime, v))
                sc2_paths[(v_prime, v)] = (s2_inv, cnt2)
    dm_logger.debug(f"SC2 inferred path pairs: {[(u.text(), v.text()) for u, v in sc2_pairs]}")
    dm_logger.debug(f"SC2 paths count: {len(sc2_paths)}")
    match_scores: dict[tuple[Vertex, Vertex], float] = {}
    for u, v in itertools.product(sc1_vertices, sc2_vertices):
        if _legal_vertices(u, v):
            matches.append((u, v))
    dm_logger.debug(f"Initial legal matches count: {len(matches)}")
    in_paths_of_sc1: dict[Vertex, list[tuple[str, int]]] = {}
    out_paths_of_sc1: dict[Vertex, list[tuple[str, int]]] = {}
    for u, v in sc1_pairs:
        if v not in in_paths_of_sc1:
            in_paths_of_sc1[v] = []
        in_paths_of_sc1[v].append(sc1_paths[(u, v)])
        if u not in out_paths_of_sc1:
            out_paths_of_sc1[u] = []
        out_paths_of_sc1[u].append(sc1_paths[(u, v)])
    for vertex in sc1_vertices:
        if vertex in in_paths_of_sc1:
            dm_logger.debug(f"SC1 Vertex '{vertex.text()}' In Paths: {[s for s, _ in in_paths_of_sc1[vertex]]}")
        if vertex in out_paths_of_sc1:
            dm_logger.debug(f"SC1 Vertex '{vertex.text()}' Out Paths: {[s for s, _ in out_paths_of_sc1[vertex]]}")
    in_paths_of_sc2: dict[Vertex, list[tuple[str, int]]] = {}
    out_paths_of_sc2: dict[Vertex, list[tuple[str, int]]] = {}
    for u, v in sc2_pairs:
        if v not in in_paths_of_sc2:
            in_paths_of_sc2[v] = []
        in_paths_of_sc2[v].append(sc2_paths[(u, v)])
        if u not in out_paths_of_sc2:
            out_paths_of_sc2[u] = []
        out_paths_of_sc2[u].append(sc2_paths[(u, v)])
    for vertex in sc2_vertices:
        if vertex in in_paths_of_sc2:
            dm_logger.debug(f"SC2 Vertex '{vertex.text()}' In Paths: {[s for s, _ in in_paths_of_sc2[vertex]]}")
        if vertex in out_paths_of_sc2:
            dm_logger.debug(f"SC2 Vertex '{vertex.text()}' Out Paths: {[s for s, _ in out_paths_of_sc2[vertex]]}")
    root_path_of_sc1: dict[Vertex, list[tuple[str, int]]] = {}
    for e in sc1.hyperedges:
        root = e.root
        root_node = e.current_node(root)
        if not (root_node.pos == Pos.VERB or root_node.pos == Pos.AUX):
            continue
        for v in e.vertices[1:]:
            v_node = e.current_node(v)
            if v_node.pos == Pos.VERB or v_node.pos == Pos.AUX:
                continue
            text = _formal_text_of(root_node, v_node)
            if v not in root_path_of_sc1:
                root_path_of_sc1[v] = []
            root_path_of_sc1[v].append((text, 2))
    root_path_of_sc2: dict[Vertex, list[tuple[str, int]]] = {}
    for e in sc2.hyperedges:
        root = e.root
        root_node = e.current_node(root)
        if not (root_node.pos == Pos.VERB or root_node.pos == Pos.AUX):
            continue
        for v in e.vertices[1:]:
            v_node = e.current_node(v)
            if v_node.pos == Pos.VERB or v_node.pos == Pos.AUX:
                continue
            text = _formal_text_of(root_node, v_node)
            if v not in root_path_of_sc2:
                root_path_of_sc2[v] = []
            root_path_of_sc2[v].append((text, 2))
    path_score_cache: dict[tuple[str, str], float] = {}
    path_pair_need_to_calc: set[tuple[str, str]] = set()
    for u, v in matches:
        for s1, cnt1 in in_paths_of_sc1.get(u, []):
            for s2, cnt2 in in_paths_of_sc2.get(v, []):
                if (s1, s2) not in path_score_cache:
                    path_pair_need_to_calc.add((s1, s2))
        for s1, cnt1 in out_paths_of_sc1.get(u, []):
            for s2, cnt2 in out_paths_of_sc2.get(v, []):
                if (s1, s2) not in path_score_cache:
                    path_pair_need_to_calc.add((s1, s2))
        for s1, cnt1 in root_path_of_sc1.get(u, []):
            for s2, cnt2 in root_path_of_sc2.get(v, []):
                if (s1, s2) not in path_score_cache:
                    path_pair_need_to_calc.add((s1, s2))
    if path_pair_need_to_calc:
        dm_logger.info(f"Computing path similarities for {len(path_pair_need_to_calc)} pairs...")
    path_list_1: list[str] = []
    path_list_2: list[str] = []
    path_pair_need_to_calc_list = list(path_pair_need_to_calc)
    for s1, s2 in path_pair_need_to_calc_list:
        path_list_1.append(s1)
        path_list_2.append(s2)
    similarities = get_similarity_batch(path_list_1, path_list_2)
    for i, (s1, s2) in enumerate(path_pair_need_to_calc_list):
        path_score_cache[(s1, s2)] = similarities[i]
    for u, v in matches:
        in_score = 0.0
        in_cnt = 0
        for s1, cnt1 in in_paths_of_sc1.get(u, []):
            for s2, cnt2 in in_paths_of_sc2.get(v, []):
                in_score += _path_score(s1, cnt1, s2, cnt2, path_score_cache)
                in_cnt += 1
        if in_cnt > 0:
            in_score /= in_cnt
        out_score = 0.0
        out_cnt = 0
        for s1, cnt1 in out_paths_of_sc1.get(u, []):
            for s2, cnt2 in out_paths_of_sc2.get(v, []):
                out_score += _path_score(s1, cnt1, s2, cnt2, path_score_cache)
                out_cnt += 1
        if out_cnt > 0:
            out_score /= out_cnt
        root_score = 0.0
        root_cnt = 0
        for s1, cnt1 in root_path_of_sc1.get(u, []):
            for s2, cnt2 in root_path_of_sc2.get(v, []):
                root_score += _path_score(s1, cnt1, s2, cnt2, path_score_cache)
                root_cnt += 1
        if root_cnt > 0:
            root_score /= root_cnt
        match_scores[(u, v)] = in_score + out_score + root_score
    matches = list(filter(lambda pair: match_scores.get(pair, 0.0) >= score_threshold, matches))
    final_matches: list[tuple[Vertex, Vertex, float]] = []
    matches_by_u: dict[Vertex, list[tuple[Vertex, float]]]  = {}
    for u, v in matches:
        score = match_scores.get((u, v), 0.0)
        if u not in matches_by_u:
            matches_by_u[u] = []
        matches_by_u[u].append((v, score))
    for u, v_scores in matches_by_u.items():
        v_scores = sorted(v_scores, key=lambda x: x[1], reverse=True)
        best_v, best_score = v_scores[0]
        final_matches.append((u, best_v, best_score))
    if final_matches:
        dm_logger.info("D-Match 完整结果:")
        for i, (u, v, score) in enumerate(final_matches, 1):
            dm_logger.info(
                f"  [{i}] Q{u.id}: '{u.text()}' "
                f"→ D{v.id}: '{v.text()}' "
                f"(score={score:.4f})"
            )
    else:
        dm_logger.info("D-Match 完整结果: 无匹配")
    if force_include:
        u, v = force_include
        if (u, v, 1.0) not in final_matches:
            final_matches.insert(0, (u, v, 1.0))
    return final_matches

# ---------------------------------------------------------------------------
# D-match estimation
# ---------------------------------------------------------------------------
def query_same_type(v1: Vertex, v2: Vertex) -> bool:
    """Apply the query-placeholder type gate."""

    if v1.query_type():
        return False
    qt = v1.query_type()
    v2_type = v2.type()
    if qt == QueryType.PERSON and v2_type:
        return v2_type == ENT.PERSON
    elif qt == QueryType.TIME and v2_type:
        return v2_type == ENT.TEMPORAL
    elif qt == QueryType.LOCATION and v2_type:
        return v2_type in {ENT.GPE, ENT.LOC, ENT.FAC, ENT.ORG}
    elif qt == QueryType.NUMBER and v2_type:
        return v2_type == ENT.NUMBER
    elif qt == QueryType.ATTRIBUTE:
        return v2.pos_equal(Pos.ADJ) or v2.pos_equal(Pos.ADV)
    return False
def _construct_description_from_path(path: list[list[Node]], start_node: Node, end_node: Node) -> str:
    type_nodes_map: dict[str, set[Node]] = {
        "LOCATION": set(),
        "TEMPORAL": set(),
        "ATTRIBUTE": set(),
        "PERSON": set(),
        "COMPONENTS": set(),
        "REASON": set(),
        "CONCEPT": set(),
        "NUMBER": set(),
        "ORGANISM": set(),
        "FOOD": set(),
        "MEDICAL": set(),
        "ANATOMY": set(),
        "SUBSTANCE": set(),
        "ASTRO": set(),
        "AWARD": set(),
        "VEHICLE": set(),
        "COUNTRY": set(),
        "ORGANIZATION": set(),
        "FACILITY": set(),
        "Geopolitical": set(),
        "NORP": set(),
        "PRODUCT": set(),
        "WORK_OF_ART": set(),
        "LAW": set(),
        "LANGUAGE": set(),
        "OCCUPATION": set(),
        "EVENT": set(),
        "THEORY": set(),
        "GROUP": set(),
        "FEATURE": set(),
        "ECONOMIC": set(),
        "SOCIOLOGY": set(),
        "PHENOMENON": set(),
    }
    node_type_map: dict[Node, str] = {}
    start_node_type = start_node.type_str()
    if start_node_type and start_node_type in type_nodes_map:
        type_nodes_map[start_node_type].add(start_node)
        index = len(type_nodes_map[start_node_type])
        node_type_map[start_node] = f"{start_node_type}#{index}"
    end_node_type = end_node.type_str()
    if end_node_type and end_node_type in type_nodes_map:
        type_nodes_map[end_node_type].add(end_node)
        index = len(type_nodes_map[end_node_type])
        node_type_map[end_node] = f"{end_node_type}#{index}"
    description_parts = []
    for nodes in path:
        if not nodes:
            continue
        node_by_index = sorted(nodes, key=lambda n: n.index)
        def node_text(n: Node) -> str:
            """Mask typed nodes with cluster-local identifiers."""

            if n in node_type_map:
                return node_type_map[n]
            node_type = n.type_str()
            if node_type and node_type in type_nodes_map:
                type_nodes_map[node_type].add(n)
                index = len(type_nodes_map[node_type])
                node_type_map[n] = f"{node_type}#{index}"
                return node_type_map[n]
            return n.text
        node_texts = [node_text(node) for node in node_by_index]
        description_parts.append(" ".join(node_texts))
    return ". ".join(description_parts)
def calc_d_match(sc1: SemanticCluster, sc2: SemanticCluster, threshold: float = 0.5) -> list[tuple[Vertex, Vertex, float]]:
    """Score the legal role-pair universe for one HC pair."""

    R: list[tuple[Vertex, Vertex]] = []
    for v1 in sc1.vertices:
        for v2 in sc2.vertices:
            if v1.is_query():
                if query_same_type(v1, v2):
                    R.append((v1, v2))
                continue
            if v1.is_verb() or v2.is_verb():
                continue
            if v1.type() == v2.type():
                R.append((v1, v2))
    R_map: Dict[Vertex, List[Vertex]] = {}
    for v1, v2 in R:
        if v1 not in R_map:
            R_map[v1] = []
        R_map[v1].append(v2)
    score_items: list[tuple[str, str, Vertex, Vertex, int]] = []
    for v1, v2 in R:
        root_tuples: list[tuple[Vertex, Vertex]] = []
        other_tuples: list[tuple[Vertex, Vertex]] = []
        for hyperedge in sc1.hyperedges:
            if v1 not in hyperedge.vertices:
                continue
            if v1 == hyperedge.root:
                for v1_prime in hyperedge.vertices[1:]:
                    root_tuples.append((v1, v1_prime))
                continue
            for v1_prime in hyperedge.vertices[1:]:
                if v1_prime == v1:
                    continue
                other_tuples.append((v1, v1_prime))
        candidate_pairs: list[tuple[Vertex, Vertex]] = []
        for _, v1_prime in root_tuples:
            for v2_prime in R_map.get(v1_prime, []):
                candidate_pairs.append((v1_prime, v2_prime))
        for _, v1_prime in other_tuples:
            for v2_prime in R_map.get(v1_prime, []):
                candidate_pairs.append((v1_prime, v2_prime))
        for index, (v1_prime, v2_prime) in enumerate(candidate_pairs):
            path1, v1_node, v1_prime_node = sc1.get_path_node_steps(v1, v1_prime)
            path2, v2_node, v2_prime_node = sc2.get_path_node_steps(v2, v2_prime)
            if not path1 or not path2 or v1_node is None or v2_node is None or v1_prime_node is None or v2_prime_node is None:
                continue
            desc1 = _construct_description_from_path(path1, start_node=v1_node, end_node=v1_prime_node)
            desc2 = _construct_description_from_path(path2, start_node=v2_node, end_node=v2_prime_node)
            score_items.append((desc1, desc2, v1, v2, index))
    if not score_items:
        return []
    score_pairs = [(desc1, desc2) for desc1, desc2, _, _, _ in score_items]
    scores = get_nli_remix_score_batch(score_pairs)
    pair_index_max: dict[tuple[Vertex, Vertex], dict[int, float]] = {}
    for (_, _, v1, v2, index), score in zip(score_items, scores):
        key = (v1, v2)
        if key not in pair_index_max:
            pair_index_max[key] = {}
        prev = pair_index_max[key].get(index)
        if prev is None or score > prev:
            pair_index_max[key][index] = score
    raw_results: list[tuple[Vertex, Vertex, float]] = []
    for (v1, v2), index_max_map in pair_index_max.items():
        if not index_max_map:
            continue
        avg_score = sum(index_max_map.values()) / len(index_max_map)
        if avg_score > threshold:
            raw_results.append((v1, v2, avg_score))
    raw_results.sort(key=lambda x: x[2], reverse=True)
    used_v1: set[Vertex] = set()
    used_v2: set[Vertex] = set()
    results: list[tuple[Vertex, Vertex, float]] = []
    for v1, v2, score in raw_results:
        if v1 in used_v1 or v2 in used_v2:
            continue
        used_v1.add(v1)
        used_v2.add(v2)
        results.append((v1, v2, score))
    return results
def calc_d_match_batch(sc_pairs: list[tuple[SemanticCluster, SemanticCluster]], threshold: float = 0.5) -> list[list[tuple[Vertex, Vertex, float]]]:
    """Batch NLI/path descriptions across multiple HC pairs."""

    start_time = time.time()
    if not sc_pairs:
        return []
    score_pairs: list[tuple[str, str]] = []
    score_meta: list[tuple[int, Vertex, Vertex, int]] = []
    pair_index_max_by_pair: list[dict[tuple[Vertex, Vertex], dict[int, float]]] = [
        {} for _ in range(len(sc_pairs))
    ]
    for pair_idx, (sc1, sc2) in enumerate(sc_pairs):
        R: list[tuple[Vertex, Vertex]] = []
        for v1 in sc1.vertices:
            for v2 in sc2.vertices:
                if v1.is_query():
                    if query_same_type(v1, v2):
                        R.append((v1, v2))
                    continue
                if v1.is_verb() or v2.is_verb():
                    continue
                if v1.type() == v2.type():
                    R.append((v1, v2))
        R_map: Dict[Vertex, List[Vertex]] = {}
        for v1, v2 in R:
            if v1 not in R_map:
                R_map[v1] = []
            R_map[v1].append(v2)
        for v1, v2 in R:
            root_tuples: list[tuple[Vertex, Vertex]] = []
            other_tuples: list[tuple[Vertex, Vertex]] = []
            for hyperedge in sc1.hyperedges:
                if v1 not in hyperedge.vertices:
                    continue
                if v1 == hyperedge.root:
                    for v1_prime in hyperedge.vertices[1:]:
                        root_tuples.append((v1, v1_prime))
                    continue
                for v1_prime in hyperedge.vertices[1:]:
                    if v1_prime == v1:
                        continue
                    other_tuples.append((v1, v1_prime))
            candidate_pairs: list[tuple[Vertex, Vertex]] = []
            for _, v1_prime in root_tuples:
                for v2_prime in R_map.get(v1_prime, []):
                    candidate_pairs.append((v1_prime, v2_prime))
            for _, v1_prime in other_tuples:
                for v2_prime in R_map.get(v1_prime, []):
                    candidate_pairs.append((v1_prime, v2_prime))
            for index, (v1_prime, v2_prime) in enumerate(candidate_pairs):
                path1, v1_node, v1_prime_node = sc1.get_path_node_steps(v1, v1_prime)
                path2, v2_node, v2_prime_node = sc2.get_path_node_steps(v2, v2_prime)
                if not path1 or not path2 or v1_node is None or v2_node is None or v1_prime_node is None or v2_prime_node is None:
                    continue
                desc1 = _construct_description_from_path(path1, v1_node, v1_prime_node)
                desc2 = _construct_description_from_path(path2, v2_node, v2_prime_node)
                score_pairs.append((desc1, desc2))
                score_meta.append((pair_idx, v1, v2, index))
    if not score_pairs:
        return [[] for _ in sc_pairs]
    time1 = time.time()
    scores = get_nli_remix_score_batch(score_pairs)
    time2 = time.time()
    for (pair_idx, v1, v2, index), score in zip(score_meta, scores):
        pair_map = pair_index_max_by_pair[pair_idx]
        key = (v1, v2)
        if key not in pair_map:
            pair_map[key] = {}
        prev = pair_map[key].get(index)
        if prev is None or score > prev:
            pair_map[key][index] = score
    all_results: list[list[tuple[Vertex, Vertex, float]]] = []
    for pair_map in pair_index_max_by_pair:
        raw_results: list[tuple[Vertex, Vertex, float]] = []
        for (v1, v2), index_max_map in pair_map.items():
            if not index_max_map:
                continue
            avg_score = sum(index_max_map.values()) / len(index_max_map)
            if avg_score > threshold:
                raw_results.append((v1, v2, avg_score))
        raw_results.sort(key=lambda x: x[2], reverse=True)
        used_v1: set[Vertex] = set()
        used_v2: set[Vertex] = set()
        results: list[tuple[Vertex, Vertex, float]] = []
        for v1, v2, score in raw_results:
            if v1 in used_v1 or v2 in used_v2:
                continue
            used_v1.add(v1)
            used_v2.add(v2)
            results.append((v1, v2, score))
        all_results.append(results)
    return all_results

# ---------------------------------------------------------------------------
# Hyper Simulation compatibility adapter
# ---------------------------------------------------------------------------
def convert_local_to_sim(
    local_hg: LocalHypergraph,
) -> Tuple[SimHypergraph, Dict[int, str], Dict[int, Vertex], Dict[int, List[SimHyperedge]], Dict[Vertex, int]]:
    """Translate the production Python hypergraph into contiguous Rust ids."""

    sim_hg = SimHypergraph()
    vertex_id_map: Dict[int, int] = {}
    node_text: Dict[int, str] = {}
    sim_id_to_vertex: Dict[int, Vertex] = {}
    node_to_edges: Dict[int, List[SimHyperedge]] = {}
    vertex_to_sim_id: Dict[Vertex, int] = {}
    for idx, vertex in enumerate(sorted(local_hg.vertices, key=lambda v: v.id)):
        sim_hg.add_node(vertex.text())
        vertex_id_map[vertex.id] = idx
        node_text[idx] = vertex.text()
        sim_id_to_vertex[idx] = vertex
        vertex_to_sim_id[vertex] = idx
    edge_id = 0
    for local_edge in local_hg.hyperedges:
        node_ids = {vertex_id_map[v.id] for v in local_edge.vertices if v.id in vertex_id_map}
        if not node_ids:
            continue
        sim_edge = SimHyperedge(node_ids, local_edge.desc, edge_id)
        sim_hg.add_hyperedge(sim_edge)
        for nid in node_ids:
            node_to_edges.setdefault(nid, []).append(sim_edge)
        edge_id += 1
    return sim_hg, node_text, sim_id_to_vertex, node_to_edges, vertex_to_sim_id
def build_delta_and_dmatch(
    query: SimHypergraph,
    data: SimHypergraph,
    query_texts: Dict[int, str],
    data_texts: Dict[int, str],
    query_node_edges: Dict[int, List[SimHyperedge]],
    data_node_edges: Dict[int, List[SimHyperedge]],
    allowed_pairs: Set[Tuple[int, int]],
    query_local_hg: LocalHypergraph,
    data_local_hg: LocalHypergraph,
    vertex_to_sim_id_q: Dict[Vertex, int],
    vertex_to_sim_id_d: Dict[Vertex, int],
    matched_vertices: dict[Vertex, set[Tuple[Vertex, float]]],
    cluster_sim_threshold: float = 0.75,
    dmatch_threshold: float = 0.3,
    branch_threshold: int = 5,
    is_multihop: bool = False,
) -> Tuple[Delta, DMatch]:
    """Register HC dependencies and D-match relations for the Rust solver."""

    delta_start = time.time()
    sc_logger = getLogger("semantic_cluster")
    sc_logger.debug("\t\tcalc the delta")
    delta = Delta()
    d_delta_matches: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    cluster_count = 0
    raw_pairs = calc_semantic_cluster_pairs(
        query_local_hg, data_local_hg, matched_vertices,
        cluster_sim_threshold, branch_threshold, is_multihop, logger=sc_logger
        )
    time1 = time.time()
    sc_logger.info(f"语义簇生成完成: 共 {len(raw_pairs)} 个原始簇对")
    candidate_cluster_pairs = []
    for sc_q, sc_d, sim_score in raw_pairs:
        q_vertices = sc_q.get_vertices()
        d_vertices = sc_d.get_vertices()
        q_edges = sc_q.hyperedges
        d_edges = sc_d.hyperedges
        q_triples = sc_q.to_triple() or []
        d_triples = sc_d.to_triple() or []
        q_triple_repr = str(q_triples[0]) if q_triples else "(no triple)"
        d_triple_repr = str(d_triples[0]) if d_triples else "(no triple)"
        q_text = sc_q.text()
        d_text = sc_d.text()
        sc_logger.info(
            f"→ 原始簇对 | score={sim_score:.3f}\n"
            f"  Q: text='{q_text}'\n"
            f"     triple={q_triple_repr}\n"
            f"     nodes={len(q_vertices)}, edges={len(q_edges)}\n"
            f"  D: text='{d_text}'\n"
            f"     triple={d_triple_repr}\n"
            f"     nodes={len(d_vertices)}, edges={len(d_edges)}"
        )
        if sim_score < 0.5:
            sc_logger.info(f"  → 跳过: 低相似度 ({sim_score:.3f})")
            continue
        q_vs = [v for v in q_vertices if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        d_vs = [v for v in d_vertices if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        if not q_vs or not d_vs:
            sc_logger.info(f"  → 跳过: 无名词节点 (Q:{len(q_vs)}/{len(q_vertices)}, D:{len(d_vs)}/{len(d_vertices)})")
            continue
        q_rep = min(q_vs, key=lambda v: v.id)
        d_rep = min(d_vs, key=lambda v: v.id)
        q_nid = vertex_to_sim_id_q.get(q_rep)
        d_nid = vertex_to_sim_id_d.get(d_rep)
        if q_nid is None or d_nid is None:
            sc_logger.info(f"  → 跳过: 映射缺失 (Q{q_rep.id}→{q_nid}, D{d_rep.id}→{d_nid})")
            continue
        q_es = list({e for v in q_vs if v in vertex_to_sim_id_q for e in query_node_edges.get(vertex_to_sim_id_q[v], []) if e})
        d_es = list({e for v in d_vs if v in vertex_to_sim_id_d for e in data_node_edges.get(vertex_to_sim_id_d[v], []) if e})
        sc_id = delta.add_sematic_cluster_pair(
            SimNode(q_nid, q_text),
            SimNode(d_nid, d_text),
            q_es,
            d_es
        )
        candidate_cluster_pairs.append({
            'sc_q': sc_q,
            'sc_d': sc_d,
            'sc_id': sc_id,
            'q_rep': q_rep,
            'd_rep': d_rep,
            'q_text': q_text,
            'd_text': d_text,
            'q_triple_repr': q_triple_repr,
            'd_triple_repr': d_triple_repr,
            'q_vertices': q_vertices,
            'd_vertices': d_vertices,
            'q_vs': q_vs,
            'd_vs': d_vs,
            'q_edges': q_edges,
            'd_edges': d_edges,
            'sim_score': sim_score,
        })
    if candidate_cluster_pairs:
        sc_pairs = [(md['sc_q'], md['sc_d']) for md in candidate_cluster_pairs]
        try:
            batch_results = calc_d_match_batch(sc_pairs, dmatch_threshold)
        except (AssertionError, AttributeError, IndexError) as e:
            sc_logger.warning(f"  → 批量匹配异常: {type(e).__name__}, 降级为空匹配")
            batch_results = [[] for _ in sc_pairs]
    else:
        batch_results = []
    for batch_idx, meta in enumerate(candidate_cluster_pairs):
        cluster_count += 1
        sc_id = meta['sc_id']
        q_rep = meta['q_rep']
        d_rep = meta['d_rep']
        q_text = meta['q_text']
        d_text = meta['d_text']
        q_triple_repr = meta['q_triple_repr']
        d_triple_repr = meta['d_triple_repr']
        q_vertices = meta['q_vertices']
        d_vertices = meta['d_vertices']
        q_vs = meta['q_vs']
        d_vs = meta['d_vs']
        q_edges = meta['q_edges']
        d_edges = meta['d_edges']
        sim_score = meta['sim_score']
        if batch_idx < len(batch_results):
            matches = {
                (vertex_to_sim_id_q[vq], vertex_to_sim_id_d[vd])
                for vq, vd, _ in batch_results[batch_idx]
                if vq in vertex_to_sim_id_q and vd in vertex_to_sim_id_d
            }
        else:
            matches = set()
        d_delta_matches[(sc_id, sc_id)] = matches
        sc_logger.info(
            f"→ 采纳 #{cluster_count} | score={sim_score:.3f}\n"
            f"  Q_rep=Q{q_rep.id}('{q_rep.text()}')\n"
            f"     full_text='{q_text}'\n"
            f"     triple={q_triple_repr}\n"
            f"     nodes={len(q_vertices)} (noun={len(q_vs)}), edges={len(q_edges)}\n"
            f"  D_rep=D{d_rep.id}('{d_rep.text()}')\n"
            f"     full_text='{d_text}'\n"
            f"     triple={d_triple_repr}\n"
            f"     nodes={len(d_vertices)} (noun={len(d_vs)}), edges={len(d_edges)}\n"
            f"  D-Match count: {len(matches)}"
        )
    sc_logger.info(f"语义簇构建完成: 原始 {len(raw_pairs)} → 有效 {cluster_count} 个簇对")
    time2 = time.time()
    return delta, DMatch.from_dict(d_delta_matches)
def _compute_standard_hyper_simulation(
    query_hg: LocalHypergraph,
    data_hg: LocalHypergraph,
    sigma_threshold: float = 0.75,
    b_threshold: int = 5,
    delta_threshold: float = 0.7,
) -> Tuple[Dict[int, Set[int]], Dict[int, Vertex], Dict[int, Vertex]]:
    """Execute h_v, HC, D-match, and the fixed-point solver."""

    sim_logger = getLogger("hyper_simulation")
    sim_logger.debug("\tStart Hyper Simulation")
    q_sim, q_texts, q_vertices, q_edges, q_vid_map = convert_local_to_sim(query_hg)
    d_sim, d_texts, d_vertices, d_edges, d_vid_map = convert_local_to_sim(data_hg)
    denial_start = time.time()
    sim_logger.debug("\tstart denial comment calc")
    dc_logger = getLogger("denial_comment")
    time1 = time.time()
    allowed, confidence_scores = compute_allowed_pairs_batch_with_score(q_vertices, d_vertices)
    time2 = time.time()
    q_vertices_list = list(q_vertices.values())
    d_vertices_list = list(d_vertices.values())
    time3 = time.time()
    match_vertices = get_top_k_matched_vertices_by_scores(q_vertices, d_vertices, confidence_scores, k=b_threshold)
    def type_same_fn(x_id: int, y_id: int) -> bool:
        """Expose the frozen allowed-pair relation to the Rust fixed point."""

        return (x_id, y_id) in allowed
    q_sim.set_type_same_fn(type_same_fn)
    d_sim.set_type_same_fn(type_same_fn)
    denial_end = time.time()
    dc_logger.info(f"\tdenial comment cost {denial_end - denial_start}s")
    sim_logger.debug(f"\tdenial comment cost {denial_end - denial_start}s")
    sim_logger.debug("\tstart build delta and d-match")
    delta, d_match = build_delta_and_dmatch(
        q_sim, d_sim, q_texts, d_texts, q_edges, d_edges, allowed,
        query_local_hg=query_hg,
        data_local_hg=data_hg,
        vertex_to_sim_id_q=q_vid_map,
        vertex_to_sim_id_d=d_vid_map,
        matched_vertices=match_vertices,
        dmatch_threshold=delta_threshold,
        cluster_sim_threshold=sigma_threshold,
        branch_threshold=b_threshold
    )
    start_time = time.time()
    sim_logger.info("\t执行超图模拟...")
    simulation = SimHypergraph.get_hyper_simulation(q_sim, d_sim, delta, d_match)
    sim_logger.info("\t=== Hyper Simulation Mapping ===")
    for q_id, d_ids in sorted(simulation.items()):
        q_text = q_vertices[q_id].text() if q_id in q_vertices else f"[Q{q_id}]"
        if d_ids:
            d_items = []
            for d_id in sorted(d_ids):
                if d_id in d_vertices:
                    d_text = d_vertices[d_id].text()
                    d_items.append(f"D{d_id}: '{d_text}'")
                else:
                    d_items.append(f"D{d_id}")
            targets = ", ".join(d_items)
        else:
            targets = "-"
        sim_logger.info(f"\t  Q{q_id}: '{q_text}' → {targets}")
    sim_logger.info("\t================================")
    end_time = time.time()
    sim_logger.info(f"\t模拟完成: {len(simulation)}个映射")
    sim_logger.info(f"\thyper simulation main cost {end_time - start_time}s")
    return simulation, q_vertices, d_vertices

# ---------------------------------------------------------------------------
# Task-level Consistency adapter
# ---------------------------------------------------------------------------
def generate_instance_id(query: str) -> str:
    """Return a whitespace-insensitive identifier for a query."""

    normalized = ''.join(query.split()).lower()
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]
def load_hypergraphs_for_instance(
    query_instance: QueryInstance,
    dataset_name: str = "hotpotqa",
    base_dir: str = "data/hypergraph"
) -> Tuple[LocalHypergraph, List[LocalHypergraph]]:
    """Load query and per-context hypergraphs for task processing."""

    instance_id = generate_instance_id(query_instance.query)
    current_query_id.set(instance_id)
    instance_dir = Path(base_dir) / dataset_name / instance_id
    if not instance_dir.exists():
        raise FileNotFoundError(
            f"Hypergraphs not found. Run build_hypergraph_batch first.\nDirectory: {instance_dir}"
        )
    query_hg = LocalHypergraph.load(str(instance_dir / "query.pkl"))
    data_hgs = []
    for idx in (range(len(query_instance.data))):
        data_path = instance_dir / f"data_{idx}.pkl"
        if data_path.exists():
            try:
                data_hgs.append(LocalHypergraph.load(str(data_path)))
            except Exception:
                data_hgs.append(None)
        else:
            data_hgs.append(None)
    return query_hg, data_hgs
def is_critical_vertex(vertex: Vertex) -> bool:
    """Identify vertices required by the binary consistency check."""

    # Query placeholders are the point of the question; an SCONJ '?Where'
    # carries no entity/POS signal and would otherwise escape the critical
    # set, letting place-free documents pass as consistent (L3 finding F5).
    if vertex.is_query():
        return True
    if any(e != Entity.NOT_ENTITY for e in vertex.ents):
        return True
    dep = vertex.dep()
    if dep in {Dep.nsubj, Dep.nsubjpass, Dep.dobj, Dep.iobj, Dep.pobj}:
        return True
    if vertex.pos_equal(Pos.NOUN) or vertex.pos_equal(Pos.PROPN):
        return True
    return False
def get_distance(text1: str, text2: str) -> float:
    """Return cosine distance between the cached embeddings of two texts."""

    emb1 = get_embedding_batch([text1])[0]
    emb2 = get_embedding_batch([text2])[0]
    return 1.0 - cosine_similarity(emb1, emb2)
def consistent_detection(
    query: QueryInstance,
    query_hg: LocalHypergraph,
    data_hg: LocalHypergraph,
    query_text: str,
    data_text: str,
    distance_threshold: float = 0.25
) -> Tuple[bool, str]:
    """Run the binary inconsistency decision and return evidence."""

    consistent_logger = getLogger("consistency")
    consistent_logger.debug("Enter the consistent detection")
    distance = get_distance(query_text, data_text)
    consistent_logger.info(f"Compute cosine distance: {distance:.4f}, threshold={distance_threshold}")
    if distance <= distance_threshold:
        evidence = f"[CONSISTENT] Distance={distance:.4f} ≤ threshold={distance_threshold}"
        consistent_logger.info(evidence)
        return False, evidence
    consistent_logger.debug("Running hyper_simulation...")
    simulation, q_vertices_map, d_vertices_map = compute_hyper_simulation(query_hg, data_hg)
    critical_q_vertices = [v for v in query_hg.vertices if is_critical_vertex(v)]
    consistent_logger.info(f"Critical Q vertices to cover: {len(critical_q_vertices)}")
    for v in critical_q_vertices:
        consistent_logger.info(f"  • Q{v.id}: '{v.text()}'")
    if not critical_q_vertices:
        evidence = f"[CONSISTENT] No critical vertices to cover (distance={distance:.4f} > threshold)"
        consistent_logger.info(evidence)
        return False, evidence
    evidence_lines = []
    has_contradiction = False
    sim_id_to_q_vertex = {sim_id: q_vertices_map[sim_id] for sim_id in q_vertices_map}
    for q_vertex in critical_q_vertices:
        matched = False
        target_sim_id = None
        for sim_id, v in q_vertices_map.items():
            if v.id == q_vertex.id:
                target_sim_id = sim_id
                break
        if target_sim_id is not None and target_sim_id in simulation:
            sim_d_ids = simulation[target_sim_id]
            if len(sim_d_ids) > 0:
                matched = True
        if not matched:
            has_contradiction = True
            evidence_lines.append(f"Q vertex unmatched in D: '{q_vertex.text()}' (ID={q_vertex.id})")
    if evidence_lines:
        consistent_logger.info("Unmatched critical vertices in Q:")
        for line in evidence_lines:
            consistent_logger.info(f"  • {line}")
    else:
        consistent_logger.info("All critical Q vertices are matched in D.")
    if has_contradiction:
        evidence = (
            f"[CONTRADICTION] Distance={distance:.4f} > threshold={distance_threshold}\n"
            + "\n".join(f"  • {line}" for line in evidence_lines)
        )
    else:
        evidence = (
            f"[CONSISTENT] Distance={distance:.4f} > threshold but structural coverage satisfied\n"
            f"  ✓ All {len(critical_q_vertices)} critical Q vertices matched in D via hyper simulation"
        )
    consistent_logger.info(evidence)
    return has_contradiction, evidence
def query_fixup(query: QueryInstance, dataset_name: str = "hotpotqa", base_dir: str = "data/hypergraph") -> QueryInstance:
    """Filter or fuse task contexts using structural consistency decisions."""

    # Importing fusion at call time avoids a cycle through the public
    # semantic-cluster facade while compatibility dependencies are loading.
    from hufu.hypergraph.union import MultiHopFusion

    query_hg, data_hgs = load_hypergraphs_for_instance(query, dataset_name, base_dir=base_dir)
    consistent_logger = getLogger("consistency", level="DEBUG")
    hg_logger = getLogger("hypergraph", level="DEBUG")
    consistent_logger.info(f"[QueryFixup] Enter query_fixup for dataset: {dataset_name}")
    consistent_logger.info(f"[QueryFixup] Query: '{query}'")
    consistent_logger.info(f"[QueryFixup] Evidence count: {len(query.data)}")
    hg_logger.debug("=== Query Text===")
    hg_logger.debug(f"{query.query}")
    hg_logger.debug("=== Query Hypergraph===")
    query_hg.log_summary(hg_logger)
    for i, d in enumerate(data_hgs):
        if d:
            hg_logger.debug(f"=== Query Text #{i}===")
            hg_logger.debug(f"{query.data[i]}")
            hg_logger.debug(f"=== Data Hypergraph #{i} ===")
            d.log_summary(hg_logger)
    multi_hop_tasks = {"musique", "multihop"}
    if dataset_name in multi_hop_tasks:
        fusion = MultiHopFusion()
        valid_data_hgs = [hg for hg in data_hgs if hg is not None]
        valid_indices = [i for i, hg in enumerate(data_hgs) if hg is not None]
        valid_hgs = [data_hgs[i] for i in valid_indices]
        valid_texts = [query.data[i] for i in valid_indices]
        consistent_logger.info(f"[Multi-hop] Valid hypergraphs: {len(valid_hgs)} / {len(data_hgs)}")
        if not valid_hgs:
            consistent_logger.warning("[Multi-hop] No valid hypergraphs, fallback to original data")
            query.fixed_data = query.data
            return query
        consistent_logger.debug("[Multi-hop] Running MultiHopFusion process...")
        context = fusion.process(query_hg, valid_hgs, valid_texts)
        active_sources = set()
        for line in context.split("\n"):
            if line.startswith("[") and "]" in line and ("SUPPORTS" in line or "Confidence" in line):
                try:
                    source_id = int(line.split("]")[0].replace("[", "").strip())
                    if "covers 0 query components" not in line:
                        active_sources.add(source_id)
                except (ValueError, IndexError):
                    pass
        consistent_logger.info(f"[Multi-hop] Active evidence sources (with matches): {sorted(active_sources)}")
        unused = set(range(len(valid_texts))) - active_sources
        if unused:
            consistent_logger.debug(f"[Multi-hop] Unused evidence sources: {sorted(unused)}")
        query.fixed_data = [context]
        consistent_logger.info(f"[Multi-hop] fixed_data generated: 1 item, {len(context)} chars")
        consistent_logger.debug(f"[Multi-hop] fixed_data preview:\n{context[:500]}...")
        return query
    consistent_logger.info(f"[Single-hop] Processing single-document consistency for {len(query.data)} documents")
    fixed_data = []
    consistent_count = 0
    inconsistent_count = 0
    for doc_text, data_hg in tqdm(zip(query.data, data_hgs), desc='\tHyper Simulation for Query.', leave=True):
        if data_hg is None:
            fixed_data.append(doc_text)
            continue
        has_contradiction, evidence = consistent_detection(
            query, query_hg, data_hg, query.query, doc_text
        )
        if has_contradiction:
            inconsistent_count += 1
            fixed_doc = (
                f"{doc_text}\n\n"
                f"[INCONSISTENT DETECTED - USE WITH CAUTION]\n"
                f"Evidence:\n{evidence}"
            )
        else:
            consistent_count += 1
            fixed_doc = doc_text
        fixed_data.append(fixed_doc)
    consistent_logger.info(f"[Single-hop] Processing completed: {consistent_count} consistent, {inconsistent_count} inconsistent")
    consistent_logger.info(f"[Single-hop] fixed_data generated: {len(fixed_data)} items")
    query.fixed_data = fixed_data
    return query


def _with_standard_dependencies(function):
    """Wrap an entry point with the optional dependency loader."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        _load_standard_dependencies()
        return function(*args, **kwargs)

    return wrapped


_STANDARD_ENTRYPOINT_NAMES = (
    "combine_hyperedges_to_cluster",
    "calc_semantic_cluster_pairs",
    "build_descendant_cluster",
    "calc_embedding_for_cluster_batch",
    "get_semantic_cluster_pairs",
    "node_sequence_to_text",
    "get_d_match",
    "query_same_type",
    "_construct_description_from_path",
    "calc_d_match",
    "calc_d_match_batch",
    "convert_local_to_sim",
    "build_delta_and_dmatch",
    "_compute_standard_hyper_simulation",
    "load_hypergraphs_for_instance",
    "is_critical_vertex",
    "get_distance",
    "consistent_detection",
    "query_fixup",
)
for _standard_entrypoint_name in _STANDARD_ENTRYPOINT_NAMES:
    globals()[_standard_entrypoint_name] = _with_standard_dependencies(
        globals()[_standard_entrypoint_name]
    )
del _standard_entrypoint_name


_STANDARD_SYMBOLS = frozenset(
    {
        "SemanticCluster",
        "TarjanLCA",
        "_better_path",
        "_cluster_sort_key",
        "_construct_description_from_path",
        "_formal_text_of",
        "_get_matched_vertices",
        "_hyperedge_signature",
        "_legal_vertices",
        "_path_score",
        "_vertex_sort_key",
        "abstraction_lca",
        "build_delta_and_dmatch",
        "build_descendant_cluster",
        "calc_d_match",
        "calc_d_match_batch",
        "calc_embedding_for_cluster_batch",
        "calc_semantic_cluster_pairs",
        "combine_hyperedges_to_cluster",
        "consistent_detection",
        "convert_local_to_sim",
        "generate_instance_id",
        "get_d_match",
        "get_distance",
        "get_semantic_cluster_pairs",
        "is_critical_vertex",
        "load_hypergraphs_for_instance",
        "node_sequence_to_text",
        "query_fixup",
        "query_same_type",
    }
)


def get_standard_symbol(name: str):
    """Return a named symbol for compatibility facade modules."""

    if name not in _STANDARD_SYMBOLS:
        raise AttributeError(name)
    # Importing structural class definitions is dependency-free.  Their
    # constructors and executable compatibility functions still load the frozen
    # model stack through the wrappers above.
    if name not in {"SemanticCluster", "TarjanLCA"}:
        _load_standard_dependencies()
    return globals()[name]
