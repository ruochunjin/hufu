"""Semantic-cluster proposal generation and compatibility entry points.

HCCalc is a structural proposal generator, not the semantic HC oracle.  It
combines four bounded routes and returns candidates *before* any HC threshold:

``relation_seeded``
    Grow connected clusters around the strongest singleton relation pairs.
``assignment_cover``
    Select a good data fact for every query fact and connect the selections.
``h_v_endpoint_shortest_path``
    Connect h_v-compatible endpoint vertices through a local hyperpath.
``father_atom_incidence_closure``
    Keep parser fragments belonging to one father-linked fact together.

Every proposal is restricted to one source context.  Sorting and content IDs
make the output independent of Python set/dict enumeration order.  The
compatibility functions at the bottom delegate lazily to the embedding/path
implementation so structural proposal imports do not load model packages.
"""

from __future__ import annotations

import importlib
from collections import deque
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .alignment import (
    assignment_cover_subsets,
    diverse_structural_shortlist,
    edge_alignment_score,
    father_atom_incidence_closures,
)
from .config import HCCalcConfig
from .contracts import Cluster, HCCandidate, Hypergraph, Pair

ScoreMatrix = Mapping[tuple[str, str], float]


def enumerate_connected_query_clusters(
    query: Hypergraph, config: HCCalcConfig
) -> tuple[Cluster, ...]:
    """Enumerate bounded query-side patterns without looking at data labels."""

    result: list[Cluster] = []
    for context_id in query.context_ids:
        edge_ids = tuple(edge.id for edge in query.edges_in_context(context_id))
        adjacency = query.edge_adjacency(context_id)
        levels: dict[int, set[frozenset[str]]] = {
            1: {frozenset({edge_id}) for edge_id in edge_ids}
        }
        for size in range(1, min(config.max_cluster_size, len(edge_ids)) + 1):
            values = sorted(levels.get(size, set()), key=lambda value: tuple(sorted(value)))
            for value in values:
                result.append(Cluster.from_edges(query, value, context_id=context_id))
            if size == config.max_cluster_size:
                break
            expanded: set[frozenset[str]] = set()
            for value in values:
                frontier = set().union(*(adjacency[edge_id] for edge_id in value)) - set(value)
                for edge_id in sorted(frontier):
                    expanded.add(value | {edge_id})
            # Query patterns are normally tiny; the beam is nevertheless
            # applied per size to keep the worst case explicit.
            levels[size + 1] = set(
                sorted(expanded, key=lambda value: tuple(sorted(value)))[
                    : config.structural_beam
                ]
            )
    return tuple(sorted(result, key=lambda value: (len(value.edge_ids), value.id)))


def enumerate_hc_candidates(
    query: Hypergraph,
    data: Hypergraph,
    hv_allowed_pairs: Iterable[Pair],
    singleton_scores: ScoreMatrix,
    config: HCCalcConfig,
    *,
    query_clusters: Sequence[Cluster] | None = None,
) -> tuple[HCCandidate, ...]:
    """Return the stable threshold-free HC candidate universe.

    ``singleton_scores[(query_edge_id, data_edge_id)]`` is computed once by
    the HC scorer.  Reusing it inside every route is the main efficiency gain:
    multi-edge structural search does not re-encode every connected subset.
    """

    if query.side != "query" or data.side != "data":
        raise ValueError("HCCalc expects a query graph and a data graph")
    clusters = tuple(
        enumerate_connected_query_clusters(query, config)
        if query_clusters is None
        else query_clusters
    )
    for cluster in clusters:
        cluster.validate(query)
    hv_pairs = frozenset((str(left), str(right)) for left, right in hv_allowed_pairs)

    candidates: list[HCCandidate] = []
    for query_cluster in clusters:
        score_rows = tuple(
            {
                data_edge.id: float(singleton_scores.get((query_edge_id, data_edge.id), -1.0))
                for data_edge in data.hyperedges
            }
            for query_edge_id in query_cluster.edge_ids
        )
        for context_id in data.context_ids:
            edge_ids = tuple(edge.id for edge in data.edges_in_context(context_id))
            if not edge_ids:
                continue
            adjacency = data.edge_adjacency(context_id)
            routed: dict[frozenset[str], set[str]] = {}

            if "relation_seeded" in config.proposal_routes:
                _add_route(
                    routed,
                    _relation_seeded_subsets(
                        edge_ids, adjacency, score_rows, config=config
                    ),
                    "relation_seeded",
                )

            if "assignment_cover" in config.proposal_routes:
                _add_route(
                    routed,
                    _assignment_cover(
                        edge_ids, adjacency, score_rows, config=config
                    ),
                    "assignment_cover",
                )

            if "h_v_endpoint_shortest_path" in config.proposal_routes:
                _add_route(
                    routed,
                    _endpoint_path_subsets(
                        query,
                        data,
                        query_cluster,
                        context_id,
                        hv_pairs,
                        score_rows,
                        config=config,
                    ),
                    "h_v_endpoint_shortest_path",
                )

            if "father_atom_incidence_closure" in config.proposal_routes:
                father_by_edge = {
                    edge_id: data.edge(edge_id).father_id for edge_id in edge_ids
                }
                vertices_by_edge = {
                    edge_id: set(data.edge(edge_id).vertex_ids) for edge_id in edge_ids
                }
                best_singleton = {
                    edge_id: max(
                        (row.get(edge_id, -1.0) for row in score_rows),
                        default=-1.0,
                    )
                    for edge_id in edge_ids
                }
                _add_route(
                    routed,
                    father_atom_incidence_closures(
                        edge_ids,
                        father_by_edge=father_by_edge,
                        vertices_by_edge=vertices_by_edge,
                        score_by_edge=best_singleton,
                        max_cluster_size=config.closure_max_cluster_size,
                        max_starts=config.structural_beam,
                    ),
                    "father_atom_incidence_closure",
                )

            candidates.extend(
                _materialize_shortlist(
                    query_cluster,
                    data,
                    context_id,
                    routed,
                    score_rows,
                    config=config,
                )
            )

    # One logical query/data cluster pair is emitted once even if several
    # routes found it.  The route union is preserved in the candidate record.
    by_id: dict[str, HCCandidate] = {}
    for candidate in candidates:
        previous = by_id.get(candidate.id)
        if previous is not None and previous != candidate:
            raise AssertionError("content-derived HC id collision")
        by_id[candidate.id] = candidate
    return tuple(by_id[key] for key in sorted(by_id))


def _relation_seeded_subsets(
    edge_ids: Sequence[str],
    adjacency: Mapping[str, set[str]],
    score_rows: Sequence[Mapping[str, float]],
    *,
    config: HCCalcConfig,
) -> tuple[frozenset[str], ...]:
    seeds = {
        edge_id
        for row in score_rows
        for edge_id, _ in sorted(
            ((edge_id, float(row.get(edge_id, -1.0))) for edge_id in edge_ids),
            key=lambda value: (-value[1], value[0]),
        )[: config.seed_edges]
    }
    all_values: set[frozenset[str]] = {frozenset({edge_id}) for edge_id in seeds}
    level = set(all_values)
    for _size in range(2, min(config.max_cluster_size, len(edge_ids)) + 1):
        expanded: set[frozenset[str]] = set()
        for value in level:
            frontier = set().union(*(adjacency[edge_id] for edge_id in value)) - set(value)
            expanded.update(value | {edge_id} for edge_id in frontier)
        ranked = sorted(
            expanded,
            key=lambda value: _subset_rank(value, score_rows, config),
        )[: config.structural_beam]
        level = set(ranked)
        all_values.update(level)
        if not level:
            break
    return tuple(sorted(all_values, key=lambda value: (len(value), tuple(sorted(value)))))


def _assignment_cover(
    edge_ids: Sequence[str],
    adjacency: Mapping[str, set[str]],
    score_rows: Sequence[Mapping[str, float]],
    *,
    config: HCCalcConfig,
) -> tuple[frozenset[str], ...]:
    # The reusable assignment primitive uses compact integer keys.  Mapping
    # local edge ids to integers keeps the public graph schema string-based.
    to_index = {edge_id: index for index, edge_id in enumerate(sorted(edge_ids))}
    to_edge = {index: edge_id for edge_id, index in to_index.items()}
    integer_adjacency = {
        to_index[edge_id]: {to_index[neighbor] for neighbor in neighbors}
        for edge_id, neighbors in adjacency.items()
    }
    integer_rows = tuple(
        {to_index[edge_id]: score for edge_id, score in row.items() if edge_id in to_index}
        for row in score_rows
    )
    subsets = assignment_cover_subsets(
        integer_adjacency,
        integer_rows,
        max_cluster_size=config.max_cluster_size,
        choices_per_query_edge=config.choices_per_query_edge,
        max_combinations=config.max_assignment_combinations,
        query_weight=config.query_coverage_weight,
    )
    return tuple(frozenset(to_edge[index] for index in value) for value in subsets)


def _endpoint_path_subsets(
    query: Hypergraph,
    data: Hypergraph,
    query_cluster: Cluster,
    context_id: str,
    hv_pairs: frozenset[Pair],
    score_rows: Sequence[Mapping[str, float]],
    *,
    config: HCCalcConfig,
) -> tuple[frozenset[str], ...]:
    query_vertices = {
        vertex_id
        for edge_id in query_cluster.edge_ids
        for vertex_id in query.edge(edge_id).vertex_ids
        if query.vertex(vertex_id).matchable
    }
    data_edges = data.edges_in_context(context_id)
    incident = {
        vertex.id: tuple(
            edge.id for edge in data_edges if vertex.id in edge.vertex_ids
        )
        for vertex in data.vertices
    }
    best_by_data_edge = {
        edge.id: max((row.get(edge.id, -1.0) for row in score_rows), default=-1.0)
        for edge in data_edges
    }
    endpoints: dict[str, tuple[str, ...]] = {}
    for query_id in sorted(query_vertices):
        values = {
            data_id
            for left, data_id in hv_pairs
            if left == query_id and incident.get(data_id)
        }
        endpoints[query_id] = tuple(
            sorted(
                values,
                key=lambda data_id: (
                    -max(best_by_data_edge[edge_id] for edge_id in incident[data_id]),
                    data_id,
                ),
            )[: config.endpoint_match_limit]
        )

    adjacency = data.edge_adjacency(context_id)
    proposed: set[frozenset[str]] = set()
    for query_id in sorted(endpoints):
        for data_id in endpoints[query_id]:
            proposed.update(frozenset({edge_id}) for edge_id in incident[data_id])
    for left_query, right_query in combinations(sorted(endpoints), 2):
        for left_data in endpoints[left_query]:
            for right_data in endpoints[right_query]:
                path = _shortest_edge_path(
                    adjacency,
                    set(incident[left_data]),
                    set(incident[right_data]),
                    max_size=config.max_cluster_size,
                )
                if path is not None:
                    proposed.add(frozenset(path))
    return tuple(sorted(proposed, key=lambda value: (len(value), tuple(sorted(value)))))


def _shortest_edge_path(
    adjacency: Mapping[str, set[str]],
    starts: set[str],
    goals: set[str],
    *,
    max_size: int,
) -> tuple[str, ...] | None:
    queue = deque((edge_id, (edge_id,)) for edge_id in sorted(starts))
    visited = set(starts)
    while queue:
        current, path = queue.popleft()
        if current in goals:
            return path
        if len(path) >= max_size:
            continue
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, (*path, neighbor)))
    return None


def _materialize_shortlist(
    query_cluster: Cluster,
    data: Hypergraph,
    context_id: str,
    routed: Mapping[frozenset[str], set[str]],
    score_rows: Sequence[Mapping[str, float]],
    *,
    config: HCCalcConfig,
) -> tuple[HCCandidate, ...]:
    values: list[tuple[frozenset[str], set[str]]] = []
    for subset, routes in routed.items():
        cap = (
            config.closure_max_cluster_size
            if "father_atom_incidence_closure" in routes
            else config.max_cluster_size
        )
        if not subset or len(subset) > cap:
            continue
        try:
            Cluster.from_edges(data, subset, context_id=context_id)
        except ValueError:
            continue
        values.append((subset, routes))

    singletons = [(value, routes) for value, routes in values if len(value) == 1]
    ordinary = [
        (value, routes)
        for value, routes in values
        if 1 < len(value) <= config.max_cluster_size
    ]
    closures = [
        (value, routes)
        for value, routes in values
        if len(value) > config.max_cluster_size
    ]
    singletons.sort(key=lambda item: _subset_rank(item[0], score_rows, config))
    ordinary.sort(key=lambda item: _subset_rank(item[0], score_rows, config))
    closures.sort(key=lambda item: _subset_rank(item[0], score_rows, config))

    chosen = singletons[: config.singleton_shortlist]
    ordinary_by_subset = {subset: routes for subset, routes in ordinary}
    diverse_subsets = diverse_structural_shortlist(
        ordinary_by_subset,
        quota=config.multi_edge_shortlist,
        rank_key=lambda value: _subset_rank(value, score_rows, config),
        signatures=ordinary_by_subset,
        diversity_slots=1,
    )
    chosen.extend(
        _quota_multi_edge(
            [(subset, ordinary_by_subset[subset]) for subset in diverse_subsets],
            config,
        )
    )
    chosen.extend(closures[: config.closure_shortlist])
    result = []
    for subset, routes in chosen:
        aligned = edge_alignment_score(
            score_rows, subset, query_weight=config.query_coverage_weight
        )
        result.append(
            HCCandidate(
                query_cluster=query_cluster,
                data_cluster=Cluster.from_edges(data, subset, context_id=context_id),
                routes=tuple(routes),
                structural_score=aligned.score,
            )
        )
    return tuple(sorted(result, key=lambda value: value.id))


def _quota_multi_edge(
    values: Sequence[tuple[frozenset[str], set[str]]], config: HCCalcConfig
) -> list[tuple[frozenset[str], set[str]]]:
    chosen: list[tuple[frozenset[str], set[str]]] = []
    chosen_sets: set[frozenset[str]] = set()
    for size, quota in config.multi_edge_size_quotas:
        for item in (item for item in values if len(item[0]) == size):
            if sum(len(value) == size for value, _ in chosen) >= quota:
                break
            chosen.append(item)
            chosen_sets.add(item[0])
    for item in values:
        if len(chosen) >= config.multi_edge_shortlist:
            break
        if item[0] not in chosen_sets:
            chosen.append(item)
            chosen_sets.add(item[0])
    return chosen[: config.multi_edge_shortlist]


def _subset_rank(
    value: frozenset[str],
    score_rows: Sequence[Mapping[str, float]],
    config: HCCalcConfig,
) -> tuple[float, float, int, tuple[str, ...]]:
    aligned = edge_alignment_score(
        score_rows, value, query_weight=config.query_coverage_weight
    )
    return (
        -aligned.score,
        -aligned.minimum_query_coverage,
        len(value),
        tuple(sorted(value)),
    )


def _add_route(
    target: dict[frozenset[str], set[str]],
    subsets: Iterable[frozenset[str]],
    route: str,
) -> None:
    for subset in subsets:
        target.setdefault(frozenset(subset), set()).add(route)


def _solver_module():
    """Return the canonical module containing both solver implementations."""

    return importlib.import_module(
        "hufu.engine.hyper_simulation"
    )


def calc_semantic_cluster_pairs(*args, **kwargs):
    """Compatibility entry point for embedding/path HCCalc."""

    return _solver_module().get_standard_symbol("calc_semantic_cluster_pairs")(
        *args, **kwargs
    )


def get_semantic_cluster_pairs(*args, **kwargs):
    """Compatibility entry point for pre-threshold cluster enumeration."""

    return _solver_module().get_standard_symbol("get_semantic_cluster_pairs")(
        *args, **kwargs
    )


def get_d_match(*args, **kwargs):
    """Compatibility entry point for the cluster-local role matcher."""

    return _solver_module().get_standard_symbol("get_d_match")(*args, **kwargs)


def __getattr__(name: str):
    """Expose solver classes, notably ``SemanticCluster``, on demand."""

    if name not in _STANDARD_SYMBOLS:
        raise AttributeError(name)
    try:
        return _solver_module().get_standard_symbol(name)
    except AttributeError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error


_STANDARD_SYMBOLS = frozenset(
    {
        "SemanticCluster",
        "TarjanLCA",
        "_better_path",
        "_cluster_sort_key",
        "_formal_text_of",
        "_get_matched_vertices",
        "_hyperedge_signature",
        "_legal_vertices",
        "_path_score",
        "_vertex_sort_key",
        "abstraction_lca",
        "build_descendant_cluster",
        "calc_embedding_for_cluster_batch",
        "calc_semantic_cluster_pairs",
        "combine_hyperedges_to_cluster",
        "get_d_match",
        "get_semantic_cluster_pairs",
        "node_sequence_to_text",
    }
)


__all__ = [
    "ScoreMatrix",
    "calc_semantic_cluster_pairs",
    "enumerate_connected_query_clusters",
    "enumerate_hc_candidates",
    "get_d_match",
    "get_semantic_cluster_pairs",
]
