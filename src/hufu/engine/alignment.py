"""Explainable edge-alignment primitives shared by HCCalc proposal routes.

The expensive way to score a multi-hyperedge cluster is to concatenate the
whole cluster and run the embedding encoder again for every connected subset.
Relation-seeded HCCalc has already computed the complete singleton
query-edge/data-edge similarity matrix.  This module reuses that matrix to
measure two explicit properties:

* query coverage: every query fact should have a comparable data fact;
* data coherence: the selected data facts should belong to that comparison,
  rather than being arbitrary connector facts.

It also constructs a small set of assignment-cover clusters: choose one of
the best data facts for each query fact and connect those choices by stable
shortest local hyperpaths.  Both operations are deterministic, context-local,
and contain no task labels or language-model calls.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Hashable, Iterable, Mapping, Sequence, TypeVar

EdgeKey = TypeVar("EdgeKey", bound=Hashable)


@dataclass(frozen=True, slots=True)
class EdgeAlignmentScore:
    """Traceable decomposition of one query/data cluster score."""

    score: float
    query_coverage: float
    data_coherence: float
    minimum_query_coverage: float
    maximum_edge_similarity: float


def edge_alignment_score(
    score_by_query_edge: Sequence[Mapping[EdgeKey, float]],
    data_subset: Sequence[EdgeKey] | frozenset[EdgeKey] | set[EdgeKey],
    *,
    query_weight: float = 0.75,
) -> EdgeAlignmentScore:
    """Aggregate a singleton edge-pair matrix for one data cluster.

    ``query_weight`` deliberately favors recall: a document may split one
    query fact over several hyperedges and may need a low-similarity connector.
    The data-coherence term still penalizes clusters padded with unrelated
    facts.  Values stay on the same scale as the underlying cosine scores.
    """

    if not 0.0 <= query_weight <= 1.0:
        raise ValueError("query_weight must be in [0, 1]")
    subset = tuple(
        sorted(
            set(data_subset),
            key=lambda value: (str(type(value)), str(value)),
        )
    )
    if not score_by_query_edge or not subset:
        return EdgeAlignmentScore(-1.0, -1.0, -1.0, -1.0, -1.0)

    per_query = tuple(
        max(float(scores.get(index, -1.0)) for index in subset)
        for scores in score_by_query_edge
    )
    per_data = tuple(
        max(float(scores.get(index, -1.0)) for scores in score_by_query_edge)
        for index in subset
    )
    query_coverage = sum(per_query) / len(per_query)
    data_coherence = sum(per_data) / len(per_data)
    score = query_weight * query_coverage + (1.0 - query_weight) * data_coherence
    return EdgeAlignmentScore(
        score=float(score),
        query_coverage=float(query_coverage),
        data_coherence=float(data_coherence),
        minimum_query_coverage=float(min(per_query)),
        maximum_edge_similarity=float(max(per_query)),
    )


def _shortest_path_to_subset(
    adjacency: Mapping[int, set[int]],
    sources: set[int],
    target: int,
    *,
    max_nodes: int,
) -> tuple[int, ...] | None:
    if target in sources:
        return (target,)
    queue = deque((source, (source,)) for source in sorted(sources))
    visited = set(sources)
    while queue:
        current, path = queue.popleft()
        if len(path) >= max_nodes:
            continue
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor in visited:
                continue
            next_path = (*path, neighbor)
            if neighbor == target:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))
    return None


def connect_assignment(
    adjacency: Mapping[int, set[int]],
    assignment: Sequence[int],
    *,
    max_cluster_size: int,
) -> frozenset[int] | None:
    """Connect one query-edge assignment by stable local shortest paths."""

    targets = tuple(dict.fromkeys(int(value) for value in assignment))
    if not targets or max_cluster_size <= 0:
        return None
    selected = {targets[0]}
    for target in targets[1:]:
        path = _shortest_path_to_subset(
            adjacency,
            selected,
            target,
            max_nodes=max_cluster_size - len(selected) + 1,
        )
        if path is None:
            return None
        selected.update(path)
        if len(selected) > max_cluster_size:
            return None
    return frozenset(selected)


def assignment_cover_subsets(
    adjacency: Mapping[int, set[int]],
    score_by_query_edge: Sequence[Mapping[int, float]],
    *,
    max_cluster_size: int,
    choices_per_query_edge: int = 2,
    max_combinations: int = 64,
    query_weight: float = 0.75,
) -> tuple[frozenset[int], ...]:
    """Return connected clusters covering high-ranked facts for every query edge.

    The bounded Cartesian product is small for real query clusters (normally
    one to three hyperedges).  Ranking by the same explainable edge-alignment
    score makes truncation stable and reproducible.
    """

    if choices_per_query_edge <= 0 or max_combinations <= 0:
        raise ValueError("assignment-cover budgets must be positive")
    if not score_by_query_edge:
        return ()
    choices: list[tuple[int, ...]] = []
    for scores in score_by_query_edge:
        ranked = tuple(
            index
            for index, _score in sorted(
                ((int(index), float(score)) for index, score in scores.items()),
                key=lambda value: (-value[1], value[0]),
            )[:choices_per_query_edge]
        )
        if not ranked:
            return ()
        choices.append(ranked)

    proposed: set[frozenset[int]] = set()
    for offset, assignment in enumerate(product(*choices)):
        if offset >= max_combinations:
            break
        connected = connect_assignment(
            adjacency, assignment, max_cluster_size=max_cluster_size
        )
        if connected is not None:
            proposed.add(connected)

    def rank(value: frozenset[int]) -> tuple[float, float, int, tuple[int, ...]]:
        aligned = edge_alignment_score(
            score_by_query_edge, value, query_weight=query_weight
        )
        return (
            -aligned.score,
            -aligned.minimum_query_coverage,
            len(value),
            tuple(sorted(value)),
        )

    return tuple(sorted(proposed, key=rank))


def anchor_incidence_subsets(
    edge_ids: Sequence[EdgeKey],
    *,
    vertices_by_edge: Mapping[EdgeKey, set[Hashable] | frozenset[Hashable]],
    anchor_vertices: set[Hashable] | frozenset[Hashable],
    score_by_edge: Mapping[EdgeKey, float],
    max_cluster_size: int,
    max_anchors: int = 4,
    candidates_per_anchor: int = 2,
) -> tuple[frozenset[EdgeKey], ...]:
    """Build small stars/closures around h_v-compatible role anchors.

    Many semantic facts are split into sibling hyperedges that share a value
    vertex but have no father link.  A shortest path or global semantic beam
    can miss the exact sibling set.  This route is linear in local incidence:
    for each strongest h_v anchor, rank incident edges by the frozen singleton
    score and emit bounded prefixes.  It remains context-local and label-free.
    """

    if max_cluster_size <= 0 or max_anchors <= 0 or candidates_per_anchor <= 0:
        raise ValueError("anchor-incidence budgets must be positive")
    stable = lambda value: (str(type(value)), str(value))
    incidence: dict[Hashable, list[EdgeKey]] = {}
    for edge in edge_ids:
        for vertex in vertices_by_edge.get(edge, set()):
            if vertex in anchor_vertices:
                incidence.setdefault(vertex, []).append(edge)
    ranked_anchors = sorted(
        incidence,
        key=lambda vertex: (
            -max((float(score_by_edge.get(edge, -1.0)) for edge in incidence[vertex]), default=-1.0),
            -len(incidence[vertex]),
            stable(vertex),
        ),
    )[:max_anchors]
    result: set[frozenset[EdgeKey]] = set()
    for vertex in ranked_anchors:
        ranked_edges = sorted(
            set(incidence[vertex]),
            key=lambda edge: (-float(score_by_edge.get(edge, -1.0)), stable(edge)),
        )[:max_cluster_size]
        # Retain the full bounded star and one shorter alternative; singleton
        # facts already have their own route and are not duplicated here.
        start = max(2, len(ranked_edges) - candidates_per_anchor + 1)
        for size in range(start, len(ranked_edges) + 1):
            result.add(frozenset(ranked_edges[:size]))
    return tuple(
        sorted(
            result,
            key=lambda values: (
                -max((float(score_by_edge.get(edge, -1.0)) for edge in values), default=-1.0),
                len(values),
                tuple(stable(value) for value in sorted(values, key=stable)),
            ),
        )
    )


def diverse_structural_shortlist(
    candidates: Iterable[EdgeKey],
    *,
    quota: int,
    rank_key,
    signatures: Mapping[EdgeKey, Iterable[Hashable]],
    diversity_slots: int = 1,
) -> tuple[EdgeKey, ...]:
    """Select ranked candidates while preserving route/coverage diversity.

    One representative of every observed signature is considered before the
    remaining slots are filled by the original rank.  The candidate universe
    and quota stay unchanged, so this improves coverage without hiding cost in
    a larger semantic-scoring budget.
    """

    if quota <= 0:
        return ()
    if diversity_slots < 0:
        raise ValueError("diversity_slots must be non-negative")
    ordered = tuple(sorted(dict.fromkeys(candidates), key=rank_key))
    if diversity_slots == 0 or not ordered:
        return ordered[:quota]
    buckets: dict[Hashable, list[EdgeKey]] = {}
    for candidate in ordered:
        values = tuple(signatures.get(candidate, ())) or ("unrouted",)
        for signature in values:
            buckets.setdefault(signature, []).append(candidate)
    # Keep the original score head; reserve only a small, explicit tail for a
    # rare route/coverage signature.  This avoids the recall regressions of a
    # fully diversity-first ordering.
    head_size = max(0, quota - min(diversity_slots, quota))
    chosen = list(ordered[:head_size])
    chosen_set = set(chosen)
    rare_candidates: list[EdgeKey] = []
    for _signature, values in sorted(
        buckets.items(), key=lambda item: (len(item[1]), str(item[0]))
    ):
        rare_candidates.extend(value for value in values if value not in chosen_set)
    rare_candidates = list(dict.fromkeys(rare_candidates))
    chosen.extend(rare_candidates[: quota - len(chosen)])
    chosen_set = set(chosen)
    if len(chosen) < quota:
        chosen.extend(value for value in ordered if value not in chosen_set)
    return tuple(chosen[:quota])


def father_atom_incidence_closures(
    edge_ids: Sequence[EdgeKey],
    *,
    father_by_edge: Mapping[EdgeKey, EdgeKey | None],
    vertices_by_edge: Mapping[EdgeKey, set[Hashable] | frozenset[Hashable]],
    score_by_edge: Mapping[EdgeKey, float],
    max_cluster_size: int,
    max_starts: int | None = None,
) -> tuple[frozenset[EdgeKey], ...]:
    """Build stable multi-fact HC proposals from semantic fact atoms.

    A parser may represent one semantic fact as a root hyperedge plus several
    father-linked complement/scope hyperedges.  Treating those fragments as
    unrelated singleton seeds makes a valid HC depend on an arbitrary BFS
    prefix and on the ordinary cluster-size bound ``m``.  This route first closes
    every father tree into one indivisible *fact atom*.  It then joins fact
    atoms only when they share a role/value vertex and emits every prefix of a
    greedy, score-ranked connected expansion.

    The operation is deliberately label-free: it consumes only frozen graph
    incidence, typed father links, and the already-computed singleton relation
    scores.  It is also context-local by construction; callers must pass the
    edges of exactly one source context.
    """

    if max_cluster_size <= 0:
        raise ValueError("max_cluster_size must be positive")
    if max_starts is not None and max_starts <= 0:
        raise ValueError("max_starts must be positive when provided")
    ordered = tuple(dict.fromkeys(edge_ids))
    if not ordered:
        return ()
    known = set(ordered)
    stable = lambda value: (str(type(value)), str(value))

    # Father links are expected to be a forest, but production parsers can be
    # imperfect.  A cycle is handled conservatively as one atom rather than
    # hanging candidate enumeration.
    def root(edge: EdgeKey) -> EdgeKey:
        current = edge
        path: list[EdgeKey] = []
        seen: set[EdgeKey] = set()
        while current not in seen:
            seen.add(current)
            path.append(current)
            father = father_by_edge.get(current)
            if father is None or father not in known:
                return current
            current = father
        cycle = path[path.index(current) :]
        return min(cycle, key=stable)

    atom_members: dict[EdgeKey, set[EdgeKey]] = {}
    for edge in ordered:
        atom_members.setdefault(root(edge), set()).add(edge)
    atoms = tuple(
        sorted(
            (frozenset(values) for values in atom_members.values()),
            key=lambda values: tuple(stable(value) for value in sorted(values, key=stable)),
        )
    )
    atom_vertices = tuple(
        frozenset().union(*(vertices_by_edge.get(edge, set()) for edge in atom))
        for atom in atoms
    )
    atom_score = tuple(
        max((float(score_by_edge.get(edge, -1.0)) for edge in atom), default=-1.0)
        for atom in atoms
    )
    adjacency = {index: set() for index in range(len(atoms))}
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            if atom_vertices[left] & atom_vertices[right]:
                adjacency[left].add(right)
                adjacency[right].add(left)

    starts = sorted(
        range(len(atoms)),
        key=lambda index: (
            -atom_score[index],
            len(atoms[index]),
            tuple(stable(value) for value in sorted(atoms[index], key=stable)),
        ),
    )
    if max_starts is not None:
        starts = starts[:max_starts]

    proposed: set[frozenset[EdgeKey]] = set()
    for start in starts:
        selected_atoms = {start}
        selected_edges = set(atoms[start])
        if len(selected_edges) <= max_cluster_size:
            proposed.add(frozenset(selected_edges))
        while True:
            frontier = set().union(*(adjacency[index] for index in selected_atoms))
            fitting = [
                index
                for index in frontier - selected_atoms
                if len(selected_edges | set(atoms[index])) <= max_cluster_size
            ]
            if not fitting:
                break
            next_atom = min(
                fitting,
                key=lambda index: (
                    -atom_score[index],
                    len(atoms[index]),
                    tuple(stable(value) for value in sorted(atoms[index], key=stable)),
                ),
            )
            selected_atoms.add(next_atom)
            selected_edges.update(atoms[next_atom])
            proposed.add(frozenset(selected_edges))

    return tuple(
        sorted(
            proposed,
            key=lambda values: (
                len(values),
                tuple(stable(value) for value in sorted(values, key=stable)),
            ),
        )
    )


__all__ = [
    "EdgeAlignmentScore",
    "assignment_cover_subsets",
    "connect_assignment",
    "anchor_incidence_subsets",
    "diverse_structural_shortlist",
    "edge_alignment_score",
    "father_atom_incidence_closures",
]
