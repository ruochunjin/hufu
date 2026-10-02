"""D-match scoring, constrained decoding, and compatibility entry points.

D-match is a relation between role vertices, not between surface positions.
Consequently active/passive realizations and converse predicates (for example,
``sell`` versus ``buy``) may align different surface roles when their canonical
frames describe the same participant.  A scorer supplies those semantic cell
scores; this module enforces the mathematical constraints around the scorer.
The embedding/path estimator is exposed through lazy compatibility functions,
so its optional model stack is not an import-time dependency of this module.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from .contracts import (
    LEXICAL_KINDS,
    Cluster,
    DMatchDecision,
    HCCandidate,
    Hypergraph,
    Pair,
    Vertex,
    iter_cluster_roles,
    validate_pairs_exist,
)
from .scoring import TYPE_GROUPS, EmbeddingBackend

DMATCH_EMBEDDING_PROMPT = (
    "Represent the marked participant role in this n-ary fact. Match roles by "
    "their real semantics, including active/passive voice and converse frames; "
    "do not match them merely because they have the same surface position."
)


@dataclass(frozen=True, slots=True)
class RoleCell:
    """One candidate role pair rendered in masked and target-surface views."""

    pair: Pair
    query_roles: tuple[str, ...]
    data_roles: tuple[str, ...]
    query_masked_text: str
    data_masked_text: str
    query_target_text: str
    data_target_text: str


@dataclass(frozen=True, slots=True)
class RoleScoringResult:
    """Scorer probabilities plus an optional whole-relation conflict signal."""

    scores: Mapping[Pair, float]
    relation_conflict: bool = False

    def __post_init__(self) -> None:
        for pair, score in self.scores.items():
            if len(pair) != 2 or not all(str(value) for value in pair):
                raise ValueError("role scores must use non-empty node pairs")
            value = float(score)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("role scores must be finite probabilities")


class RoleCellScorer(Protocol):
    """Structural interface for an injected semantic-role scorer."""

    def score(self, cells: Sequence[RoleCell]) -> RoleScoringResult:
        """Score a batch of candidate role cells."""

        ...


@dataclass(slots=True)
class CallableRoleCellScorer:
    """Adapt a plain callable to the role-cell scorer protocol."""

    function: Callable[[Sequence[RoleCell]], RoleScoringResult]

    def score(self, cells: Sequence[RoleCell]) -> RoleScoringResult:
        """Delegate a cell batch to the wrapped callable."""

        return self.function(cells)


def default_type_compatible(query_vertex: Vertex, data_vertex: Vertex) -> bool:
    """Conservative hard gate shared with h_v candidate generation."""

    left = query_vertex.feature_type
    right = data_vertex.feature_type
    if left == right:
        return True
    return TYPE_GROUPS.get(left, left) == TYPE_GROUPS.get(right, right)


def build_role_cells(
    query: Hypergraph,
    data: Hypergraph,
    candidate: HCCandidate,
    *,
    hv_allowed_pairs: Iterable[Pair],
    type_compatible: Callable[[Vertex, Vertex], bool] = default_type_compatible,
) -> tuple[RoleCell, ...]:
    """Build only h_v-allowed, type-compatible D-match cells."""

    hv_allowed = validate_pairs_exist(hv_allowed_pairs, query, data)
    query_roles = _roles_by_vertex(query, candidate.query_cluster)
    data_roles = _roles_by_vertex(data, candidate.data_cluster)
    result: list[RoleCell] = []
    for query_id in sorted(query_roles):
        query_vertex = query.vertex(query_id)
        if not query_vertex.matchable:
            continue
        for data_id in sorted(data_roles):
            data_vertex = data.vertex(data_id)
            pair = (query_id, data_id)
            if (
                not data_vertex.matchable
                or pair not in hv_allowed
                or not type_compatible(query_vertex, data_vertex)
            ):
                continue
            result.append(
                RoleCell(
                    pair=pair,
                    query_roles=query_roles[query_id],
                    data_roles=data_roles[data_id],
                    query_masked_text=_render_role_side(
                        query,
                        candidate.query_cluster,
                        query_id,
                        query_roles[query_id],
                        include_target_lexical=False,
                    ),
                    data_masked_text=_render_role_side(
                        data,
                        candidate.data_cluster,
                        data_id,
                        data_roles[data_id],
                        include_target_lexical=False,
                    ),
                    query_target_text=_render_role_side(
                        query,
                        candidate.query_cluster,
                        query_id,
                        query_roles[query_id],
                        include_target_lexical=True,
                    ),
                    data_target_text=_render_role_side(
                        data,
                        candidate.data_cluster,
                        data_id,
                        data_roles[data_id],
                        include_target_lexical=True,
                    ),
                )
            )
    return tuple(result)


def compute_dmatch(
    query: Hypergraph,
    data: Hypergraph,
    candidate: HCCandidate,
    scorer: RoleCellScorer,
    *,
    hv_allowed_pairs: Iterable[Pair],
    anchor_pairs: Iterable[Pair],
    threshold: float,
    type_compatible: Callable[[Vertex, Vertex], bool] = default_type_compatible,
) -> DMatchDecision:
    """Score and decode one HC, abstaining on inference failure.

    A *valid* semantic conflict is represented by an empty effective D-match.
    A scorer exception is different evidence: it says nothing about semantics.
    It therefore returns an explicit ``inference_error_abstain`` state.  The
    pipeline omits that HC dependency, so dependency closure retains the current
    node pair without inventing role matches from the h_v cross-product.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("D-match threshold must be in [0, 1]")
    anchors = validate_pairs_exist(anchor_pairs, query, data)
    hv_allowed = validate_pairs_exist(hv_allowed_pairs, query, data)
    if not anchors <= hv_allowed:
        raise ValueError("D-match anchors must already pass h_v")
    for left, right in anchors:
        if not type_compatible(query.vertex(left), data.vertex(right)):
            raise ValueError("D-match anchor failed the hard type gate")

    cells = build_role_cells(
        query,
        data,
        candidate,
        hv_allowed_pairs=hv_allowed,
        type_compatible=type_compatible,
    )
    try:
        scored = scorer.score(cells)
    except Exception:  # model/network/OOM errors are not semantic contradictions
        return DMatchDecision(
            pairs=frozenset(),
            relation_conflict=False,
            status="inference_error_abstain",
        )
    if scored.relation_conflict:
        return DMatchDecision(
            pairs=frozenset(), relation_conflict=True, status="relation_conflict"
        )
    legal_pairs = {cell.pair for cell in cells}
    unknown = set(scored.scores) - legal_pairs
    if unknown:
        raise ValueError(f"role scorer returned illegal cells: {sorted(unknown)}")
    pairs = stable_partial_one_to_one(scored.scores, threshold=threshold)
    return DMatchDecision(
        pairs=pairs,
        status="ok" if cells else "no_compatible_roles",
        pair_scores=tuple(
            sorted((pair, float(scored.scores[pair])) for pair in pairs)
        ),
    )


def stable_partial_one_to_one(
    scores: Mapping[Pair, float], *, threshold: float
) -> frozenset[Pair]:
    """Greedily decode a reproducible partial one-to-one relation."""

    normalized: list[tuple[Pair, float]] = []
    for pair, raw_score in scores.items():
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("D-match scores must be finite probabilities")
        normalized.append(((str(pair[0]), str(pair[1])), score))

    selected: set[Pair] = set()
    used_left: set[str] = set()
    used_right: set[str] = set()
    for (left, right), score in sorted(
        normalized,
        key=lambda value: (-value[1], value[0][0], value[0][1]),
    ):
        if score < threshold or left in used_left or right in used_right:
            continue
        selected.add((left, right))
        used_left.add(left)
        used_right.add(right)
    return frozenset(selected)


class EmbeddingRoleScorer:
    """Two-view embedding scorer; model loading remains outside the core."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        masked_weight: float = 0.5,
        target_weight: float = 0.5,
        conflict_detector: Callable[[Sequence[RoleCell]], bool] | None = None,
    ) -> None:
        if abs(masked_weight + target_weight - 1.0) > 1e-9:
            raise ValueError("D-match view weights must sum to one")
        self.backend = backend
        self.masked_weight = masked_weight
        self.target_weight = target_weight
        self.conflict_detector = conflict_detector

    def score(self, cells: Sequence[RoleCell]) -> RoleScoringResult:
        """Score role cells from masked and target-surface embedding views."""

        if not cells:
            return RoleScoringResult({})
        texts = tuple(
            dict.fromkeys(
                text
                for cell in cells
                for text in (
                    cell.query_masked_text,
                    cell.data_masked_text,
                    cell.query_target_text,
                    cell.data_target_text,
                )
            )
        )
        vectors = self.backend.encode(texts, prompt=DMATCH_EMBEDDING_PROMPT)
        if len(vectors) != len(texts):
            raise ValueError("embedding backend returned the wrong batch length")
        cache = {text: tuple(float(value) for value in vector) for text, vector in zip(texts, vectors, strict=True)}
        scores = {}
        for cell in cells:
            masked = (_cosine(
                cache[cell.query_masked_text], cache[cell.data_masked_text]
            ) + 1.0) / 2.0
            target = (_cosine(
                cache[cell.query_target_text], cache[cell.data_target_text]
            ) + 1.0) / 2.0
            scores[cell.pair] = self.masked_weight * masked + self.target_weight * target
        conflict = bool(self.conflict_detector and self.conflict_detector(cells))
        return RoleScoringResult(scores, relation_conflict=conflict)


def _roles_by_vertex(graph: Hypergraph, cluster: Cluster) -> dict[str, tuple[str, ...]]:
    mutable: dict[str, set[str]] = {}
    for edge_id, role in iter_cluster_roles(graph, cluster):
        vertex = graph.vertex(role.vertex_id)
        if vertex.matchable:
            edge = graph.edge(edge_id)
            frame = edge.canonical_frame or graph.vertex(edge.predicate_id).text
            mutable.setdefault(vertex.id, set()).add(f"{frame}:{role.name}")
    return {key: tuple(sorted(values)) for key, values in mutable.items()}


def _render_role_side(
    graph: Hypergraph,
    cluster: Cluster,
    vertex_id: str,
    roles: tuple[str, ...],
    *,
    include_target_lexical: bool,
) -> str:
    vertex = graph.vertex(vertex_id)
    return (
        f"frames={_cluster_frames(graph, cluster)}; "
        f"target={_target_value(vertex, include_target_lexical)}; "
        f"roles={','.join(roles)}"
    )


def _target_value(vertex: Vertex, include_lexical: bool) -> str:
    if include_lexical and vertex.kind in LEXICAL_KINDS:
        return vertex.text.strip().lower()
    return f"<{vertex.feature_type}>"


def _cluster_frames(graph: Hypergraph, cluster: Cluster) -> str:
    values = []
    for edge_id in cluster.edge_ids:
        edge = graph.edge(edge_id)
        predicate = graph.vertex(edge.predicate_id).text.strip().lower()
        values.append(
            f"{edge.canonical_frame or predicate}/{edge.voice}/{edge.polarity}/{edge.modality}"
        )
    return " | ".join(values)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("role embedding dimensions do not match")
    if not all(math.isfinite(float(value)) for value in (*left, *right)):
        raise ValueError("role embedding contains a non-finite value")
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    score = sum(float(a) * float(b) for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return max(-1.0, min(1.0, score))


def _solver_module():
    """Return the canonical module containing both solver implementations."""

    return importlib.import_module("hufu.engine.hyper_simulation")


def calc_d_match(*args, **kwargs):
    """Compatibility entry point for the embedding/path D-match estimator."""

    return _solver_module().get_standard_symbol("calc_d_match")(*args, **kwargs)


def calc_d_match_batch(*args, **kwargs):
    """Compatibility entry point for batched embedding/path estimation."""

    return _solver_module().get_standard_symbol("calc_d_match_batch")(*args, **kwargs)


def __getattr__(name: str):
    """Lazily expose remaining helpers used by compatibility scripts."""

    if name not in _STANDARD_SYMBOLS:
        raise AttributeError(name)
    try:
        return _solver_module().get_standard_symbol(name)
    except AttributeError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error


_STANDARD_SYMBOLS = frozenset(
    {
        "_construct_description_from_path",
        "calc_d_match",
        "calc_d_match_batch",
        "query_same_type",
    }
)


__all__ = [
    "CallableRoleCellScorer",
    "DMATCH_EMBEDDING_PROMPT",
    "EmbeddingRoleScorer",
    "RoleCell",
    "RoleCellScorer",
    "RoleScoringResult",
    "build_role_cells",
    "calc_d_match",
    "calc_d_match_batch",
    "compute_dmatch",
    "default_type_compatible",
    "stable_partial_one_to_one",
]
