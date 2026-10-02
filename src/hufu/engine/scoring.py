"""Model-agnostic HC rendering and batched embedding scoring.

The release intentionally ships no neural weights.  Any encoder implementing
``EmbeddingBackend`` can be injected; the production system uses a prompted
Qwen embedding model.  Entity surface values are never exposed to HC.  They
are replaced by cluster-local type slots, while predicates and linguistic
modifiers retain the lexical information needed to compare relations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

from .contracts import LEXICAL_KINDS, Cluster, HCCandidate, HCScore, Hypergraph

HC_EMBEDDING_PROMPT = (
    "Represent the masked n-ary fact pattern for retrieving facts that express "
    "the same relation or a directly contradiction-comparable relation. "
    "Respect semantic roles, voice, polarity, modality, time, and quantity."
)


class EmbeddingBackend(Protocol):
    """Minimal adapter implemented by a local or remote text encoder."""

    def encode(
        self, texts: Sequence[str], *, prompt: str | None = None
    ) -> Sequence[Sequence[float]]:
        """Encode a batch of texts, optionally under a model prompt."""

        ...


@dataclass(slots=True)
class CallableEmbeddingBackend:
    """Wrap a plain callable without importing a model framework."""

    function: Callable[[Sequence[str], str | None], Sequence[Sequence[float]]]

    def encode(
        self, texts: Sequence[str], *, prompt: str | None = None
    ) -> Sequence[Sequence[float]]:
        """Delegate batch encoding to the wrapped callable."""

        return self.function(texts, prompt)


class CachedEmbeddingBackend:
    """Reuse rendered-text vectors across singleton and full-cluster scoring."""

    def __init__(self, backend: EmbeddingBackend) -> None:
        self.backend = backend
        self._cache: dict[tuple[str | None, str], tuple[float, ...]] = {}

    def encode(
        self, texts: Sequence[str], *, prompt: str | None = None
    ) -> Sequence[Sequence[float]]:
        """Return cached vectors and encode only previously unseen texts."""

        missing = tuple(
            dict.fromkeys(
                text for text in texts if (prompt, text) not in self._cache
            )
        )
        if missing:
            vectors = self.backend.encode(missing, prompt=prompt)
            if len(vectors) != len(missing):
                raise ValueError("embedding backend returned the wrong batch length")
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[(prompt, text)] = _validated_vector(vector)
        return tuple(self._cache[(prompt, text)] for text in texts)


TYPE_GROUPS: Mapping[str, str] = {
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "DATE": "TEMPORAL",
    "TIME": "TEMPORAL",
    "TEMPORAL": "TEMPORAL",
    "PERSON": "AGENT",
    "NORP": "AGENT",
    "ORG": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "MONEY": "QUANTITY",
    "PERCENT": "QUANTITY",
    "QUANTITY": "QUANTITY",
    "CARDINAL": "QUANTITY",
    "ORDINAL": "QUANTITY",
}


def render_cluster(graph: Hypergraph, cluster: Cluster, *, view: str) -> str:
    """Render one HC with fine, grouped, or role-structural entity masks."""

    if view not in {"fine", "group", "role"}:
        raise ValueError(f"unknown HC view: {view!r}")
    cluster.validate(graph)
    slots = _cluster_slots(graph, cluster, view=view)
    local_edge_index = {edge_id: index for index, edge_id in enumerate(cluster.edge_ids)}
    lines: list[str] = []
    for edge_id in cluster.edge_ids:
        edge = graph.edge(edge_id)
        predicate = graph.vertex(edge.predicate_id).text.strip().lower()
        frame = edge.canonical_frame.strip().lower() or predicate
        scope = [
            f"frame={frame}",
            f"predicate={predicate}",
            f"voice={edge.voice.lower()}",
            f"polarity={edge.polarity.lower()}",
            f"modality={edge.modality.lower()}",
        ]
        if edge.time:
            scope.append("time=<TEMPORAL>")
        if edge.quantity:
            scope.append("quantity=<QUANTITY>")
        if edge.father_id in local_edge_index:
            scope.append(f"father=e{local_edge_index[edge.father_id]}")
        role_values = []
        for role in sorted(edge.roles, key=lambda value: (value.name, value.vertex_id)):
            vertex = graph.vertex(role.vertex_id)
            value = (
                vertex.text.strip().lower()
                if vertex.kind in LEXICAL_KINDS
                else slots[vertex.id]
            )
            role_values.append(f"{role.name.lower()}={value}")
        lines.append(
            f"e{local_edge_index[edge_id]}[{' ; '.join(scope)}] "
            + " | ".join(role_values)
        )
    return "\n".join(lines)


def score_singleton_relations(
    query: Hypergraph,
    data: Hypergraph,
    backend: EmbeddingBackend,
    *,
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
    calibrate: Callable[[float], float] | None = None,
) -> dict[tuple[str, str], float]:
    """Encode every singleton once and return the HCCalc seed matrix."""

    pairs: list[tuple[Cluster, Cluster]] = []
    keys: list[tuple[str, str]] = []
    for query_edge in sorted(query.hyperedges, key=lambda value: value.id):
        query_cluster = Cluster.from_edges(
            query, (query_edge.id,), context_id=query_edge.context_id
        )
        for data_edge in sorted(data.hyperedges, key=lambda value: value.id):
            data_cluster = Cluster.from_edges(
                data, (data_edge.id,), context_id=data_edge.context_id
            )
            pairs.append((query_cluster, data_cluster))
            keys.append((query_edge.id, data_edge.id))
    probabilities = _score_cluster_pairs(
        query, data, pairs, backend, weights=weights, calibrate=calibrate
    )
    return dict(zip(keys, probabilities, strict=True))


def score_hc_candidates(
    query: Hypergraph,
    data: Hypergraph,
    candidates: Sequence[HCCandidate],
    backend: EmbeddingBackend,
    *,
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
    calibrate: Callable[[float], float] | None = None,
) -> dict[str, HCScore]:
    """Score all threshold-free candidates in one deduplicated encoder batch."""

    pairs = tuple(
        (candidate.query_cluster, candidate.data_cluster) for candidate in candidates
    )
    details = _score_cluster_pairs_with_views(
        query, data, pairs, backend, weights=weights, calibrate=calibrate
    )
    return {
        candidate.id: HCScore(probability=probability, view_scores=view_scores)
        for candidate, (probability, view_scores) in zip(candidates, details, strict=True)
    }


def _score_cluster_pairs(
    query: Hypergraph,
    data: Hypergraph,
    pairs: Sequence[tuple[Cluster, Cluster]],
    backend: EmbeddingBackend,
    *,
    weights: tuple[float, float, float],
    calibrate: Callable[[float], float] | None,
) -> tuple[float, ...]:
    return tuple(
        probability
        for probability, _ in _score_cluster_pairs_with_views(
            query, data, pairs, backend, weights=weights, calibrate=calibrate
        )
    )


def _score_cluster_pairs_with_views(
    query: Hypergraph,
    data: Hypergraph,
    pairs: Sequence[tuple[Cluster, Cluster]],
    backend: EmbeddingBackend,
    *,
    weights: tuple[float, float, float],
    calibrate: Callable[[float], float] | None,
) -> tuple[tuple[float, tuple[tuple[str, float], ...]], ...]:
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("HC view weights must sum to one")
    views = ("fine", "group", "role")
    rendered: list[tuple[str, ...]] = []
    for query_cluster, data_cluster in pairs:
        rendered.append(
            tuple(
                value
                for view in views
                for value in (
                    render_cluster(query, query_cluster, view=view),
                    render_cluster(data, data_cluster, view=view),
                )
            )
        )

    unique_texts = tuple(dict.fromkeys(text for row in rendered for text in row))
    if not unique_texts:
        return ()
    vectors = backend.encode(unique_texts, prompt=HC_EMBEDDING_PROMPT)
    if len(vectors) != len(unique_texts):
        raise ValueError("embedding backend returned the wrong batch length")
    cache = {
        text: _validated_vector(vector) for text, vector in zip(unique_texts, vectors, strict=True)
    }
    result = []
    for row in rendered:
        view_scores = []
        for index, view in enumerate(views):
            cosine = _cosine(cache[row[index * 2]], cache[row[index * 2 + 1]])
            view_scores.append((view, (cosine + 1.0) / 2.0))
        fused = sum(weight * value for weight, (_, value) in zip(weights, view_scores))
        probability = calibrate(fused) if calibrate is not None else fused
        probability = max(0.0, min(1.0, float(probability)))
        result.append((probability, tuple(view_scores)))
    return tuple(result)


def _cluster_slots(graph: Hypergraph, cluster: Cluster, *, view: str) -> dict[str, str]:
    counters: dict[str, int] = {}
    result: dict[str, str] = {}
    for edge_id in cluster.edge_ids:
        edge = graph.edge(edge_id)
        for role in sorted(edge.roles, key=lambda value: (value.name, value.vertex_id)):
            vertex = graph.vertex(role.vertex_id)
            if vertex.kind in LEXICAL_KINDS or vertex.id in result:
                continue
            semantic_type = vertex.feature_type
            label = TYPE_GROUPS.get(semantic_type, semantic_type) if view != "fine" else semantic_type
            if view == "role":
                # The role view emphasizes structural correspondence without
                # exposing entity values; the role name already appears next
                # to this generic type family in the rendered edge.
                label = TYPE_GROUPS.get(semantic_type, "ENTITY")
            counters[label] = counters.get(label, 0) + 1
            result[vertex.id] = f"{label}#{counters[label]}"
    return result


def _validated_vector(value: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(item) for item in value)
    if not vector or not all(math.isfinite(item) for item in vector):
        raise ValueError("embedding backend returned an invalid vector")
    return vector


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    value = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return max(-1.0, min(1.0, value))


__all__ = [
    "CachedEmbeddingBackend",
    "CallableEmbeddingBackend",
    "EmbeddingBackend",
    "HC_EMBEDDING_PROMPT",
    "TYPE_GROUPS",
    "render_cluster",
    "score_hc_candidates",
    "score_singleton_relations",
]
