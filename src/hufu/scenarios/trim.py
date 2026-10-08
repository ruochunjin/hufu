"""Scenario 2: conflict-aware trimming of retrieval evidence and index edges.

Two levels of conflict resolution, both driven by rule-confirmed conflict
pairs (from ``detect_memory_conflicts`` or ``check_retrieval``):

1. **Evidence trimming** (dependency-free): choose which side of each
   conflicting pair to drop before the evidence reaches the generator.
   Handles conflict *chains* (A↔B, B↔C) by voting, not by pairwise deletion,
   so transitive conflicts cannot silently empty the evidence set.

2. **HNSW edge trimming** (optional, needs faiss): cut the graph edges
   between conflicting vectors so future nearest-neighbor searches stop
   surfacing the conflict — the index-level repair from the CDR work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class TrimDecision:
    """Why one item was dropped."""

    index: int
    lost_to: tuple[int, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class TrimResult:
    """Outcome of evidence trimming."""

    kept: tuple[int, ...]
    dropped: tuple[TrimDecision, ...]

    def apply(self, items: Sequence[Any]) -> list[Any]:
        """Materialize the kept items, in input order."""

        return [items[i] for i in self.kept]


def _attr(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def trim_evidence(
    items: Sequence[Any],
    conflicts: Sequence[tuple[int, int]],
    *,
    strategy: str = "keep_recent",
    items_attrs: Sequence[Mapping[str, Any]] | None = None,
) -> TrimResult:
    """Resolve conflicting pairs by dropping one side of each.

    ``items`` is the retrieval/memory list in input order; ``conflicts``
    contains ``(i, j)`` index pairs.  Strategies:

    - ``keep_recent``: prefer the item with the larger ``timestamp``
      attribute (falls back to input order: later wins);
    - ``keep_first`` / ``keep_last``: positional, no attributes needed;
    - ``keep_priority``: prefer the larger numeric ``priority`` attribute.

    Greedy minimum-cover semantics: repeatedly drop the item involved in
    the most *unresolved* conflicts (chains like A↔B↔C drop only B, never
    everything); ties are broken against the strategy's preference.  A
    conflict stops counting once one side is dropped, so survivors never
    conflict with each other.  Note this is an *approximation*: greedy
    maximum-degree cover does not guarantee the globally smallest deletion
    set on adversarial graphs.
    """

    if strategy not in {"keep_recent", "keep_first", "keep_last", "keep_priority"}:
        raise ValueError(f"unknown strategy: {strategy!r}")
    n = len(items)
    for i, j in conflicts:
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(f"conflict pair {(i, j)} out of range for {n} items")

    def key(idx: int) -> tuple:
        attrs = items_attrs[idx] if items_attrs is not None else {}
        value = _attr(attrs, "timestamp") if strategy == "keep_recent" else None
        if strategy == "keep_priority":
            value = _attr(attrs, "priority")
        if strategy == "keep_first":
            return (0, -idx)  # earlier wins
        if strategy == "keep_last":
            return (0, idx)  # later wins
        if value is None:
            return (0, idx)  # undated: later wins
        return (1, value)  # dated: newer/higher wins

    neighbors: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in conflicts:
        neighbors[i].add(j)
        neighbors[j].add(i)

    dropped: set[int] = set()
    drop_order: list[int] = []
    while True:
        active = {
            idx: {j for j in adj if j not in dropped}
            for idx, adj in neighbors.items()
            if idx not in dropped
        }
        degree = {idx: len(adj) for idx, adj in active.items()}
        max_degree = max(degree.values(), default=0)
        if max_degree == 0:
            break
        # Among max-degree items, drop the strategy-least-preferred.
        candidate = min(
            (idx for idx in active if degree[idx] == max_degree),
            key=lambda idx: key(idx),
        )
        dropped.add(candidate)
        drop_order.append(candidate)

    kept = tuple(i for i in range(n) if i not in dropped)
    decisions = tuple(
        TrimDecision(
            index=i,
            lost_to=tuple(sorted(neighbors[i])),
            reason=f"involved in {len(neighbors[i])} conflict(s); "
            f"dropped to resolve them ({strategy})",
        )
        for i in drop_order
    )
    return TrimResult(kept=kept, dropped=decisions)


def trim_hnsw_edges(
    index: Any,
    conflicts: Sequence[tuple[int, int]],
) -> int:
    """Remove HNSW neighbor links between conflicting vectors, in place.

    Follows the CDR index-trimming idea: after trimming, approximate
    nearest-neighbor search no longer travels between conflicting memories.
    ``conflicts`` uses the index's internal vector ids (``faiss`` ids, not
    positions in your document list — map them beforehand).

    Requires a ``faiss.IndexHNSW*`` instance.  Returns the number of links
    removed.  Experimental: relies on faiss's neighbor-array layout.
    """

    import faiss  # lazy: optional dependency

    hnsw = index.hnsw
    neighbors = faiss.vector_to_array(hnsw.neighbors)
    offsets = faiss.vector_to_array(hnsw.offsets)
    if len(offsets) < 2:
        return 0
    removed = 0
    conflict_set = {frozenset(pair) for pair in conflicts}
    for i, j in conflict_set:
        for src, dst in ((i, j), (j, i)):
            if src + 1 >= len(offsets):
                continue
            # faiss stores each node's links in the block
            # [offsets[src], offsets[src + 1]) — level 0 first, then any
            # upper levels.  Scan the whole block and cut every link to dst.
            begin, end = int(offsets[src]), int(offsets[src + 1])
            segment = neighbors[begin:end]
            for pos in range(len(segment)):
                if segment[pos] == dst:
                    segment[pos] = -1
                    removed += 1
    return removed


def infer_conflicts_from_judgments(
    judgments: Mapping[int, bool],
) -> list[tuple[int, int]]:
    """Infer inter-item conflicts from per-item consistency judgments.

    The CDR *Trimming Theorem*: when the same witness (query) judges item
    s1 consistent and item s2 conflicting, s1 and s2 contradict each other.
    Conflict pairs are therefore the Cartesian product of the two judgment
    classes — this derives item↔item conflicts from item↔query verdicts
    without a dedicated pairwise detector (CDRs `context_relation`).

    ``judgments`` maps item index → True (consistent with the witness) or
    False (conflicting).  Returns sorted ``(i, j)`` pairs, ``i < j``.
    """

    consistent = sorted(index for index, ok in judgments.items() if ok)
    dissenting = sorted(index for index, ok in judgments.items() if not ok)
    return [(i, j) for i in consistent for j in dissenting]


__all__ = [
    "TrimDecision",
    "TrimResult",
    "infer_conflicts_from_judgments",
    "trim_evidence",
    "trim_hnsw_edges",
]
