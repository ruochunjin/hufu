"""Query Consistency, complete support cores, and document marking.

The implementation keeps three objects separate because they answer different
questions:

* query Consistency is left coverage of the maximum relation ``Pi``;
* a complete core is an inclusion-minimal set of surviving HC pairs whose
  query clusters cover every query hyperedge and whose D-matches cover every
  query comparison vertex;
* a document mark is derived in the strict order ``consistent``, ``conflict``,
  ``useless``, ``irrelevant``.

No additional HC-connectivity or cross-HC single-binding constraint is imposed
here.  Those constraints were useful in earlier GOLDEN analyses, but they are
not part of the ``FullCore`` predicate and would create false negatives in this
implementation.  Task helpers are loaded lazily from the central solver
module.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from .contracts import HCDependency, HCFailure, HyperSimulationResult, Pair

DocumentCategory = Literal["consistent", "conflict", "useless", "irrelevant"]


@dataclass(frozen=True, slots=True)
class QueryConsistencyResult:
    """Left-side query coverage over the maximum Hyper Simulation relation."""

    required_query_vertex_ids: tuple[str, ...]
    covered_query_vertex_ids: tuple[str, ...]
    uncovered_query_vertex_ids: tuple[str, ...]
    consistent: bool


@dataclass(frozen=True, slots=True)
class MatchedHCWitness:
    """One surviving, context-local matched-HC witness.

    The HC is accepted by the estimator, its D-match is non-empty and entirely
    contained in the final relation, and its data cluster belongs to one source
    context.  The latter is the implementation's provenance-preserving HC
    contract; a complete core may still combine witnesses from many contexts.
    """

    id: str
    query_edge_ids: tuple[str, ...]
    data_edge_ids: tuple[str, ...]
    context_id: str
    dmatch_pairs: tuple[Pair, ...]

    def validate(self) -> None:
        """Validate witness completeness and partial one-to-one D-match shape."""

        if not self.id:
            raise ValueError("a matched HC witness needs an id")
        if not self.query_edge_ids:
            raise ValueError(f"matched HC witness {self.id!r} covers no query edge")
        if not self.data_edge_ids:
            raise ValueError(f"matched HC witness {self.id!r} contains no data edge")
        if not self.context_id:
            raise ValueError(f"matched HC witness {self.id!r} has no source context")
        if not self.dmatch_pairs:
            raise ValueError(f"matched HC witness {self.id!r} has an empty D-match")
        left = [pair[0] for pair in self.dmatch_pairs]
        right = [pair[1] for pair in self.dmatch_pairs]
        if len(left) != len(set(left)) or len(right) != len(set(right)):
            raise ValueError(f"matched HC witness {self.id!r} is not partial one-to-one")


@dataclass(frozen=True, slots=True)
class MinimalCompleteCore:
    """One inclusion-minimal set covering query edges and query vertices."""

    witness_ids: tuple[str, ...]
    query_edge_ids: tuple[str, ...]
    query_vertex_ids: tuple[str, ...]
    data_edge_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    dmatch_pairs: tuple[Pair, ...]


@dataclass(frozen=True, slots=True)
class CompleteCoreResult:
    """All enumerated inclusion-minimal complete cores."""

    required_query_edge_ids: tuple[str, ...]
    required_query_vertex_ids: tuple[str, ...]
    minimal_cores: tuple[MinimalCompleteCore, ...]
    selected_core: MinimalCompleteCore | None
    full_core_context_ids: tuple[str, ...]
    uncovered_query_edge_ids: tuple[str, ...]
    uncovered_query_vertex_ids: tuple[str, ...]
    complete: bool
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class DocumentMark:
    """One query-specific category plus compact diagnostic evidence."""

    context_id: str
    category: DocumentCategory
    reason: str
    full_core: bool
    conflict_conditions: tuple[str, ...] = ()
    surviving_pairs: tuple[Pair, ...] = ()
    core_witness_ids: tuple[str, ...] = ()
    removed_pairs: tuple[Pair, ...] = ()

    @property
    def label(self) -> str:
        """Compatibility spelling used by task adapters."""

        return self.category


def compute_query_consistency(
    required_query_vertex_ids: Iterable[str],
    relation: Iterable[Pair],
) -> QueryConsistencyResult:
    """Evaluate ``forall u in H_Q.V, exists v: (u, v) in Pi``.

    In this compact schema, predicate vertices are represented by hyperedge
    heads and are deliberately outside D-match.  The caller therefore passes
    the semantic comparison-domain projection of ``H_Q.V``; complete query
    hyperedge coverage is checked separately by :func:`compute_complete_cores`.
    """

    required = _canonical(required_query_vertex_ids)
    if not required:
        raise ValueError("query Consistency needs query comparison vertices")
    relation_value = frozenset((str(left), str(right)) for left, right in relation)
    covered_set = {left for left, _ in relation_value} & set(required)
    covered = tuple(sorted(covered_set))
    uncovered = tuple(sorted(set(required) - covered_set))
    return QueryConsistencyResult(required, covered, uncovered, not uncovered)


def compute_complete_cores(
    required_query_edge_ids: Iterable[str],
    required_query_vertex_ids: Iterable[str],
    witnesses: Sequence[MatchedHCWitness],
    *,
    max_cores: int = 256,
) -> CompleteCoreResult:
    """Enumerate inclusion-minimal sets covering both required query domains.

    Query hyperedges and query vertices are both conjunctive requirements;
    alternative data HCs are existential witnesses.  Search stops as soon as a
    selection covers both domains, so every recorded solution is candidate-
    minimal.  A final set-inclusion pass removes any redundant solution that
    arose through a different enumeration order.
    """

    required_edges = _canonical(required_query_edge_ids)
    required_vertices = _canonical(required_query_vertex_ids)
    if not required_edges:
        raise ValueError("a complete core needs at least one query hyperedge")
    if not required_vertices:
        raise ValueError("a complete core needs query comparison vertices")
    if max_cores <= 0:
        raise ValueError("max_cores must be positive")

    ordered = tuple(sorted(witnesses, key=_witness_sort_key))
    witness_ids = [value.id for value in ordered]
    if len(witness_ids) != len(set(witness_ids)):
        raise ValueError("matched HC witnesses contain duplicate ids")

    required_edge_set = set(required_edges)
    required_vertex_set = set(required_vertices)
    for witness in ordered:
        witness.validate()
        unknown = set(witness.query_edge_ids) - required_edge_set
        if unknown:
            raise ValueError(
                f"matched HC witness {witness.id!r} covers unknown query edges: "
                f"{sorted(unknown)}"
            )

    by_edge = {
        edge_id: tuple(
            index
            for index, witness in enumerate(ordered)
            if edge_id in witness.query_edge_ids
        )
        for edge_id in required_edges
    }
    by_vertex = {
        vertex_id: tuple(
            index
            for index, witness in enumerate(ordered)
            if vertex_id in {left for left, _ in witness.dmatch_pairs}
        )
        for vertex_id in required_vertices
    }
    impossible_edges = tuple(sorted(key for key, values in by_edge.items() if not values))
    impossible_vertices = tuple(
        sorted(key for key, values in by_vertex.items() if not values)
    )
    if impossible_edges or impossible_vertices:
        return CompleteCoreResult(
            required_query_edge_ids=required_edges,
            required_query_vertex_ids=required_vertices,
            minimal_cores=(),
            selected_core=None,
            full_core_context_ids=(),
            uncovered_query_edge_ids=impossible_edges,
            uncovered_query_vertex_ids=impossible_vertices,
            complete=False,
        )

    raw_solutions: set[frozenset[int]] = set()
    truncated = False

    def search(
        selected: frozenset[int],
        covered_edges: frozenset[str],
        covered_vertices: frozenset[str],
    ) -> None:
        nonlocal truncated
        if any(solution <= selected for solution in raw_solutions):
            return
        if required_edge_set <= covered_edges and required_vertex_set <= covered_vertices:
            if len(raw_solutions) >= max_cores:
                truncated = True
                return
            raw_solutions.add(selected)
            return

        requirements = [
            ("edge", value, by_edge[value])
            for value in required_edge_set - covered_edges
        ] + [
            ("vertex", value, by_vertex[value])
            for value in required_vertex_set - covered_vertices
        ]
        _kind, _value, candidate_indexes = min(
            requirements,
            key=lambda value: (len(value[2]), value[0], value[1]),
        )
        for index in candidate_indexes:
            if index in selected:
                continue
            witness = ordered[index]
            search(
                selected | {index},
                covered_edges | frozenset(witness.query_edge_ids),
                covered_vertices
                | frozenset(left for left, _ in witness.dmatch_pairs),
            )

    search(frozenset(), frozenset(), frozenset())
    minimal_sets = tuple(
        value
        for value in sorted(raw_solutions, key=lambda item: (len(item), tuple(item)))
        if not any(other < value for other in raw_solutions)
    )
    cores = tuple(_materialize_core(value, ordered) for value in minimal_sets)
    selected = min(cores, key=_core_selection_key, default=None)
    contexts = tuple(
        sorted({context_id for core in cores for context_id in core.context_ids})
    )
    covered_edges = {value for core in cores for value in core.query_edge_ids}
    covered_vertices = {value for core in cores for value in core.query_vertex_ids}
    uncovered_edges = tuple(sorted(required_edge_set - covered_edges))
    uncovered_vertices = tuple(sorted(required_vertex_set - covered_vertices))
    return CompleteCoreResult(
        required_query_edge_ids=required_edges,
        required_query_vertex_ids=required_vertices,
        minimal_cores=cores,
        selected_core=selected,
        full_core_context_ids=contexts,
        uncovered_query_edge_ids=uncovered_edges,
        uncovered_query_vertex_ids=uncovered_vertices,
        complete=bool(cores) and not uncovered_edges and not uncovered_vertices,
        truncated=truncated,
    )


def mark_documents(
    context_ids: Iterable[str],
    *,
    context_vertex_ids: Mapping[str, Iterable[str]],
    fixed_point: HyperSimulationResult,
    complete_cores: CompleteCoreResult,
    dependencies: Sequence[HCDependency],
) -> tuple[DocumentMark, ...]:
    """Apply the fixed document-category precedence exactly.

    ``consistent`` is tested first, so a document that belongs to a minimal
    complete core remains consistent even if another HC from that document has
    a deletion record.  ``ConflictCert`` is HC-provenance based: endpoint
    provenance is never guessed for a fused vertex.
    """

    if fixed_point.mode != "all_hc":
        raise ValueError("document marking requires the all-HC fixed point")
    contexts = _canonical(context_ids)
    vertices = {
        str(context_id): {str(value) for value in values}
        for context_id, values in context_vertex_ids.items()
    }
    dependency_by_id = {value.id: value for value in dependencies}
    if len(dependency_by_id) != len(dependencies):
        raise ValueError("HC dependencies contain duplicate ids")

    full_core_contexts = set(complete_cores.full_core_context_ids)
    unknown_core_contexts = full_core_contexts - set(contexts)
    if unknown_core_contexts:
        raise ValueError(
            f"complete cores reference unknown contexts: {sorted(unknown_core_contexts)}"
        )

    core_witnesses: dict[str, set[str]] = {}
    for core in complete_cores.minimal_cores:
        for context_id in core.context_ids:
            core_witnesses.setdefault(context_id, set()).update(core.witness_ids)

    conflict_records: dict[str, list[tuple[Pair, HCFailure]]] = {}
    for removal in fixed_point.removals:
        if removal.pair not in fixed_point.initial_relation:
            raise ValueError("a deletion record refers to a pair outside the initial relation")
        for failure in removal.failures:
            if failure.condition == "diagnostic":
                continue
            dependency = dependency_by_id.get(failure.hc_id)
            if dependency is None:
                raise ValueError(
                    f"deletion record references unknown HC {failure.hc_id!r}"
                )
            dmatch = dependency.dmatch.effective_pairs
            # An empty D-match is an explicit anchor-membership certificate.
            anchor_membership_failed = (
                failure.condition == "anchor_membership" and not dmatch
            )
            dependency_closure_failed = (
                failure.condition == "dependency_closure"
                and bool(set(failure.missing_pairs) & (dmatch - fixed_point.relation))
            )
            if anchor_membership_failed or dependency_closure_failed:
                conflict_records.setdefault(
                    dependency.data_cluster.context_id, []
                ).append((removal.pair, failure))

    result: list[DocumentMark] = []
    for context_id in contexts:
        local_vertices = vertices.get(context_id, set())
        pi_i = tuple(
            sorted(
                pair
                for pair in fixed_point.relation
                if pair[1] in local_vertices
            )
        )
        records = conflict_records.get(context_id, [])

        if context_id in full_core_contexts:
            result.append(
                DocumentMark(
                    context_id=context_id,
                    category="consistent",
                    reason=(
                        "This context contributes relations and entity matches "
                        "to a complete query evidence core."
                    ),
                    full_core=True,
                    surviving_pairs=pi_i,
                    core_witness_ids=tuple(sorted(core_witnesses[context_id])),
                    removed_pairs=tuple(sorted({pair for pair, _ in records})),
                )
            )
        elif records:
            conditions = tuple(sorted({failure.condition for _, failure in records}))
            result.append(
                DocumentMark(
                    context_id=context_id,
                    category="conflict",
                    reason=(
                        "A relation in this context was selected for comparison, "
                        "but a required entity match failed."
                    ),
                    full_core=False,
                    conflict_conditions=conditions,
                    surviving_pairs=pi_i,
                    removed_pairs=tuple(sorted({pair for pair, _ in records})),
                )
            )
        elif pi_i:
            result.append(
                DocumentMark(
                    context_id=context_id,
                    category="useless",
                    reason=(
                        "This context has query-related entity matches, but it "
                        "does not participate in a complete evidence core."
                    ),
                    full_core=False,
                    surviving_pairs=pi_i,
                )
            )
        else:
            result.append(
                DocumentMark(
                    context_id=context_id,
                    category="irrelevant",
                    reason="No query entity remains matched to this context.",
                    full_core=False,
                )
            )
    return tuple(result)


def _canonical(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values}))


def _materialize_core(
    selected: frozenset[int], witnesses: Sequence[MatchedHCWitness]
) -> MinimalCompleteCore:
    values = tuple(witnesses[index] for index in sorted(selected))
    dmatch_pairs = tuple(
        sorted({pair for witness in values for pair in witness.dmatch_pairs})
    )
    return MinimalCompleteCore(
        witness_ids=tuple(sorted(value.id for value in values)),
        query_edge_ids=tuple(
            sorted({edge_id for value in values for edge_id in value.query_edge_ids})
        ),
        query_vertex_ids=tuple(sorted({left for left, _ in dmatch_pairs})),
        data_edge_ids=tuple(
            sorted({edge_id for value in values for edge_id in value.data_edge_ids})
        ),
        context_ids=tuple(sorted({value.context_id for value in values})),
        dmatch_pairs=dmatch_pairs,
    )


def _witness_sort_key(value: MatchedHCWitness) -> tuple[object, ...]:
    return (
        value.query_edge_ids,
        value.context_id,
        value.data_edge_ids,
        value.dmatch_pairs,
        value.id,
    )


def _core_selection_key(value: MinimalCompleteCore) -> tuple[object, ...]:
    """Choose one deterministic core while retaining every minimal alternative."""

    return (
        len(value.data_edge_ids),
        len(value.context_ids),
        len(value.witness_ids),
        value.context_ids,
        value.data_edge_ids,
        value.witness_ids,
    )


def _solver_module():
    """Return the canonical module containing both solver implementations."""

    return importlib.import_module("hufu.engine.hyper_simulation")


def consistent_detection(*args, **kwargs):
    """Compatibility entry point for binary Consistency detection."""

    return _solver_module().get_standard_symbol("consistent_detection")(*args, **kwargs)


def load_hypergraphs_for_instance(*args, **kwargs):
    """Compatibility loader for query and context hypergraphs."""

    return _solver_module().get_standard_symbol("load_hypergraphs_for_instance")(
        *args, **kwargs
    )


def query_fixup(*args, **kwargs):
    """Compatibility entry point for the RAG task adapter."""

    return _solver_module().get_standard_symbol("query_fixup")(*args, **kwargs)


def __getattr__(name: str):
    """Lazily expose remaining task-adapter helpers."""

    if name not in _STANDARD_SYMBOLS:
        raise AttributeError(name)
    try:
        return _solver_module().get_standard_symbol(name)
    except AttributeError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error


_STANDARD_SYMBOLS = frozenset(
    {
        "consistent_detection",
        "generate_instance_id",
        "get_distance",
        "is_critical_vertex",
        "load_hypergraphs_for_instance",
        "query_fixup",
    }
)


__all__ = [
    "CompleteCoreResult",
    "DocumentCategory",
    "DocumentMark",
    "MatchedHCWitness",
    "MinimalCompleteCore",
    "QueryConsistencyResult",
    "compute_complete_cores",
    "compute_query_consistency",
    "consistent_detection",
    "load_hypergraphs_for_instance",
    "mark_documents",
    "query_fixup",
]
