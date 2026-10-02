"""Reference implementation of the Hyper Simulation greatest fixed point.

The optimized Rust backend implements the same monotone deletion process with
reverse dependency indices.  This compact Python implementation is intentionally
direct: it keeps the two HC failure conditions, parallel iterations, and
deletion certificates explicit and traceable.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from .contracts import HCDependency, HCFailure, HyperSimulationResult, Pair, Removal


class FixedPointMode(str, Enum):
    """Quantification over HCs associated with the same anchor pair."""

    ALL_HC = "all_hc"
    WITNESS_OR = "query_and_data_or_diagnostic"


def compute_hyper_simulation(
    hv_allowed_pairs: Iterable[Pair],
    dependencies: Iterable[HCDependency],
    *,
    mode: FixedPointMode | str = FixedPointMode.ALL_HC,
) -> HyperSimulationResult:
    """Compute the unique greatest relation closed under all HC dependencies.

    The relation starts from h_v.  A full HC dependency retains its anchor
    ``(u,v)`` exactly when (a) the anchor belongs to D-match and (b) every pair
    in that D-match is still in the relation.  Deletions are gathered against
    one immutable iteration snapshot and applied together.

    A pair with no associated HC is vacuously valid under this closure.  It can
    remain in the relation but cannot by itself enter a complete support core.
    """

    mode = FixedPointMode(mode)
    relation = {(str(left), str(right)) for left, right in hv_allowed_pairs}
    initial_relation = frozenset(relation)
    ordered = tuple(sorted(dependencies, key=lambda value: value.id))
    ids = [dependency.id for dependency in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("logical HC ids must be unique")
    by_anchor: dict[Pair, list[HCDependency]] = {}
    for dependency in ordered:
        # support_only HCs may be used by Consistency as positive witnesses,
        # but they are deliberately unable to create deletion certificates.
        if not dependency.destructive:
            continue
        for pair in sorted(dependency.anchor_pairs):
            by_anchor.setdefault(pair, []).append(dependency)

    removals: list[Removal] = []
    iteration = 0
    while True:
        doomed: list[Removal] = []
        snapshot = frozenset(relation)
        for pair in sorted(snapshot):
            local = tuple(by_anchor.get(pair, ()))
            failures = (
                _all_hc_failures(pair, local, snapshot)
                if mode is FixedPointMode.ALL_HC
                else _witness_failures(pair, local, snapshot)
            )
            if failures:
                doomed.append(
                    Removal(
                        iteration=iteration + 1,
                        pair=pair,
                        failures=failures,
                    )
                )
        if not doomed:
            break
        iteration += 1
        relation.difference_update(removal.pair for removal in doomed)
        removals.extend(doomed)
    return HyperSimulationResult(
        initial_relation=initial_relation,
        relation=frozenset(relation),
        iterations=iteration,
        removals=tuple(removals),
        mode=mode.value,
    )


def _all_hc_failures(
    pair: Pair,
    dependencies: tuple[HCDependency, ...],
    relation: frozenset[Pair],
) -> tuple[HCFailure, ...]:
    """Return one exact contract-failure record for every failed HC.

    The production Rust solver uses reverse indices to discover the same
    failures.  This implementation records them explicitly because conflict
    attribution follows the source document of the failing HC, not merely the
    endpoint of a deleted pair.
    """

    failures: list[HCFailure] = []
    for dependency in dependencies:
        dmatch = dependency.dmatch.effective_pairs
        if pair not in dmatch:
            failures.append(
                HCFailure(
                    hc_id=dependency.id,
                    condition="anchor_membership",
                    reason=("empty_dmatch" if not dmatch else "anchor_not_in_dmatch"),
                )
            )
            continue
        absent = dmatch - relation
        if absent:
            failures.append(
                HCFailure(
                    hc_id=dependency.id,
                    condition="dependency_closure",
                    reason="dmatch_not_closed",
                    missing_pairs=tuple(sorted(absent)),
                )
            )
    return tuple(failures)


def _witness_failures(
    pair: Pair,
    dependencies: tuple[HCDependency, ...],
    relation: frozenset[Pair],
) -> tuple[HCFailure, ...]:
    """Diagnostic alternative: AND query clusters, OR data witnesses."""

    if not dependencies:
        return ()
    by_query_cluster: dict[str, list[HCDependency]] = {}
    for dependency in dependencies:
        by_query_cluster.setdefault(dependency.query_cluster.id, []).append(dependency)
    failed_groups: list[str] = []
    failed_hcs: list[str] = []
    missing: set[Pair] = set()
    for query_cluster_id, alternatives in sorted(by_query_cluster.items()):
        if any(
            pair in dependency.dmatch.effective_pairs
            and dependency.dmatch.effective_pairs <= relation
            for dependency in alternatives
        ):
            continue
        failed_groups.append(query_cluster_id)
        for dependency in alternatives:
            failed_hcs.append(dependency.id)
            dmatch = dependency.dmatch.effective_pairs
            if pair not in dmatch:
                missing.add(pair)
            missing.update(dmatch - relation)
    if not failed_groups:
        return ()
    reason = "no_valid_data_witness:" + ",".join(failed_groups)
    return tuple(
        HCFailure(
            hc_id=hc_id,
            condition="diagnostic",
            reason=reason,
            missing_pairs=tuple(sorted(missing)),
        )
        for hc_id in sorted(set(failed_hcs))
    )


__all__ = ["FixedPointMode", "compute_hyper_simulation"]
