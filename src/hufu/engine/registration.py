"""Explainable two-stage registration for HyperMatch HC candidates.

The semantic HC scorer is intentionally recall-oriented.  Registering every
moderately similar relation in the fixed point, however, turns scorer noise
into false deletions.  This module separates those two decisions:

* a high-confidence HC may participate in the full fixed-point constraint;
* a medium-confidence HC may be *rescued for positive support only* when its
  independently decoded D-match provides a grounded, type-compatible witness.

The rescue never creates a deletion certificate.  It therefore improves the
recall of answer-supporting components without weakening the destructive
threshold used for conflict propagation.  Every outcome returns stable reason
codes so the final context marking remains traceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

Pair = tuple[str, str]


@dataclass(frozen=True, slots=True)
class HCRegistrationDecision:
    """Explicit decision describing whether and how an HC is registered."""

    hc_accepted: bool
    active: bool
    mode: str
    rescued: bool
    positive_support: bool
    destructive_use: bool
    reason_codes: tuple[str, ...]


def _pairs(values: Iterable[tuple[str, str]]) -> frozenset[Pair]:
    return frozenset((str(left), str(right)) for left, right in values)


def decide_hc_registration(
    *,
    probability: float,
    relation_conflict: bool,
    dmatch_pairs: Iterable[Pair],
    anchor_pairs: Iterable[Pair],
    exact_fixed_pairs: Iterable[Pair] = (),
    canonical_role_pairs: Iterable[Pair] = (),
    low_confidence_role_pairs: Iterable[Pair] = (),
    query_variable_ids: Iterable[str] = (),
    support_threshold: float = 0.65,
    rescue_threshold: float = 0.55,
    single_answer_rescue_threshold: float = 0.60,
    destructive_threshold: float = 0.85,
) -> HCRegistrationDecision:
    """Choose full/support-only/none registration using observable evidence.

    A rescue requires an h_v-compatible partial one-to-one D-match (the caller
    validates one-to-one) and one of two high-precision grounding patterns:

    * at least two matched roles, one of which is a fixed lexical anchor or
      both of which have an unambiguous shared canonical role name;
    * one answer-variable pair with an unambiguous shared role name and a
      slightly stronger HC probability.

    A rescued candidate is always ``support_only``.  Empty/contradictory role
    mappings can only become destructive at ``destructive_threshold``.
    """

    thresholds = (
        float(rescue_threshold),
        float(single_answer_rescue_threshold),
        float(support_threshold),
        float(destructive_threshold),
    )
    if not all(0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("HC registration thresholds must be in [0, 1]")
    if not (
        rescue_threshold
        <= single_answer_rescue_threshold
        <= support_threshold
        <= destructive_threshold
    ):
        raise ValueError("HC registration thresholds are not monotonic")

    dmatch = _pairs(dmatch_pairs)
    anchors = _pairs(anchor_pairs)
    fixed = _pairs(exact_fixed_pairs) & dmatch
    canonical = _pairs(canonical_role_pairs) & dmatch
    low_confidence = _pairs(low_confidence_role_pairs) & dmatch
    query_variables = {str(value) for value in query_variable_ids}
    query_pairs = {pair for pair in dmatch if pair[0] in query_variables}

    positive_support = bool(dmatch) and not relation_conflict and dmatch <= anchors
    destructive_use = (
        relation_conflict
        or not dmatch
        or not dmatch <= anchors
        or bool(anchors - dmatch)
    )
    reason_codes: list[str] = []

    ordinary_accept = probability >= support_threshold
    rescued = False
    if not ordinary_accept and probability >= rescue_threshold and positive_support:
        multi_role_grounding = len(dmatch) >= 2 and (
            bool(fixed) or len(canonical) >= 2
        )
        single_answer_grounding = (
            probability >= single_answer_rescue_threshold
            and len(dmatch) == 1
            and bool(query_pairs & canonical)
        )
        if multi_role_grounding:
            rescued = True
            reason_codes.append("dmatch_multi_role_grounding")
        elif single_answer_grounding:
            rescued = True
            reason_codes.append("canonical_answer_role_grounding")

    hc_accepted = ordinary_accept or rescued
    if rescued:
        # Critical safety contract: a medium-confidence semantic relation may
        # contribute evidence but can never delete a match.
        return HCRegistrationDecision(
            hc_accepted=True,
            active=True,
            mode="support_only",
            rescued=True,
            positive_support=True,
            destructive_use=destructive_use,
            reason_codes=tuple(reason_codes + ["below_primary_hc_threshold"]),
        )

    if not hc_accepted:
        return HCRegistrationDecision(
            hc_accepted=False,
            active=False,
            mode="none",
            rescued=False,
            positive_support=positive_support,
            destructive_use=destructive_use,
            reason_codes=("below_hc_threshold_without_grounded_rescue",),
        )

    if low_confidence and positive_support:
        reason_codes.append("positive_mapping_requires_support_only")
        return HCRegistrationDecision(
            hc_accepted=True,
            active=True,
            mode="support_only",
            rescued=False,
            positive_support=True,
            destructive_use=destructive_use,
            reason_codes=tuple(reason_codes),
        )
    if probability >= destructive_threshold:
        reason_codes.append("destructive_threshold_satisfied")
        return HCRegistrationDecision(
            hc_accepted=True,
            active=True,
            mode="full",
            rescued=False,
            positive_support=positive_support,
            destructive_use=destructive_use,
            reason_codes=tuple(reason_codes),
        )
    if positive_support:
        reason_codes.append("grounded_positive_support")
        return HCRegistrationDecision(
            hc_accepted=True,
            active=True,
            mode="support_only" if destructive_use else "full",
            rescued=False,
            positive_support=True,
            destructive_use=destructive_use,
            reason_codes=tuple(reason_codes),
        )
    return HCRegistrationDecision(
        hc_accepted=True,
        active=False,
        mode="none",
        rescued=False,
        positive_support=False,
        destructive_use=destructive_use,
        reason_codes=("low_confidence_destructive_abstention",),
    )


__all__ = ["HCRegistrationDecision", "decide_hc_registration"]
