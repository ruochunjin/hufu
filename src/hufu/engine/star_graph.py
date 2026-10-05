"""Star-graph contradiction detection over the production hypergraph.

Port of the DocChecker (Doc4LLM) technical line: a document can only be
self-contradictory if two of its facts talk about the *same* entity — so
detection is organized around entity-centered star patterns instead of
blind sentence-pair NLI.  The shared entity is the dual-star's virtual link;
its incident hyperedges are the star edges whose values must agree.

Rule set (each rule fires on fact pairs sharing one center):
- R-NUMERIC: same slot (predicate/role), both NUMBER-typed, conflicting
  values after unit conversion ("300 men" vs "3000 men");
- R-NEGATION: same slot, negation asymmetry on the predicate
  ("attended" vs "did not attend");
- R-VALUE: same role and value TYPE (GPE/TEMPORAL/PERSON/...), conflicting
  surface values ("born in London" vs "born in Paris");
- R-CONTRA: NLI contradiction on the rendered edge texts, used only as
  confirmation for pairs already carrying a structural cue.

This is the graph-rule alternative to the hyper-simulation document path;
callers choose per document which engine to run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hufu.core.log import getLogger
from hufu.hypergraph.hypergraph import Hypergraph as LocalHypergraph
from hufu.hypergraph.hypergraph import Vertex as LocalVertex

logger = getLogger(__name__)

_VALUE_TYPES = {"NUMBER", "TEMPORAL", "GPE", "LOC", "FAC", "COUNTRY", "PERSON", "ORG"}
_NEGATION = re.compile(r"\b(not|n't|never|no|none|nothing|nobody|nowhere|neither|nor|without)\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class StarFact:
    """One incident hyperedge of a star center, with its co-participants."""

    edge_index: int
    predicate: str
    center_role: str
    others: tuple[tuple[str, str, str], ...]  # (value, value_type, role)
    edge_text: str
    sentence: str
    negated: bool


@dataclass(frozen=True)
class Star:
    """An entity appearing in two or more hyperedges, with its facts."""

    center: LocalVertex
    facts: tuple[StarFact, ...]


@dataclass(frozen=True)
class StarConflict:
    """A detected contradiction with full provenance."""

    rule: str
    center_text: str
    fact_a: StarFact
    fact_b: StarFact
    explanation: str
    cosine: float = 0.0
    nli_prob: float = 0.0


def _value_type_of(vertex: LocalVertex) -> str:
    """Coarse value type for conflict typing, aligned with the bridge rules.

    Quantity phrases lose their NUM token in retokenization, so digit/word
    prefixes are checked here exactly as in ``bridge.local_semantic_type``;
    without that, "300 men" types PERSON (head noun) and R-NUMERIC never
    fires.  Adjective/adverb participants report ADJ/ADV for R-ANTONYM.
    """

    from .bridge import _contains_number

    if _contains_number(vertex):
        return "NUMBER"
    if vertex.is_adjective():
        return "ADJ"
    if vertex.is_adverb():
        return "ADV"
    entity_type = vertex.type()
    return entity_type.name if entity_type is not None else "UNKNOWN"


_SUBJECT_ROLES = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
_OBJECT_ROLES = {"dobj", "iobj", "pobj", "dative", "attr", "oprd", "pcomp"}

# Plural weekday/month forms ("on Mondays", "on Fridays") and explicit
# recurrence markers signal a recurring STATE, where differing temporal
# values are inclusive rather than conflicting (golden CLEAN-open-daily).
_RECURRING_TEMPORAL = re.compile(
    r"\b(mondays|tuesdays|wednesdays|thursdays|fridays|saturdays|sundays|"
    r"januaries|februaries|marches|aprils|mays|junes|julys|augusts|"
    r"septembers|octobers|novembers|decembers|"
    r"every|each|daily|weekly|monthly|yearly|annually)\b",
    re.IGNORECASE,
)


def _role_class(role: str) -> str:
    """Grammatical role class used by R-ROLE (reversal detection)."""

    if role in _SUBJECT_ROLES:
        return "SUBJ"
    if role in _OBJECT_ROLES:
        return "OBJ"
    return "OTHER"


def extract_stars(local: LocalHypergraph) -> list[Star]:
    """Group incident hyperedges by shared vertices (dual-star centers).

    A vertex with two or more incident hyperedges is a star center: those
    edges are the facts that must mutually agree.  Each fact records the
    edge predicate plus the co-participants (value/type/role) the center
    shares the edge with.
    """

    incidence: dict[int, list[int]] = {}
    for edge_index, edge in enumerate(local.hyperedges):
        for vertex in edge.vertices:
            incidence.setdefault(vertex.id, []).append(edge_index)

    stars: list[Star] = []
    for vertex in local.vertices:
        edge_ids = incidence.get(vertex.id, [])
        if len(edge_ids) < 2:
            continue
        facts: list[StarFact] = []
        for edge_index in edge_ids:
            edge = local.hyperedges[edge_index]
            center_node = edge.current_node(vertex)
            others: list[tuple[str, str, str]] = []
            for participant in edge.vertices:
                if participant == vertex:
                    continue
                node = edge.current_node(participant)
                others.append(
                    (
                        # Original mention text, NOT the coref-resolved vertex
                        # text: resolution merges same-head mentions and
                        # would erase conflicts like "300 men" vs "3000 men".
                        node.text,
                        _value_type_of(participant),
                        node.dep.name,
                    )
                )
            facts.append(
                StarFact(
                    edge_index=edge_index,
                    predicate=edge.root.text(),
                    center_role=center_node.dep.name,
                    others=tuple(others),
                    edge_text=edge.text(),
                    sentence=(edge.full_desc or edge.text()),
                    negated=bool(_NEGATION.search(edge.desc or "")),
                )
            )
        if len(facts) >= 2:
            stars.append(Star(center=vertex, facts=tuple(facts)))
    return stars


def _numeric_conflict(fa: StarFact, fb: StarFact) -> tuple[str, str] | None:
    """Same-typed NUMBER co-values with different numbers (same role or any)."""

    nums_a: dict[str, set[str]] = {}
    nums_b: dict[str, set[str]] = {}
    for value, vtype, role in fa.others:
        if vtype == "NUMBER":
            nums_a.setdefault(role, set()).update(_NUMBER.findall(value))
    for value, vtype, role in fb.others:
        if vtype == "NUMBER":
            nums_b.setdefault(role, set()).update(_NUMBER.findall(value))
    if not nums_a or not nums_b:
        return None
    flat_a = {n for values in nums_a.values() for n in values}
    flat_b = {n for values in nums_b.values() for n in values}
    if flat_a and flat_b and flat_a != flat_b:
        shared_roles = set(nums_a) & set(nums_b)
        role = sorted(shared_roles)[0] if shared_roles else sorted(nums_a)[0]
        return role, f"{sorted(flat_a)} vs {sorted(flat_b)}"
    return None


def _value_conflicts(fa: StarFact, fb: StarFact) -> tuple[str, str, str, str] | None:
    """Same role + same value TYPE + different surface (typed slot conflict)."""

    by_role_a: dict[tuple[str, str], str] = {}
    for value, vtype, role in fa.others:
        if vtype in _VALUE_TYPES:
            by_role_a[(role, vtype)] = value
    for value, vtype, role in fb.others:
        key = (role, vtype)
        if vtype in _VALUE_TYPES and key in by_role_a:
            other = by_role_a[key]
            if other.strip().casefold() != value.strip().casefold():
                return role, vtype, other, value
    return None


def check_star(
    star: Star,
    *,
    nli_threshold: float = 0.5,
    similar_threshold: float = 0.8,
    ***REMOVED***: bool = False,
) -> list[StarConflict]:
    """Apply the star rule set over fact pairs sharing one center."""

    import math

    from hufu.engine.embedding import get_embedding_batch
    from hufu.engine.nli import get_nli_contradiction_probs_batch

    conflicts: list[StarConflict] = []
    cued_pairs: list[tuple[StarFact, StarFact, str]] = []
    for i, fa in enumerate(star.facts):
        for fb in star.facts[i + 1 :]:
            if fa.edge_index == fb.edge_index:
                continue
            same_predicate = (
                fa.predicate.strip().casefold() == fb.predicate.strip().casefold()
            )
            if same_predicate:
                # R-ROLE: same predicate, the center sits in subject position
                # in one fact and object position in the other — a role
                # reversal ("museum sold X" vs "X sold to museum").  Passive
                # alternation is safe: nsubjpass is SUBJ-class, agent is SUBJ.
                if (
                    _role_class(fa.center_role) != _role_class(fb.center_role)
                    and "OTHER" not in {_role_class(fa.center_role), _role_class(fb.center_role)}
                ):
                    conflicts.append(
                        StarConflict(
                            rule="R-ROLE",
                            center_text=star.center.text(),
                            fact_a=fa,
                            fact_b=fb,
                            explanation=(
                                f"role reversal on {fa.predicate!r}: center appears as "
                                f"{fa.center_role} and {fb.center_role}"
                            ),
                        )
                    )
                    continue
                # R-NUMERIC: same predicate, numbers differ in a shared role.
                numeric = _numeric_conflict(fa, fb)
                if numeric is not None:
                    role, detail = numeric
                    conflicts.append(
                        StarConflict(
                            rule="R-NUMERIC",
                            center_text=star.center.text(),
                            fact_a=fa,
                            fact_b=fb,
                            explanation=f"numbers conflict in role {role}: {detail}",
                        )
                    )
                    continue
                # R-VALUE: same predicate slot, typed values differ.
                value_conflict = _value_conflicts(fa, fb)
                if value_conflict is not None:
                    role, vtype, va, vb = value_conflict
                    if vtype == "TEMPORAL":
                        # Recurring-state ambiguity: "open on Mondays" vs "on
                        # Tuesdays" is inclusive, not a conflict; unique
                        # events (signed 1918 vs 1919) still go to NLI.
                        if not (
                            _RECURRING_TEMPORAL.search(fa.edge_text)
                            or _RECURRING_TEMPORAL.search(fb.edge_text)
                        ):
                            cued_pairs.append((fa, fb, "R-TEMPORAL"))
                    else:
                        conflicts.append(
                            StarConflict(
                                rule="R-VALUE",
                                center_text=star.center.text(),
                                fact_a=fa,
                                fact_b=fb,
                                explanation=(
                                    f"{vtype} values conflict in role {role}: {va!r} vs {vb!r}"
                                ),
                            )
                        )
                    continue
            # R-NEGATION: same center role (the center sits in the same slot
            # of both facts), but one fact negates its predicate.
            if fa.center_role == fb.center_role and fa.negated != fb.negated:
                cued_pairs.append((fa, fb, "R-NEGATION"))
            # R-SIMILAR: nearly-identical edge texts sharing a center but
            # phrased differently — catches antonymy and parse asymmetries
            # ("is open on Mondays" vs "is closed on Mondays") that exact
            # predicate matching misses.  NLI decides below.  When both
            # facts share the predicate AND a recurring temporal marker
            # ("open on Mondays" vs "on Tuesdays"), the readings are
            # inclusive and not conflicts; differing predicates
            # ("open" vs "closed") still fire.
            if (
                fa.predicate.strip().casefold() == fb.predicate.strip().casefold()
                and (
                    _RECURRING_TEMPORAL.search(fa.sentence)
                    or _RECURRING_TEMPORAL.search(fb.sentence)
                )
            ):
                continue
            cued_pairs.append((fa, fb, "R-SIMILAR"))

    if not cued_pairs:
        return conflicts
    # One embedding batch for all cued pairs: R-SIMILAR cosine + NLI probs.
    texts: list[str] = []
    for fa, fb, _rule in cued_pairs:
        texts.append(fa.sentence)
        texts.append(fb.sentence)
    vectors = get_embedding_batch(texts)
    nli_pairs: list[tuple[StarFact, StarFact, str, float]] = []
    for index, (fa, fb, rule) in enumerate(cued_pairs):
        va, vb = vectors[2 * index], vectors[2 * index + 1]
        norm_a = math.sqrt(sum(x * x for x in va))
        norm_b = math.sqrt(sum(x * x for x in vb))
        cosine = (
            sum(x * y for x, y in zip(va, vb)) / (norm_a * norm_b)
            if norm_a and norm_b
            else 0.0
        )
        if rule == "R-SIMILAR" and cosine < similar_threshold:
            continue
        nli_pairs.append((fa, fb, rule, cosine))
    if nli_pairs:
        if ***REMOVED***:
            # DocChecker ***REMOVED*** stage: the local small model judges all
            # cued pairs of this star in ONE batched generation.
            from .***REMOVED*** import ***REMOVED***_batch

            verdicts = ***REMOVED***_batch(
                [(fa.sentence, fb.sentence) for fa, fb, _rule, _c in nli_pairs]
            )
            for (fa, fb, rule, cosine), (is_contra, ctype) in zip(nli_pairs, verdicts):
                if is_contra:
                    conflicts.append(
                        StarConflict(
                            rule=rule,
                            center_text=star.center.text(),
                            fact_a=fa,
                            fact_b=fb,
                            explanation=f"***REMOVED*** {rule} conflict (type={ctype}, cos={cosine:.2f})",
                            cosine=cosine,
                            nli_prob=1.0,
                        )
                    )
            return conflicts
        # NLI runs over full source sentences (full_desc), not the relation
        # fragments — fragmentary inputs ("open on Mondays") lose the
        # polarity signal the cross-encoder needs (CONTRA-antonym).
        probs = get_nli_contradiction_probs_batch(
            [(fa.sentence, fb.sentence) for fa, fb, _rule, _c in nli_pairs]
        )
        for (fa, fb, rule, cosine), prob in zip(nli_pairs, probs):
            if prob >= nli_threshold:
                detail = (
                    f"negation-polarity conflict confirmed by NLI (p={prob:.2f})"
                    if rule == "R-NEGATION"
                    else f"near-identical facts conflict (cos={cosine:.2f}, NLI p={prob:.2f})"
                )
                conflicts.append(
                    StarConflict(
                        rule=rule,
                        center_text=star.center.text(),
                        fact_a=fa,
                        fact_b=fb,
                        explanation=detail,
                        cosine=cosine,
                        nli_prob=prob,
                    )
                )
    return conflicts


def detect_conflicts(
    local: LocalHypergraph, *, nli_threshold: float = 0.5, ***REMOVED***: bool = False
) -> tuple[list[StarConflict], float]:
    """Run star extraction and the rule set over one built hypergraph."""

    import time

    start = time.perf_counter()
    stars = extract_stars(local)
    conflicts: list[StarConflict] = []
    for star in stars:
        conflicts.extend(
            check_star(star, nli_threshold=nli_threshold, ***REMOVED***=***REMOVED***)
        )
    logger.info(
        f"star-graph check: {len(stars)} stars, {len(conflicts)} conflicts "
        f"in {time.perf_counter() - start:.2f}s"
    )
    return conflicts, time.perf_counter() - start


def check_document_text(
    text: str, *, nli_threshold: float = 0.5, ***REMOVED***: bool = False
) -> tuple[bool, float, str, list[StarConflict]]:
    """One-call document self-contradiction check (graph-rule path).

    Builds the production hypergraph (pronoun-gated coref applies), extracts
    entity stars, and applies the rule set.  Returns
    ``(has_contradiction, seconds, evidence, conflicts)``.
    """

    import time

    from hufu.engine.build_hypergraph import text_to_hypergraph

    start = time.perf_counter()
    local = text_to_hypergraph(text)
    conflicts, _ = detect_conflicts(
        local, nli_threshold=nli_threshold, ***REMOVED***=***REMOVED***
    )
    elapsed = time.perf_counter() - start
    if conflicts:
        first = conflicts[0]
        evidence = (
            f"[CONTRADICTION] {first.rule} around '{first.center_text}':\n"
            f"  • {first.fact_a.edge_text[:160]}\n"
            f"  • {first.fact_b.edge_text[:160]}\n"
            f"  reason: {first.explanation}"
        )
        return True, elapsed, evidence, conflicts
    return (
        False,
        elapsed,
        f"[CLEAN] {len(local.vertices)} vertices, {len(local.hyperedges)} edges, no star conflicts",
        conflicts,
    )


__all__ = [
    "Star",
    "StarConflict",
    "StarFact",
    "check_document_text",
    "check_star",
    "detect_conflicts",
    "extract_stars",
]
