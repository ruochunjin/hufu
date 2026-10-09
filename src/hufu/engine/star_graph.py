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
from typing import TYPE_CHECKING

from hufu.core.log import getLogger
from hufu.hypergraph.hypergraph import Hypergraph as LocalHypergraph
from hufu.hypergraph.hypergraph import Vertex as LocalVertex

if TYPE_CHECKING:
    from .decision import DecisionChain

logger = getLogger(__name__)

_VALUE_TYPES = {"NUMBER", "TEMPORAL", "GPE", "LOC", "FAC", "COUNTRY", "PERSON", "ORG"}


def _value_families() -> frozenset[str]:
    """Type families that carry identity values (LOCATION/AGENT/QUANTITY/...)."""

    return frozenset(_type_family(t) for t in _VALUE_TYPES)
_NEGATION = re.compile(r"\b(not|n't|never|no|none|nothing|nobody|nowhere|neither|nor|without)\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUM_UNIT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([A-Za-zÀ-ÿ]+)?")


@dataclass(frozen=True)
class StarFact:
    """One incident hyperedge of a star center, with its co-participants."""

    edge_index: int
    predicate: str
    predicate_is_verb: bool
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
    # direct=True: fired from structure alone (typed slot swap, numeric
    # mismatch, role reversal, antonym pair, frame-identical polarity flip).
    # direct=False: the NLI model confirmed a cued candidate — the
    # strict profile counts only direct fires in the verdict.
    direct: bool = True


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
# Passive subjects: dobj↔nsubjpass is voice alternation, not role reversal.
_PASSIVE_ROLES = {"nsubjpass", "csubjpass"}

# R-ROLE fires directly only for asymmetric relation verbs — reversing
# these is a genuine conflict ("sold", "hired", "evicted").  Reciprocal or
# benign verbs ("love", "told", "met") swap freely in narrative and are
# demoted to the confirmation stage.  Nominal predicates never qualify.
_ASYMMETRIC_VERBS = {
    "sell", "sold", "buy", "bought", "marry", "married", "employ", "employed",
    "hire", "hired", "fire", "fired", "evict", "evicted", "acquire", "acquired",
    "defeat", "defeated", "beat", "own", "owned", "inherit", "inherited",
    "arrest", "arrested", "convict", "convicted", "appoint", "appointed",
    "promote", "promoted", "command", "commanded", "govern", "governed",
    "adopt", "adopted", "kill", "killed", "murder", "murdered", "invade",
    "invaded", "conquer", "conquered", "sentence", "sentenced", "sue", "sued",
    "prosecute", "prosecuted", "expel", "expelled", "exile", "exiled",
    "bite", "bit", "bitten", "poach", "poached",
}
# Speech / propositional-attitude predicates: their SUBJECTS (speakers,
# thinkers) swap freely in narrative — "Spider-Man said ..." / "the father
# said ..." is dialogue, not a conflict; quoted polarity flips are likewise
# per-speaker.  Value-slot and negation rules never fire on a subject slot
# of these (doc300 FP attribution, 2026-10-07).
_SPEECH_VERBS = {
    "say", "said", "tell", "told", "report", "reported", "claim", "claimed",
    "add", "added", "announce", "announced", "state", "stated", "declare",
    "declared", "explain", "explained", "ask", "asked", "reply", "replied",
    "respond", "responded", "comment", "commented", "think", "thought",
    "believe", "believed", "note", "noted", "remark", "remarked",
    "according",
}
# Verb/adjective antonym pairs for R-ANTONYM: same slot with antonymous
# predicates or values is a direct conflict (rose/fell, success/failure).
_ANTONYM_PAIRS = frozenset(
    frozenset(pair)
    for pair in [
        ("rise", "fall"), ("rose", "fell"), ("open", "close"), ("increase", "decrease"),
        ("succeed", "fail"), ("success", "failure"), ("win", "lose"), ("won", "lost"),
        ("gain", "lose"), ("expand", "shrink"), ("strengthen", "weaken"),
        ("approve", "reject"), ("support", "oppose"), ("accept", "refuse"),
        ("advance", "retreat"), ("ascend", "descend"), ("love", "hate"),
        ("praise", "blame"), ("promote", "demote"), ("improve", "worsen"),
        ("survive", "die"), ("live", "die"), ("win", "defeat"), ("grew", "shrank"),
        ("rise", "decline"), ("rose", "declined"), ("fell", "rose"),
        ("pass", "fail"), ("expand", "shrank"),
    ]
)


def _word_stems(word: str) -> set[str]:
    """Crude inflection stems: closed→close, increases→increase, opposed→oppose.

    The antonym lexicon lists base forms; narrative text inflects them, so
    matching without stemming misses "supported vs opposed" (golden S-ANT-5).
    """

    stems = {word}
    if word.endswith("ies") and len(word) > 4:
        stems.add(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        stems.add(word[:-2])
        stems.add(word[:-1])
    elif word.endswith("s") and len(word) > 3:
        stems.add(word[:-1])
    if word.endswith("ed") and len(word) > 3:
        stems.add(word[:-2])
        stems.add(word[:-1])
    if word.endswith("ing") and len(word) > 4:
        stems.add(word[:-3])
        stems.add(word[:-3] + "e")
    return stems


def _antonymous(text_a: str, text_b: str) -> bool:
    """Whether two predicate/value texts are antonymic via the lexicon."""

    words_a = set(re.findall(r"[a-z]+", text_a.lower()))
    words_b = set(re.findall(r"[a-z]+", text_b.lower()))
    stems_a = {stem for word in words_a for stem in _word_stems(word)}
    stems_b = {stem for word in words_b for stem in _word_stems(word)}
    for word_a in stems_a:
        for word_b in stems_b:
            if word_a != word_b and frozenset({word_a, word_b}) in _ANTONYM_PAIRS:
                return True
    return False


_RECIPROCAL_VERBS = {
    "love", "loved", "meet", "met", "tell", "told", "say", "said", "visit",
    "visited", "know", "knew", "like", "liked", "marry", "kiss", "hug",
    "call", "called", "email", "emailed", "phone", "phoned", "see", "saw",
    "is", "was", "are", "were", "be", "been",
}

# Function words stripped before comparing predicate cores under negation:
# "attended" vs "did not attend" must reduce to the same stem.
_NEGATION_FUNCTION_WORDS = {
    "do", "does", "did", "not", "never", "no", "nt", "cannot",
    "is", "was", "are", "were", "be", "been", "have", "has", "had",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
}


def _predicate_core(predicate: str) -> set[str]:
    """Content stems of a predicate with auxiliaries/negation stripped."""

    return {
        stem
        for word in re.findall(r"[a-z]+", predicate.lower())
        if word not in _NEGATION_FUNCTION_WORDS
        for stem in _word_stems(word)
    }


_SPEECH_VERB_STEMS = frozenset(
    stem for word in _SPEECH_VERBS for stem in _word_stems(word)
)


_DETERMINERS = {
    "the", "a", "an", "this", "that", "these", "those",
    "his", "her", "its", "their", "our", "my", "your",
}


def _norm_value(text: str) -> str:
    """Surface normalize for value comparison: casefold + drop leading
    determiners ("that winter" == "winter"; parser demonstrative flips must
    not break frame anchors — S-NEG-5)."""

    words = text.strip().casefold().split()
    while words and words[0] in _DETERMINERS:
        words = words[1:]
    return " ".join(words)


def _heads_overlap(only_a: set[str], only_b: set[str]) -> bool:
    """Whether swapped values share a head-token stem ("an official" vs
    "government officials" — a description variant, not a conflict)."""

    def heads(values: set[str]) -> set[str]:
        result: set[str] = set()
        for value in values:
            words = re.findall(r"[a-z]+", value)
            if words:
                result.update(_word_stems(words[-1]))
        return result

    return bool(heads(only_a) & heads(only_b))

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

# Ordinals/cardinals/"only" mark event tallies ("first defeat", "only two
# wins") where antonymy is not boolean polarity (R-ANTONYM guard).
_COUNTED_CONTEXT = re.compile(
    r"\b(first|second|third|fourth|fifth|only|twice|thrice|\d+|"
    r"one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)


def _role_class(role: str) -> str:
    """Grammatical role class used by R-ROLE (reversal detection)."""

    if role in _SUBJECT_ROLES:
        return "SUBJ"
    if role in _OBJECT_ROLES:
        return "OBJ"
    return "OTHER"


def _is_passive_fact(fact: "StarFact") -> bool:
    """Fact uses passive voice (center or any participant is a passive subject).

    "The painting was sold to the museum by the collector" parses the agent
    as plain pobj, so a role-class check alone reads the active/passive pair
    as a reversal (golden S-CLEAN-3); voice must be compared instead.
    """

    return fact.center_role in _PASSIVE_ROLES or any(
        role in _PASSIVE_ROLES for _text, _type, role in fact.others
    )


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
    # Text-identity fusion: vertices with identical resolved surface text are
    # one entity for star purposes even when coref did not merge them
    # ("Alexander Fleming discovered penicillin. Howard Florey discovered
    # penicillin." — S-PERSON-3, two same-text penicillin vertices).  Word-
    # boundary containment fuses further ("discovered penicillin" ⊇
    # "penicillin" — parse asymmetry merges verb+object on one side only).
    texts = [vertex.text().strip().casefold() for vertex in local.vertices]
    parent = list(range(len(local.vertices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(local.vertices)):
        for j in range(i + 1, len(local.vertices)):
            if not texts[i] or not texts[j]:
                continue
            if texts[i] == texts[j]:
                parent[find(i)] = find(j)
                continue
            shorter, longer = (texts[i], texts[j]) if len(texts[i]) <= len(texts[j]) else (texts[j], texts[i])
            if re.search(r"\b" + re.escape(shorter) + r"\b", longer):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for index, vertex in enumerate(local.vertices):
        groups.setdefault(find(index), []).append(vertex.id)
    for center_ids in groups.values():
        vertex = next(v for v in local.vertices if v.id == center_ids[0])
        edge_ids: list[int] = []
        for vid in center_ids:
            edge_ids.extend(incidence.get(vid, []))
        if len(set(edge_ids)) < 2:
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
                    predicate_is_verb=edge.root.is_verb(),
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


def _numbers_with_units(value: str) -> set[float]:
    """Numeric tokens of a value text, unit-converted when a known unit
    follows (42.195 km == 42195 m — golden S-CLEAN-10)."""

    from .reference_check import _UNIT_TABLE

    found: set[float] = set()
    for match in _NUM_UNIT.finditer(value):
        number = float(match.group(1).replace(",", ""))
        unit = match.group(2).lower() if match.group(2) else None
        if unit and unit in _UNIT_TABLE:
            number *= _UNIT_TABLE[unit][1]
        found.add(round(number, 6))
    return found


def _numeric_conflict(fa: StarFact, fb: StarFact) -> tuple[str, str, int, int] | None:
    """Same-typed NUMBER co-values with different numbers (same role or any).

    Returns (role, detail, count_a, count_b) — the per-fact distinct-number
    counts let callers distinguish a functional value (1 number per fact)
    from an enumeration list ("Nars $23; Revlon $8" — doc300 FP guard).
    """

    nums_a: dict[str, set[float]] = {}
    nums_b: dict[str, set[float]] = {}
    for value, vtype, role in fa.others:
        if vtype == "NUMBER":
            nums_a.setdefault(role, set()).update(_numbers_with_units(value))
    for value, vtype, role in fb.others:
        if vtype == "NUMBER":
            nums_b.setdefault(role, set()).update(_numbers_with_units(value))
    if not nums_a or not nums_b:
        return None
    flat_a = {n for values in nums_a.values() for n in values}
    flat_b = {n for values in nums_b.values() for n in values}
    if flat_a and flat_b and flat_a != flat_b:
        shared_roles = set(nums_a) & set(nums_b)
        role = sorted(shared_roles)[0] if shared_roles else sorted(nums_a)[0]
        return role, f"{sorted(flat_a)} vs {sorted(flat_b)}", len(flat_a), len(flat_b)
    return None


def _type_family(vtype: str) -> str:
    """Bridge TYPE_GROUPS parity: GPE/LOC are both LOCATION, PERSON/NORP are
    both AGENT, so "built in Chartres (GPE)" vs "built in Reims (LOC)"
    shares a slot family (golden S-PLACE-3)."""

    from .scoring import TYPE_GROUPS

    if vtype == "NUMBER":
        return "QUANTITY"
    return TYPE_GROUPS.get(vtype, vtype)


def _value_conflicts(fa: StarFact, fb: StarFact) -> tuple[str, str, str, str] | None:
    """Same role + same value-type FAMILY + different surface (typed slot)."""

    by_role_a: dict[tuple[str, str], tuple[str, str]] = {}
    for value, vtype, role in fa.others:
        if vtype in _VALUE_TYPES:
            by_role_a[(role, _type_family(vtype))] = (value, vtype)
    for value, vtype, role in fb.others:
        key = (role, _type_family(vtype))
        if vtype in _VALUE_TYPES and key in by_role_a:
            other, other_type = by_role_a[key]
            if other.strip().casefold() != value.strip().casefold():
                return role, vtype, other, value
    return None


def check_star(
    star: Star,
    *,
    nli_threshold: float = 0.5,
    similar_threshold: float = 0.8,
    chain=None,
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
            pa = fa.predicate.strip().casefold()
            pb = fb.predicate.strip().casefold()
            shorter_p, longer_p = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
            # Parse asymmetry: "discovered penicillin" vs "discovered" — a
            # verb+object merge on one side only (S-PERSON-3).  Word-boundary
            # containment counts as the same predicate.
            same_predicate = pa == pb or bool(
                shorter_p and re.search(r"\b" + re.escape(shorter_p) + r"\b", longer_p)
            )
            # Value-slot rules only apply when the center is the SUBJECT of
            # both facts: "Hemingway born in X vs Y" is a real conflict;
            # "Margaret's return vs Chris's return" (center is an event noun
            # in an object slot) is just two different events (corpus FP
            # attribution: R-VALUE dominated false flags).
            center_is_subject = (
                fa.center_role in _SUBJECT_ROLES and fb.center_role in _SUBJECT_ROLES
            )
            # Predicate-centered stars (the shared center IS the root in both
            # facts) compare subject values — exactly the slot we want:
            # "Fleming discovered penicillin" vs "Florey discovered penicillin".
            center_is_predicate = fa.center_role == "ROOT" and fb.center_role == "ROOT"
            # Value-slot rules compare ASSERTED values; a polarity mismatch
            # makes values incomparable only on NON-CONTENT predicates —
            # copula/aux-only or speech frames ("he wasn't the best
            # collaborator" vs "he was an actor" is noise).  On content verbs
            # the value difference still carries the conflict ("released in
            # 1998" vs "not released until 2001" — CD-POS recall).
            core_for_asym = _predicate_core(pa)
            negation_asym = (fa.negated != fb.negated) and (
                not core_for_asym or bool(core_for_asym & _SPEECH_VERB_STEMS)
            )
            if same_predicate:
                # R-ROLE: same predicate, the center sits in subject position
                # in one fact and object position in the other — a role
                # reversal ("museum sold X" vs "X sold to museum").  Voice
                # alternation is NOT a reversal: "the collector sold the
                # painting" vs "the painting was sold by the collector" puts
                # the collector in nsubj vs pobj yet describes the same event,
                # so mixed-voice pairs (and any passive-subject role) are
                # skipped (golden S-CLEAN-3).
                if (
                    fa.predicate_is_verb
                    and _role_class(fa.center_role) != _role_class(fb.center_role)
                    and "OTHER" not in {_role_class(fa.center_role), _role_class(fb.center_role)}
                    and not ({fa.center_role, fb.center_role} & _PASSIVE_ROLES)
                    and _is_passive_fact(fa) == _is_passive_fact(fb)
                ):
                    lemma = fa.predicate.strip().lower().split()[0]
                    if lemma in _RECIPROCAL_VERBS:
                        continue
                    if lemma not in _ASYMMETRIC_VERBS:
                        cued_pairs.append((fa, fb, "R-ROLE"))
                        continue
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
                    if chain is not None:
                        chain.add(
                            "rule",
                            f"R-ROLE fires: role reversal ({fa.center_role} vs {fb.center_role})",
                            rule="R-ROLE",
                        )
                    continue
                # R-NUMERIC: same predicate, numbers differ in a shared role.
                # Only when the center is the subject of both facts — or the
                # predicate itself, in which case the SUBJECTS must agree:
                # "the striker scored 23 goals" vs "the midfielder scored 4
                # goals" is two players, not a numeric conflict (S-CLEAN-9).
                if center_is_subject or center_is_predicate:
                    subject_mismatch = False
                    if center_is_predicate:
                        subj_a = {
                            t.strip().casefold()
                            for t, _ty, r in fa.others
                            if r in _SUBJECT_ROLES
                        }
                        subj_b = {
                            t.strip().casefold()
                            for t, _ty, r in fb.others
                            if r in _SUBJECT_ROLES
                        }
                        subject_mismatch = bool(
                            subj_a and subj_b and subj_a != subj_b
                        )
                    numeric = None if subject_mismatch or negation_asym else _numeric_conflict(fa, fb)
                    # appos/conj roles holding MULTIPLE numbers per fact are
                    # enumeration lists ("Nars $23; Revlon $8" vs "MAC $14"),
                    # not functional slots; a single-number appos is a parse
                    # variant of the same slot and still counts (S-NUM-5).
                    if numeric is not None:
                        role, detail, count_a, count_b = numeric
                        enumeration = role in {"appos", "conj"} and (
                            count_a > 1 or count_b > 1
                        )
                    else:
                        enumeration = True
                    if numeric is not None and not enumeration:
                        role, detail = numeric[0], numeric[1]
                        conflicts.append(
                            StarConflict(
                                rule="R-NUMERIC",
                                center_text=star.center.text(),
                                fact_a=fa,
                                fact_b=fb,
                                explanation=f"numbers conflict in role {role}: {detail}",
                            )
                        )
                        if chain is not None:
                            chain.add(
                                "rule",
                                f"R-NUMERIC fires: numbers differ in role {role} ({detail})",
                                rule="R-NUMERIC", role=role, values=detail,
                            )
                        continue
                # R-VALUE (single-slot difference): the two facts share the
                # predicate frame; across the roles BOTH facts have, exactly
                # one differs (a single typed value swap) while the others are
                # identical.  The frame anchor is an identical shared slot OR
                # the predicate itself (equal predicates anchor the frame;
                # "discovered penicillin" vs "discovered" counts via
                # word-boundary containment).  "Fleming vs Florey discovered
                # penicillin" fires (only nsubj differs); "Margaret's return
                # vs Chris's return" does not (nsubj AND dobj differ);
                # "landed on Mars in 2021" vs "on Venus in 2021" fires (only
                # the LOC value swaps, 2021 identical — S-PLACE-4).
                if (center_is_subject or center_is_predicate) and not negation_asym:
                    vals_a: dict[str, set[str]] = {}
                    vals_b: dict[str, set[str]] = {}
                    type_of_a: dict[tuple[str, str], str] = {}
                    type_of_b: dict[tuple[str, str], str] = {}
                    # UNKNOWN-typed values COUNT for slot-difference (an
                    # unexplained extra difference = a different event —
                    # S-CLEAN-9's "midfielder" must veto the goals slot);
                    # they just cannot be the firing value themselves.
                    for text, vtype, role in fa.others:
                        if role == "ROOT":
                            continue
                        key = text.strip().casefold()
                        vals_a.setdefault(role, set()).add(key)
                        type_of_a[(role, key)] = vtype
                    for text, vtype, role in fb.others:
                        if role == "ROOT":
                            continue
                        key = text.strip().casefold()
                        vals_b.setdefault(role, set()).add(key)
                        type_of_b[(role, key)] = vtype
                    shared_roles = set(vals_a) & set(vals_b)
                    all_a = {v for vs in vals_a.values() for v in vs}
                    all_b = {v for vs in vals_b.values() for v in vs}
                    identical = [r for r in shared_roles if vals_a[r] == vals_b[r]]
                    frame_anchor = (
                        pa == pb
                        or bool(identical)
                        or bool(
                            shorter_p
                            and re.search(
                                r"\b" + re.escape(shorter_p) + r"\b", longer_p
                            )
                        )
                    )
                    # A value appearing ANYWHERE in the other fact is an
                    # anchor, not a difference — parser role-label flips
                    # (Emma: dobj vs npadvmod) must not manufacture a second
                    # differing slot (S-PERSON-1 flakiness).
                    differing_slots: list[tuple[str, set[str], set[str]]] = []
                    for r in shared_roles:
                        oa = (vals_a[r] - vals_b[r]) - all_b
                        ob = (vals_b[r] - vals_a[r]) - all_a
                        if oa or ob:
                            differing_slots.append((r, oa, ob))
                    if len(differing_slots) == 1 and frame_anchor:
                        role, only_a, only_b = differing_slots[0]
                        fams_a = {
                            _type_family(type_of_a.get((role, v), "UNKNOWN"))
                            for v in only_a
                        }
                        fams_b = {
                            _type_family(type_of_b.get((role, v), "UNKNOWN"))
                            for v in only_b
                        }
                        # A true slot conflict swaps values of ONE shared type
                        # family — "Austin, Texas" vs "Boston, Massachusetts"
                        # is one swap (S-PLACE-2); mixed-type asymmetries are
                        # different events.
                        if (
                            only_a
                            and only_b
                            and len(fams_a) == 1
                            and fams_a == fams_b
                            and fams_a <= _value_families()
                        ):
                            # Subjects of speech/report verbs swap freely
                            # (different speakers); copula/aux-only predicates
                            # are non-functional for subject slots ("one-third
                            # of Americans are X; one in 10 physicians are X").
                            core = _predicate_core(pa)
                            if role in _SUBJECT_ROLES and (
                                not core or core & _SPEECH_VERB_STEMS
                            ):
                                continue
                            # Same-head agent values are description variants
                            # ("an official" vs "government officials").
                            if fams_a == {"AGENT"} and _heads_overlap(only_a, only_b):
                                continue
                            # A single span holding a conjunct list ("Heidi
                            # Klum and Jamie-Lynn Sigler") is an enumeration,
                            # not a functional slot — lists legitimately
                            # differ across sentences (doc300 FP attribution).
                            # "and" only: commas appear in plain values
                            # ("Springfield, Illinois"), not just lists.
                            if any(" and " in v for v in only_a | only_b):
                                continue
                            vtype = type_of_a[(role, sorted(only_a)[0])]
                            va = ", ".join(sorted(only_a))
                            vb = ", ".join(sorted(only_b))
                            if fams_a == {"QUANTITY"}:
                                nums_a = {
                                    n for v in only_a for n in _numbers_with_units(v)
                                }
                                nums_b = {
                                    n for v in only_b for n in _numbers_with_units(v)
                                }
                                if nums_a == nums_b:
                                    # Unit-converted equality is not a conflict
                                    # (42.195 km == 42195 m).
                                    continue
                            if vtype == "TEMPORAL" and (
                                _RECURRING_TEMPORAL.search(fa.edge_text)
                                or _RECURRING_TEMPORAL.search(fb.edge_text)
                            ):
                                # Recurring states are inclusive ("open on
                                # Mondays" + "open on Tuesdays").
                                continue
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
                            if chain is not None:
                                chain.add(
                                    "rule",
                                    f"R-VALUE fires: {vtype} values differ in role {role} ({va!r} vs {vb!r})",
                                    rule="R-VALUE", value_type=vtype, role=role,
                                )
                            continue
            # R-ANTONYM: same center role with antonymous predicates
            # ("stock price rose" vs "fell") or antonymous values
            # ("a success" vs "a failure") — the antonym lexicon is the
            # deterministic skeleton, no model needed.  Guard: when both
            # facts carry explicit subjects and they differ ("the price rose"
            # vs "the temperature fell" sharing only an adverb center), the
            # antonymy describes different events, not a conflict.
            if fa.center_role == fb.center_role:
                subj_a = {
                    t.strip().casefold()
                    for t, _ty, r in fa.others
                    if r in _SUBJECT_ROLES
                }
                subj_b = {
                    t.strip().casefold()
                    for t, _ty, r in fb.others
                    if r in _SUBJECT_ROLES
                }
                if subj_a and subj_b and subj_a != subj_b:
                    continue
                # Counted/ordinal contexts ("a first home defeat" vs "only
                # two wins") are event tallies, not boolean polarity —
                # antonymy does not apply (doc300 FP attribution).
                if _COUNTED_CONTEXT.search(fa.edge_text) or _COUNTED_CONTEXT.search(
                    fb.edge_text
                ):
                    continue
                pred_antonym = _antonymous(fa.predicate, fb.predicate)
                if pred_antonym:
                    conflicts.append(
                        StarConflict(
                            rule="R-ANTONYM",
                            center_text=star.center.text(),
                            fact_a=fa,
                            fact_b=fb,
                            explanation=f"antonymous predicates: {fa.predicate!r} vs {fb.predicate!r}",
                        )
                    )
                    continue
                for ta, _tya, _ra in fa.others:
                    for tb, _tyb, _rb in fb.others:
                        if ta.strip().casefold() != tb.strip().casefold() and _antonymous(ta, tb):
                            conflicts.append(
                                StarConflict(
                                    rule="R-ANTONYM",
                                    center_text=star.center.text(),
                                    fact_a=fa,
                                    fact_b=fb,
                                    explanation=f"antonymous values: {ta!r} vs {tb!r}",
                                )
                            )
                            break
                    else:
                        continue
                    break

            # R-NEGATION (structural form, direct fire): the center sits in
            # the same slot of both facts, one fact negates its predicate,
            # the predicate CORES match with negation/auxiliaries stripped
            # ("attended" vs "did not attend"), and the facts share at least
            # one anchor value while carrying no conflicting identity-bearing
            # extras — an anchored frame with flipped polarity is a
            # structural conflict ("the queen attended the ceremony" vs "the
            # queen did not attend the ceremony").  Comparison is
            # role-agnostic: parser role flips ("that winter" as npadvmod vs
            # dobj) must not break the anchor (S-NEG-5).  Differing
            # places/people/times signal a genuinely different event.
            if (
                fa.center_role == fb.center_role
                and fa.negated != fb.negated
                and (center_is_subject or center_is_predicate)
                and _predicate_core(fa.predicate) & _predicate_core(fb.predicate)
                # Quoted/reported polarity flips are per-speaker narrative
                # ("I think X" vs "I don't think X" in dialogue), not
                # structural conflicts.
                and not (_predicate_core(fa.predicate) & _SPEECH_VERB_STEMS)
            ):
                vals_a = {
                    _norm_value(text): vtype
                    for text, vtype, role in fa.others
                    if role != "ROOT"
                }
                vals_b = {
                    _norm_value(text): vtype
                    for text, vtype, role in fb.others
                    if role != "ROOT"
                }
                anchors = set(vals_a) & set(vals_b)
                identity_types = {"GPE", "LOC", "FAC", "COUNTRY", "PERSON", "ORG", "TEMPORAL"}
                extras = (set(vals_a) | set(vals_b)) - anchors
                if anchors and not any(
                    (vals_a.get(v) or vals_b.get(v)) in identity_types for v in extras
                ):
                    conflicts.append(
                        StarConflict(
                            rule="R-NEGATION",
                            center_text=star.center.text(),
                            fact_a=fa,
                            fact_b=fb,
                            explanation=(
                                f"negation asymmetry on identical frame: "
                                f"{fa.predicate!r} vs {fb.predicate!r}"
                            ),
                        )
                    )
                    if chain is not None:
                        chain.add(
                            "rule",
                            f"R-NEGATION fires: polarity flip on identical frame "
                            f"({fa.predicate!r} vs {fb.predicate!r})",
                            rule="R-NEGATION",
                        )
                    continue
            # R-NEGATION (weak form, cued): same center role (the center sits
            # in the same slot of both facts), one fact negates its
            # predicate, AND the pair shares a co-value beyond the center
            # ("stayed in Paris" vs "never visited Paris" anchors on
            # Paris/Yorick; "rats have not yet ..." vs "regular rats" shares
            # nothing and is narrative).
            if fa.center_role == fb.center_role and fa.negated != fb.negated:
                center_surface = star.center.text().strip().casefold()
                shared_values = {
                    text.strip().casefold()
                    for text, _vtype, _role in fa.others
                } & {
                    text.strip().casefold()
                    for text, _vtype, _role in fb.others
                }
                if any(value and value != center_surface for value in shared_values):
                    cued_pairs.append((fa, fb, "R-NEGATION"))
            # R-SIMILAR: nearly-identical edge texts sharing a center but
            # phrased differently — catches antonymy and parse asymmetries
            # ("is open on Mondays" vs "is closed on Mondays") that exact
            # predicate matching misses.  NLI decides below.  Guards:
            # reciprocal verbs ("A loved B" / "B loved A" is normal, not a
            # conflict — checked on either predicate, copulas excluded so
            # "is open" vs "is closed" still fires), and same-predicate
            # recurring temporals where the ONLY difference is the temporal
            # value itself ("open on Mondays" vs "on Tuesdays"); pairs that
            # also differ in another attribute ("open" vs "closed" on
            # Mondays) still fire.
            first_a = fa.predicate.strip().lower().split()[0]
            first_b = fb.predicate.strip().lower().split()[0]
            _reciprocal_no_copula = _RECIPROCAL_VERBS - {"is", "was", "are", "were", "be", "been"}
            if first_a in _reciprocal_no_copula or first_b in _reciprocal_no_copula:
                continue
            if fa.predicate.strip().casefold() == fb.predicate.strip().casefold() and (
                _RECURRING_TEMPORAL.search(fa.sentence)
                or _RECURRING_TEMPORAL.search(fb.sentence)
            ):
                def _non_temporal_diff() -> bool:
                    values_a = {
                        (role, vtype, text.strip().casefold())
                        for text, vtype, role in fa.others
                        if vtype not in {"TEMPORAL", "UNKNOWN"}
                    }
                    values_b = {
                        (role, vtype, text.strip().casefold())
                        for text, vtype, role in fb.others
                        if vtype not in {"TEMPORAL", "UNKNOWN"}
                    }
                    return values_a != values_b
                if not _non_temporal_diff():
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
        # NLI runs over full source sentences (full_desc), not the relation
        # fragments — fragmentary inputs ("open on Mondays") lose the
        # polarity signal the cross-encoder needs (CONTRA-antonym).
        probs = get_nli_contradiction_probs_batch(
            [(fa.sentence, fb.sentence) for fa, fb, _rule, _c in nli_pairs]
        )
        for (fa, fb, rule, cosine), prob in zip(nli_pairs, probs):
            if chain is not None:
                chain.add(
                    "confirm",
                    f"NLI on {rule} pair: p={prob:.2f} ({'fires' if prob >= nli_threshold else 'below threshold'})",
                    rule=rule, nli_prob=prob,
                )
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
                        direct=False,
                    )
                )
    return conflicts


def detect_conflicts(
    local: LocalHypergraph,
    *,
    nli_threshold: float = 0.5,
    chain=None,
) -> tuple[list[StarConflict], float]:
    """Run star extraction and the rule set over one built hypergraph."""

    import time

    start = time.perf_counter()
    stars = extract_stars(local)
    if chain is not None:
        chain.add(
            "stars",
            f"{len(stars)} entity stars (shared entities with 2+ facts)",
            stars=len(stars),
        )
    conflicts: list[StarConflict] = []
    for star in stars:
        conflicts.extend(
            check_star(star, nli_threshold=nli_threshold, chain=chain)
        )
    logger.info(
        f"star-graph check: {len(stars)} stars, {len(conflicts)} conflicts "
        f"in {time.perf_counter() - start:.2f}s"
    )
    return conflicts, time.perf_counter() - start


STRICT_RULES = frozenset({"R-NUMERIC", "R-ROLE", "R-VALUE", "R-ANTONYM", "R-NEGATION"})


def check_document_text(
    text: str,
    *,
    nli_threshold: float = 0.5,
    profile: str = "strict",
) -> tuple[bool, float, str, list[StarConflict], "DecisionChain"]:
    """One-call document self-contradiction check (graph-rule path).

    Builds the production hypergraph (pronoun-gated coref applies), extracts
    entity stars, and applies the rule set.  ``profile="strict"`` (default)
    counts only DIRECT structural fires (typed slot swaps, numeric mismatch,
    role reversal, antonym pairs, frame-anchored polarity flips) — conflicts
    provable from structure alone, which carry the measured doc-level
    precision on real text; ``profile="full"`` also reports the
    NLI-confirmed channels (higher recall, lower precision).
    Returns ``(has_contradiction, seconds, evidence, conflicts,
    decision_chain)`` — the chain explains every step of the verdict.
    """

    import time

    from hufu.engine.build_hypergraph import text_to_hypergraph

    from .decision import ChainBuilder

    start = time.perf_counter()
    local = text_to_hypergraph(text)
    chain = ChainBuilder()
    chain.add(
        "build",
        f"document graph built: {len(local.vertices)} vertices, {len(local.hyperedges)} hyperedges",
        vertices=len(local.vertices), edges=len(local.hyperedges),
    )
    conflicts, _ = detect_conflicts(
        local, nli_threshold=nli_threshold, chain=chain
    )
    if profile == "strict":
        # Data-backed default (doc300, 2026-10-06): only conflicts provable
        # from structure alone count in the verdict — typed slot swaps
        # (R-VALUE/R-NUMERIC), role reversals (R-ROLE), antonym pairs
        # (R-ANTONYM), and frame-identical polarity flips (R-NEGATION).
        # NLI-confirmed cues (on similar/negated pairs) dominate
        # false flags on real narrative text, so they are findings-only here.
        conflicts = [
            c for c in conflicts if c.rule in STRICT_RULES and c.direct
        ]
    elif profile != "full":
        raise ValueError(f"unknown profile {profile!r}; expected 'strict' or 'full'")
    elapsed = time.perf_counter() - start
    if conflicts:
        chain.add(
            "verdict",
            f"CONTRADICTORY: {len(conflicts)} conflict(s), first via {conflicts[0].rule}",
            conflicts=len(conflicts),
        )
        first = conflicts[0]
        evidence = (
            f"[CONTRADICTION] {first.rule} around '{first.center_text}':\n"
            f"  • {first.fact_a.edge_text[:160]}\n"
            f"  • {first.fact_b.edge_text[:160]}\n"
            f"  reason: {first.explanation}"
        )
        return True, elapsed, evidence, conflicts, chain.freeze()
    chain.add("verdict", "CLEAN: no star conflicts", conflicts=0)
    return (
        False,
        elapsed,
        f"[CLEAN] {len(local.vertices)} vertices, {len(local.hyperedges)} edges, no star conflicts"
        + (f" (profile={profile})" if profile != "full" else ""),
        conflicts,
        chain.freeze(),
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
