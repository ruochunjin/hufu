"""Built-in predicate library.

Covers the predicate families of the three integrated rule systems:

- similarity: ``vec_sim`` (CDR's VecSim)
- NLI: ``nli_entails`` / ``nli_contradicts`` (RICO's inference predicates)
- attribute logic: ``attr_equal`` / ``attr_conflict`` / ``attr_present``
  (CDR object/action comparisons, RICO entity/value predicates, OMR logic)
- numeric/temporal: ``numeric_conflict`` / ``temporal_order`` (RICO)
- lexical: ``fuzzy_match`` (schema/label matching; stdlib only)

All neural predicates go through ``Services`` — no model is imported or
loaded unless a rule actually uses them.
"""

from __future__ import annotations

import difflib
from typing import Any, Mapping

from hufu.rules.registry import Item, Services, predicate


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    import math

    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


# ---------------------------------------------------------------------------
# Similarity predicates (CDR VecSim)
# ---------------------------------------------------------------------------


@predicate("vec_sim")
def vec_sim(a: Item, b: Item, params: Mapping[str, Any], services: Services) -> float:
    """Cosine similarity of the two items' embeddings.

    Uses precomputed ``Item.embedding`` vectors when present, otherwise
    encodes ``Item.text`` through the embedding service.
    """

    if a.embedding is not None and b.embedding is not None:
        return _cosine(a.embedding, b.embedding)
    emb_a, emb_b = services.embed([a.text, b.text])
    return _cosine(tuple(emb_a), tuple(emb_b))


# ---------------------------------------------------------------------------
# NLI predicates (RICO inference predicates)
# ---------------------------------------------------------------------------


@predicate("nli_entails")
def nli_entails(a: Item, b: Item, params: Mapping[str, Any], services: Services) -> float:
    """Score that text a entails text b (directional)."""

    return services.nli_score(a.text, b.text, "entailment")


@predicate("nli_contradicts")
def nli_contradicts(a: Item, b: Item, params: Mapping[str, Any], services: Services) -> float:
    """Score that text a contradicts text b (directional)."""

    return services.nli_score(a.text, b.text, "contradiction")


# ---------------------------------------------------------------------------
# Attribute logic predicates (CDR object/action, RICO entity, OMR logic)
# ---------------------------------------------------------------------------


def _attr(item: Item, name: str) -> Any:
    return item.attrs.get(name)


@predicate("attr_equal")
def attr_equal(a: Item, b: Item, params: Mapping[str, Any], services: Services) -> float:
    """1.0 when both items share the same value for ``attr`` (non-null)."""

    name = str(params["attr"])
    va, vb = _attr(a, name), _attr(b, name)
    if va is None or vb is None:
        return 0.0
    return 1.0 if va == vb else 0.0


@predicate("attr_conflict")
def attr_conflict(a: Item, b: Item, params: Mapping[str, Any], services: Services) -> float:
    """1.0 when both items have non-null, different values for ``attr``.

    With ``opposites: [x, y]`` in params, fires only when the two values are
    that specific opposing pair (CDR action conflicts: on/off, open/close).
    """

    name = str(params["attr"])
    va, vb = _attr(a, name), _attr(b, name)
    if va is None or vb is None or va == vb:
        return 0.0
    opposites = params.get("opposites")
    if opposites is not None:
        lo, hi = (str(v) for v in opposites)
        return 1.0 if {str(va), str(vb)} == {lo, hi} else 0.0
    return 1.0


@predicate("attr_present")
def attr_present(a: Item, b: Item, params: Mapping[str, Any], services: Services) -> float:
    """1.0 when ``attr`` is present on both items."""

    name = str(params["attr"])
    return 1.0 if _attr(a, name) is not None and _attr(b, name) is not None else 0.0


# ---------------------------------------------------------------------------
# Numeric / temporal predicates (RICO numeric & tense constraints)
# ---------------------------------------------------------------------------


@predicate("numeric_conflict")
def numeric_conflict(a: Item, b: Item, params: Mapping[str, Any], services: Services) -> float:
    """1.0 when numeric ``attr`` values differ by more than ``rel_tol``.

    ``rel_tol`` (default 0.0) is the tolerated relative difference; with
    ``unit`` set, both values must also share the same unit attribute.
    """

    name = str(params["attr"])
    va, vb = _attr(a, name), _attr(b, name)
    if va is None or vb is None:
        return 0.0
    try:
        fa, fb = float(va), float(vb)
    except (TypeError, ValueError):
        return 0.0
    unit = params.get("unit")
    if unit is not None and _attr(a, str(unit)) != _attr(b, str(unit)):
        return 0.0
    rel_tol = float(params.get("rel_tol", 0.0))
    scale = max(abs(fa), abs(fb), 1e-12)
    return 1.0 if abs(fa - fb) / scale > rel_tol else 0.0


@predicate("temporal_order")
def temporal_order(a: Item, b: Item, params: Mapping[str, Any], services: Services) -> float:
    """1.0 when ``attr`` of a strictly precedes that of b (ISO-8601 strings)."""

    name = str(params.get("attr", "timestamp"))
    va, vb = _attr(a, name), _attr(b, name)
    if va is None or vb is None:
        return 0.0
    return 1.0 if str(va) < str(vb) else 0.0


# ---------------------------------------------------------------------------
# Lexical predicates
# ---------------------------------------------------------------------------


@predicate("fuzzy_match")
def fuzzy_match(a: Item, b: Item, params: Mapping[str, Any], services: Services) -> float:
    """Sequence similarity of two strings (attribute values or raw texts).

    With ``attr`` set, compares that attribute of both items; otherwise
    compares ``Item.text``.  Pure stdlib (difflib) — no dependencies.
    """

    name = params.get("attr")
    if name is not None:
        va, vb = _attr(a, str(name)), _attr(b, str(name))
        if va is None or vb is None:
            return 0.0
        left, right = str(va), str(vb)
    else:
        left, right = a.text, b.text
    return difflib.SequenceMatcher(None, left.lower(), right.lower()).ratio()


__all__: list[str] = []
