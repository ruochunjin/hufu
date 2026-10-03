"""L3 dual-track comparison: standard (Rust) vs reference (pure Python) engines.

Both engines answer the same question — does the document structurally cover
the query? — through different pipelines.  Agreement on the golden cases is
the bridge's correctness evidence; divergences must each have a recorded
root cause.

Recorded divergences (2026-10-04):
- F4 (title-merge asymmetry): quoted titles in the query merge into one
  vertex ('The Old Man and the Sea'), while the same title unquoted in the
  document stays split (Old/Man/Sea).  The standard path lists the merged
  title as a critical vertex and cannot match it → contradiction; the
  reference path demotes it (predicate root) → consistent.  Pinned below as
  a documented divergence until the builder fuses known titles.
- F5 (placeholder criticality — FIXED): ``is_critical_vertex`` ignored
  query placeholders without entity/POS signals, so place-free documents
  passed as consistent.  Fixed 2026-10-04: ``is_query()`` vertices are
  always critical.
- F6 (standard matching strictness): "how many meters" splits into
  '?How many' + 'meters'; the Rust simulation matches the placeholder to
  '42195 meters' but rejects 'meters' ↔ '42195 meters' (role/text
  strictness), while the reference D-match accepts it.  The reference
  verdict matches the golden label; the standard path is over-strict here.
- F7 (copula/passive structural asymmetry): interrogatives build a separate
  copula hyperedge (root 'was', role advmod=?Where) while declaratives merge
  'was born' into one root.  In minimal supporting documents the standard
  simulation cannot align the copula edge and misses '?Where' ↔ 'Oak Park';
  the reference vertex-level D-match finds it (golden-correct).  In F4 the
  same gap was masked because '?Where' spuriously matched the title's
  LOC-typed 'Sea' (same root cause as F3).
"""

from __future__ import annotations

import importlib.util
import os
import unittest

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = all(
    importlib.util.find_spec(name)
    for name in ("spacy", "sentence_transformers", "simulation")
)

# Cases engineered to avoid the F4/F6 asymmetries: no quoted titles, no
# unit-noun duplication of the placeholder.
AGREEMENT_CASES = [
    # (query, document, expected_consistent)
    (
        "How many meters is the marathon race?",
        "The marathon race is very long.",
        False,
    ),
    (
        "When was the treaty signed?",
        "The treaty was signed on 11 November 1918 in Compiègne.",
        True,
    ),
    (
        "Where was Ernest Hemingway born?",
        "Ernest Hemingway was a famous writer. He wrote many novels.",
        False,
    ),
]

F4_CASE_QUERY = "Where was the author of 'The Old Man and the Sea' born?"
F4_CASE_DOC = (
    "Ernest Hemingway wrote The Old Man and the Sea. "
    "He was born in Oak Park, Illinois, in 1899."
)
F6_CASE_QUERY = "How many meters is the marathon race?"
F6_CASE_DOC = "The marathon race is 42195 meters."
F7_CASE_QUERY = "Where was Ernest Hemingway born?"
F7_CASE_DOC = "Ernest Hemingway was born in Oak Park, Illinois, in 1899."


def _check(query: str, doc: str, *, engine: str):
    import hufu

    return hufu.check_pair(query, doc, distance_threshold=-1.0, engine=engine)


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1, models, and simulation")
class DualTrackAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("HUFU_SPACY_MODEL", "en_core_web_sm")

    def test_verdicts_agree_on_golden_cases(self) -> None:
        disagreements = []
        for query, doc, expected in AGREEMENT_CASES:
            with self.subTest(query=query):
                standard = _check(query, doc, engine="standard")
                reference = _check(query, doc, engine="reference")
                if standard.consistent != reference.consistent:
                    disagreements.append(
                        f"{query!r}: standard={standard.consistent} "
                        f"({standard.evidence.splitlines()[0]}), "
                        f"reference={reference.consistent} "
                        f"({reference.evidence.splitlines()[0]})"
                    )
                # Both engines should also match the golden expectation.
                self.assertEqual(
                    standard.consistent,
                    expected,
                    f"standard engine wrong on {query!r}: {standard.evidence}",
                )
                self.assertEqual(
                    reference.consistent,
                    expected,
                    f"reference engine wrong on {query!r}: {reference.evidence}",
                )
        self.assertFalse(
            disagreements,
            "dual-track disagreements:\n" + "\n".join(disagreements),
        )


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1, models, and simulation")
class DocumentedDivergenceTests(unittest.TestCase):
    """Pin the known F4 divergence so fixing it flips this test."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("HUFU_SPACY_MODEL", "en_core_web_sm")

    def test_f4_title_merge_divergence(self) -> None:
        standard = _check(F4_CASE_QUERY, F4_CASE_DOC, engine="standard")
        reference = _check(F4_CASE_QUERY, F4_CASE_DOC, engine="reference")
        # Standard: merged title vertex unmatched → contradiction.
        self.assertFalse(standard.consistent, standard.evidence)
        self.assertIn("The Old Man and the Sea", standard.evidence)
        # Reference: title demoted to predicate root, ?Where covered → consistent.
        self.assertTrue(reference.consistent, reference.evidence)

    def test_f6_unit_noun_strictness_divergence(self) -> None:
        standard = _check(F6_CASE_QUERY, F6_CASE_DOC, engine="standard")
        reference = _check(F6_CASE_QUERY, F6_CASE_DOC, engine="reference")
        # Standard rejects 'meters' ↔ '42195 meters' (over-strict) → contradiction.
        self.assertFalse(standard.consistent, standard.evidence)
        # Reference accepts the pair; golden label is consistent.
        self.assertTrue(reference.consistent, reference.evidence)

    def test_f7_copula_asymmetry_divergence(self) -> None:
        standard = _check(F7_CASE_QUERY, F7_CASE_DOC, engine="standard")
        reference = _check(F7_CASE_QUERY, F7_CASE_DOC, engine="reference")
        # Standard cannot align the copula edge in the minimal supporting
        # document → ?Where unmatched → contradiction (golden: consistent).
        self.assertFalse(standard.consistent, standard.evidence)
        self.assertIn("?Where", standard.evidence)
        # Reference matches ?Where ↔ Oak Park; golden label is consistent.
        self.assertTrue(reference.consistent, reference.evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
