"""L2 end-to-end consistency verdicts over the reference engine.

Golden query–document pairs with expected verdicts, run through
``check_pair(engine='reference')`` (pure Python, no Rust).  Two modes:

* default distance gate (0.25) — realistic RAG post-checker behaviour;
* forced structural mode (threshold=-1) — bypasses the distance gate so the
  HyperMatch structural check alone decides, which is what the L3 dual-track
  comparison validates.

Requires the embedding model (HUFU_EMBED_MODEL, default Qwen3-Embedding-0.6B).
"""

from __future__ import annotations

import importlib.util
import os
import unittest

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = all(
    importlib.util.find_spec(name) for name in ("spacy", "sentence_transformers")
)

HEMINGWAY_QUERY = "Where was the author of 'The Old Man and the Sea' born?"
HEMINGWAY_SUPPORT = (
    "Ernest Hemingway wrote The Old Man and the Sea. "
    "He was born in Oak Park, Illinois, in 1899."
)
# Same entities, but no place mention at all: the ?Where placeholder has no
# type-compatible match target, so structural coverage must fail.  (Mentioning
# the book title does not work here: its "Sea" token is WordNet-typed LOC and
# would spuriously cover the placeholder — recorded as finding F3.)
HEMINGWAY_NO_PLACE = "Ernest Hemingway was a famous writer. He wrote many novels."

RACE_QUERY = "How many meters is the marathon race?"
RACE_SUPPORT = "The marathon race is 42195 meters."
RACE_NO_NUMBER = "The marathon race is very long."


def _check(query: str, doc: str, *, threshold: float):
    import hufu

    return hufu.check_pair(query, doc, distance_threshold=threshold, engine="reference")


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and models extra")
class ReferenceDistanceGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["HUFU_SPACY_MODEL"] = "en_core_web_sm"  # pinned (get_nlp reloads per suite)

    def test_supporting_pair_consistent_via_gate(self) -> None:
        verdict = _check(HEMINGWAY_QUERY, HEMINGWAY_SUPPORT, threshold=0.25)
        self.assertTrue(verdict.consistent, verdict.evidence)
        self.assertIn("[CONSISTENT]", verdict.evidence)

    def test_verdict_carries_distance(self) -> None:
        verdict = _check(HEMINGWAY_QUERY, HEMINGWAY_SUPPORT, threshold=0.25)
        self.assertIsNotNone(verdict.distance)
        self.assertGreaterEqual(verdict.distance, 0.0)


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and models extra")
class ReferenceStructuralTests(unittest.TestCase):
    """threshold=-1 forces the structural HyperMatch decision."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["HUFU_SPACY_MODEL"] = "en_core_web_sm"  # pinned (get_nlp reloads per suite)

    def test_supporting_doc_covers_placeholder(self) -> None:
        verdict = _check(HEMINGWAY_QUERY, HEMINGWAY_SUPPORT, threshold=-1.0)
        self.assertTrue(verdict.consistent, verdict.evidence)

    def test_place_free_doc_flags_contradiction(self) -> None:
        verdict = _check(HEMINGWAY_QUERY, HEMINGWAY_NO_PLACE, threshold=-1.0)
        self.assertFalse(verdict.consistent, verdict.evidence)
        self.assertIn("[CONTRADICTION]", verdict.evidence)
        self.assertIn("?Where", verdict.evidence)

    def test_number_answered_consistent(self) -> None:
        verdict = _check(RACE_QUERY, RACE_SUPPORT, threshold=-1.0)
        self.assertTrue(verdict.consistent, verdict.evidence)

    def test_number_missing_flags_contradiction(self) -> None:
        verdict = _check(RACE_QUERY, RACE_NO_NUMBER, threshold=-1.0)
        self.assertFalse(verdict.consistent, verdict.evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
