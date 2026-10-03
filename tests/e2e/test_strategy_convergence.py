"""L3-transition: hv strategy convergence over golden pairs.

The working record asks which h_v candidate source should be frozen as the
bridge default (typed / domain / denial).  ``run_hypermatch`` filters every
supplied pair through matchable-kind and type-compatibility gates, so the
strategies must be compared *after* that filter, not at generation time.

Finding (2026-10-03): after the gates, *denial* converges exactly to typed,
while *domain* collapses to a strict subset (its NER-asymmetry clause misses
every placeholder↔answer pair, and its noisy extra pairs are all filtered
out).  Typed is therefore the cheapest sufficient source — no NLI model, no
downstream loss.  These tests pin that result as a regression guard.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = all(
    importlib.util.find_spec(name) for name in ("spacy", "sentence_transformers")
)

CASES = [
    (
        "Where was the author of 'The Old Man and the Sea' born?",
        "Ernest Hemingway wrote The Old Man and the Sea. "
        "He was born in Oak Park, Illinois, in 1899.",
    ),
    ("How many meters is the marathon race?", "The marathon race is 42195 meters."),
    ("When was the treaty signed?", "The treaty was signed on 11 November 1918 in Compiègne."),
]


def _filtered(pairs, query_core, data_core) -> set[tuple[str, str]]:
    """Apply the reference core's own h_v validation gates."""

    from hufu.engine.d_match import default_type_compatible

    result = set()
    for q_id, d_id in pairs:
        q_vertex = query_core.vertex(q_id)
        d_vertex = data_core.vertex(d_id)
        if not q_vertex.matchable or not d_vertex.matchable:
            continue
        if default_type_compatible(q_vertex, d_vertex):
            result.add((q_id, d_id))
    return result


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and models extra")
class StrategyConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("HUFU_SPACY_MODEL", "en_core_web_sm")

    def _graphs(self, query_text: str, doc_text: str):
        from hufu.engine.bridge import bridge_hypergraph
        from hufu.engine.build_hypergraph import text_to_hypergraph

        query_local = text_to_hypergraph(query_text, is_query=True)
        data_local = text_to_hypergraph(doc_text)
        query_core = bridge_hypergraph(query_local, side="query").hypergraph
        data_core = bridge_hypergraph(data_local, side="data").hypergraph
        return query_local, data_local, query_core, data_core

    def test_filtered_domain_is_subset_of_typed(self) -> None:
        from hufu.engine.bridge import bridge_allowed_pairs

        for query_text, doc_text in CASES:
            with self.subTest(query=query_text):
                ql, dl, qc, dc = self._graphs(query_text, doc_text)
                typed = bridge_allowed_pairs(ql, dl, strategy="typed")
                domain = bridge_allowed_pairs(ql, dl, strategy="domain")
                self.assertLessEqual(_filtered(domain, qc, dc), typed)

    def test_filtered_domain_misses_placeholder_pairs(self) -> None:
        # Domain's NER-asymmetry clause drops every ?placeholder ↔ answer
        # pair on these golden cases — the reason it can only be a subset.
        from hufu.engine.bridge import bridge_allowed_pairs

        ql, dl, qc, dc = self._graphs(*CASES[0])
        typed = bridge_allowed_pairs(ql, dl, strategy="typed")
        domain = bridge_allowed_pairs(ql, dl, strategy="domain")
        self.assertTrue(typed)
        self.assertFalse(_filtered(domain, qc, dc))

    def test_typed_equals_filtered_denial(self) -> None:
        from hufu.engine.bridge import bridge_allowed_pairs

        for query_text, doc_text in CASES:
            with self.subTest(query=query_text):
                ql, dl, qc, dc = self._graphs(query_text, doc_text)
                typed = bridge_allowed_pairs(ql, dl, strategy="typed")
                denial = bridge_allowed_pairs(ql, dl, strategy="denial")
                self.assertEqual(typed, _filtered(denial, qc, dc))

    def test_typed_pairs_are_nonempty_on_answerable_cases(self) -> None:
        from hufu.engine.bridge import bridge_allowed_pairs

        for query_text, doc_text in CASES:
            with self.subTest(query=query_text):
                ql, dl, _qc, _dc = self._graphs(query_text, doc_text)
                self.assertTrue(
                    bridge_allowed_pairs(ql, dl, strategy="typed"),
                    f"no typed pairs for {query_text!r}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
