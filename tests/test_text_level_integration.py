"""Optional end-to-end integration test for the text-level checker.

Skipped automatically unless the heavy optional dependencies are installed
(spaCy model, sentence-transformers, fastcoref) AND the Rust fixed-point
extension (``simulation-alg``) is importable.  Enable explicitly with:

    HUFU_RUN_INTEGRATION=1 python -m unittest tests.test_text_level_integration
"""

from __future__ import annotations

import importlib.util
import os
import unittest

_REQUIRED_MODULES = ["spacy", "sentence_transformers", "simulation"]
_AVAILABLE = all(importlib.util.find_spec(name) for name in _REQUIRED_MODULES)
_ENABLED = os.environ.get("HUFU_RUN_INTEGRATION") == "1"


@unittest.skipUnless(_AVAILABLE and _ENABLED, "requires models extra, Rust extension, and HUFU_RUN_INTEGRATION=1")
class TextLevelIntegrationTests(unittest.TestCase):
    def test_check_retrieval_marks_supporting_document_consistent(self) -> None:
        import hufu

        verdicts = hufu.check_retrieval(
            "Where was the author of 'The Old Man and the Sea' born?",
            [
                "Ernest Hemingway wrote The Old Man and the Sea. "
                "He was born in Oak Park, Illinois, in 1899.",
            ],
        )
        self.assertEqual(len(verdicts), 1)
        self.assertTrue(verdicts[0].consistent)
        self.assertTrue(verdicts[0].evidence)

    def test_pronoun_led_declarative_claim_is_not_coverage_blocked(self) -> None:
        """DocNLI finding (id 103): a declarative hypothesis whose subject is
        an unresolved pronoun ("It ...") must not fail coverage on the
        pronoun itself — pronouns carry no critical constraint of their own;
        resolution, when possible, happens through coreference clusters.
        """

        import hufu

        verdict = hufu.check_pair(
            "It had a population of over 3,000 people at the 2010 census.",
            "The 2010 census counted a population of 3,092 people in "
            "Washington Township, up from 2,870 in 2000.",
            engine="reference",
            query_is_question=False,
            distance_threshold=-1.0,
        )
        self.assertTrue(verdict.consistent, verdict.evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
