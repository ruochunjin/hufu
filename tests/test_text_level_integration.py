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


if __name__ == "__main__":
    unittest.main(verbosity=2)
