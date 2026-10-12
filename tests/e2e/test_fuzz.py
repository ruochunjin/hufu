"""Property / fuzz tests for HuFu's text engines.

Invariants under adversarial input:
- never crash (empty, whitespace, unicode soup, huge repetition)
- verdict is a bool; evidence is a string; decision chain present
- symmetry: conflict(A+B) == conflict(B+A) for the star engine
- determinism: same input twice -> same verdict
"""

from __future__ import annotations

import importlib.util
import os
import random
import unittest

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = importlib.util.find_spec("spacy") is not None


def _adversarial_texts(seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    texts = [
        "",
        "   ",
        "\n\n\t  \n",
        "a",
        "!@#$%^&*()_+-=[]{}|;':\",.<>/?`~",
        "The the the the the the the.",
        "字 符 混 排 mixed 内容 中文 English 日本語 한국어.",
        "The bridge was opened in 1883. " * 50,
        "emoji 😀🎉🚀 test with numbers 42.5 and dates 1883-05-24.",
        "Quotes 'nested \"double' quotes\" end.",
    ]
    words = ["alpha", "beta", "300", "Paris", "running", "not", "the", "of"]
    for _ in range(10):
        texts.append(" ".join(rng.choice(words) for _ in range(rng.randint(1, 40))))
    return texts


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and spacy")
class StarEngineFuzzTests(unittest.TestCase):
    def test_never_crashes_and_shape(self) -> None:
        from hufu.engine.star_graph import check_document_text

        for text in _adversarial_texts():
            has, secs, evidence, conflicts, chain = check_document_text(text)
            self.assertIsInstance(has, bool, repr(text[:30]))
            self.assertIsInstance(evidence, str)
            self.assertIsInstance(conflicts, list)
            self.assertGreaterEqual(secs, 0.0)

    def test_determinism(self) -> None:
        from hufu.engine.star_graph import check_document_text

        text = "The museum is open on Mondays. The museum is closed on Mondays."
        first = check_document_text(text)
        second = check_document_text(text)
        self.assertEqual(first[0], second[0])
        self.assertEqual([c.rule for c in first[3]], [c.rule for c in second[3]])

    def test_pair_order_symmetry(self) -> None:
        from hufu.engine.star_graph import check_document_text

        a = "The regiment lost 300 men at the ridge."
        b = "The regiment lost 3000 men at the ridge."
        self.assertEqual(check_document_text(a + " " + b)[0], check_document_text(b + " " + a)[0])


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and spacy")
class ReferenceEngineFuzzTests(unittest.TestCase):
    def test_never_crashes(self) -> None:
        from hufu.engine.reference_check import reference_consistent_detection

        query = "Where was the treaty signed?"
        for text in _adversarial_texts():
            has_contra, distance, evidence = reference_consistent_detection(query, text)
            self.assertIsInstance(has_contra, bool)
            self.assertIsInstance(evidence, str)
            if distance is not None:
                self.assertGreaterEqual(distance, -1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
