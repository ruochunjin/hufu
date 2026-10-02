"""Dependency-free tests for document contradiction and outlier scenarios."""

from __future__ import annotations

import unittest

from hufu.rules import RuleEngine
from hufu.scenarios.documents import detect_contradictions, split_statements
from hufu.scenarios.outliers import (
    apply_repairs,
    classify_outliers,
    outlier_scores,
)


class FakeServices:
    def embed(self, texts):
        return [(1.0, 0.0) for _ in texts]

    def nli_label(self, a, b):
        return "neutral"

    def nli_score(self, a, b, label):
        # Fake NLI: strong contradiction when one says increase, other decrease.
        if label == "contradiction":
            pair = {a, b}
            if any("increased" in t for t in pair) and any("decreased" in t for t in pair):
                return 0.95
        return 0.1


def make_engine():
    return RuleEngine(services=FakeServices())


class DocumentScenarioTests(unittest.TestCase):
    def test_nli_contradiction_across_documents_with_provenance(self) -> None:
        docs = [
            "The company did well. Revenue increased by 20% in 2025.",
            "The firm struggled. Revenue decreased by 20% in 2025.",
        ]
        findings = detect_contradictions(docs, engine=make_engine(), scope="inter")
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.scope, "inter")
        self.assertIn("contradiction:nli", finding.labels)
        self.assertEqual(finding.statement_a.doc_id, 0)
        self.assertEqual(finding.statement_b.doc_id, 1)
        self.assertIn("Revenue increased", finding.statement_a.text)

    def test_intra_scope_excludes_cross_document_pairs(self) -> None:
        docs = [
            "Revenue increased by 20%. Revenue decreased by 20%.",
            "Nothing to see here.",
        ]
        inter = detect_contradictions(docs, engine=make_engine(), scope="inter")
        intra = detect_contradictions(docs, engine=make_engine(), scope="intra")
        self.assertEqual(inter, [])
        self.assertEqual(len(intra), 1)
        self.assertEqual(intra[0].scope, "intra")

    def test_attribute_rules_with_structured_statements(self) -> None:
        docs = ["Revenue was 100 million.", "Revenue was 250 million."]
        attrs = [
            [{"entity": "revenue", "value": 100}],
            [{"entity": "revenue", "value": 250}],
        ]
        findings = detect_contradictions(docs, attrs=attrs, engine=make_engine())
        self.assertEqual(len(findings), 1)
        self.assertIn("contradiction:numeric", findings[0].labels)

    def test_split_statements_tracks_positions(self) -> None:
        statements = split_statements(["One. Two!", "Three?"])
        self.assertEqual(
            [(s.doc_id, s.sent_idx) for s in statements],
            [(0, 0), (0, 1), (1, 0)],
        )


class OutlierScenarioTests(unittest.TestCase):
    RECORDS = [
        {"x": 10.0, "y": 1.0},
        {"x": 11.0, "y": 0.0},
        {"x": 9.5, "y": 1.0},
        {"x": 10.5, "y": 0.0},
        {"x": 10.2, "y": 1.0},
        {"x": 9.8, "y": 0.0},
        {"x": 999.0, "y": 1.0},  # index 6: huge outlier
    ]
    LABELS = [1, 0, 1, 0, 1, 0, 1]

    def test_outlier_scores_flag_extreme_record(self) -> None:
        scores = outlier_scores(self.RECORDS)
        self.assertEqual(max(scores), scores[6])
        self.assertGreater(scores[6], 3.0)
        self.assertTrue(all(s < 3.0 for i, s in enumerate(scores) if i != 6))

    def test_classify_ugly_when_repair_helps(self) -> None:
        def evaluator(records, labels):
            # Fake model: accuracy collapses while the extreme x is present.
            xs = [r["x"] for r in records]
            return 0.5 if max(xs) > 100 else 0.9

        reports = classify_outliers(
            self.RECORDS, self.LABELS, evaluator, gain_threshold=0.05
        )
        self.assertEqual(reports[6].category, "ugly")
        self.assertAlmostEqual(reports[6].accuracy_delta, 0.4)
        self.assertIsNotNone(reports[6].repaired)
        self.assertLess(reports[6].repaired["x"], 100.0)
        self.assertTrue(all(r.category == "normal" for i, r in enumerate(reports) if i != 6))
        repaired = apply_repairs(self.RECORDS, reports)
        self.assertLess(max(r["x"] for r in repaired), 100.0)

    def test_classify_bad_when_repair_is_irrelevant(self) -> None:
        def evaluator(records, labels):
            return 0.8  # model ignores the outlier entirely

        reports = classify_outliers(
            self.RECORDS, self.LABELS, evaluator, gain_threshold=0.05
        )
        self.assertEqual(reports[6].category, "bad")
        self.assertIsNone(reports[6].repaired)

    def test_classify_good_when_repair_hurts(self) -> None:
        def evaluator(records, labels):
            # Fake model that *needs* the extreme record (rare region).
            xs = [r["x"] for r in records]
            return 0.95 if max(xs) > 100 else 0.6

        reports = classify_outliers(
            self.RECORDS, self.LABELS, evaluator, gain_threshold=0.05
        )
        self.assertEqual(reports[6].category, "good")
        self.assertIsNone(reports[6].repaired)

    def test_length_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_outliers(self.RECORDS, [1, 0], lambda r, y: 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
