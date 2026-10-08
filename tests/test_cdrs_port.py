"""Unit tests for the CDRs-ported components (threshold learning, witness
conflict inference).  Pure stdlib paths — no models, no e2e gate."""

from __future__ import annotations

import unittest

from hufu.rules import learn_threshold_youden
from hufu.scenarios import infer_conflicts_from_judgments, trim_evidence


class ThresholdLearningTests(unittest.TestCase):
    def test_separable_scores_perfect_split(self) -> None:
        report = learn_threshold_youden(
            positive_scores=[1.4, 1.5, 1.6, 1.35],
            negative_scores=[0.9, 1.0, 1.1, 0.8],
        )
        self.assertEqual(report.j_statistic, 1.0)
        self.assertEqual(report.tpr, 1.0)
        self.assertEqual(report.fpr, 0.0)
        # The learned threshold separates the two clouds.
        self.assertGreater(report.threshold, 1.1)
        self.assertLess(report.threshold, 1.35)

    def test_overlapping_scores_youden_point(self) -> None:
        # Hand-computed: scores 1,2 (neg) and 2,3 (pos).
        # Candidates: midpoints 1.5 (tpr=1, fpr=0.5, J=0.5),
        # 2.0-exact not a midpoint; 2.5 (tpr=0.5, fpr=0, J=0.5).
        # Tie -> larger threshold wins (precision-leaning).
        report = learn_threshold_youden([2.0, 3.0], [1.0, 2.0])
        self.assertAlmostEqual(report.j_statistic, 0.5)
        self.assertAlmostEqual(report.threshold, 2.5)

    def test_empty_side_raises(self) -> None:
        with self.assertRaises(ValueError):
            learn_threshold_youden([], [1.0])
        with self.assertRaises(ValueError):
            learn_threshold_youden([1.0], [])


class WitnessInferenceTests(unittest.TestCase):
    def test_cartesian_conflict_inference(self) -> None:
        # The Trimming Theorem: divergent judgments against the same witness
        # imply inter-item conflicts.
        pairs = infer_conflicts_from_judgments({0: True, 1: False, 2: True, 3: False})
        self.assertEqual(pairs, [(0, 1), (0, 3), (2, 1), (2, 3)])

    def test_all_consistent_no_conflicts(self) -> None:
        self.assertEqual(infer_conflicts_from_judgments({0: True, 1: True}), [])

    def test_inferred_conflicts_drive_trimming(self) -> None:
        # Retrieval-ranked items 0..3; judgments say 1,3 dissent.
        # keep_first trims the lower-ranked side of each inferred conflict.
        judgments = {0: True, 1: False, 2: True, 3: False}
        pairs = infer_conflicts_from_judgments(judgments)
        result = trim_evidence(["a", "b", "c", "d"], pairs, strategy="keep_first")
        self.assertEqual(result.kept, (0, 2))
        self.assertEqual(sorted(d.index for d in result.dropped), [1, 3])


if __name__ == "__main__":
    unittest.main()
