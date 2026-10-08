"""Golden tests for the OHunt port (detectors, hinge classification, repairs).

Synthetic data with injected-noise ground truth, following OHunt's
evaluation protocol (clean clusters + a known noisy subset).  sklearn-backed
tests skip when sklearn is unavailable; detector tests are pure stdlib.
"""

from __future__ import annotations

import importlib.util
import random
import unittest

from hufu.scenarios.detectors import GapIsolationDetector, MADZScoreDetector

_SKLEARN = importlib.util.find_spec("sklearn") is not None


def _make_dataset(
    *, n_per_class: int = 60, n_noise: int = 5, n_mild: int = 4, seed: int = 7
) -> tuple[list[dict[str, float]], list[str], set[int], set[int]]:
    """Two Gaussian blobs + injected outliers of two known kinds.

    - damaging: points placed far from BOTH clusters keeping a cluster label
      (feature noise; the SVM margin disagrees -> hinge-class bad/ugly);
    - benign: points 3-sigma out on the CORRECT side of their cluster
      (genuinely outlying yet classified with full margin -> good).

    Returns (records, labels, damaging_indices, benign_indices).
    """

    rng = random.Random(seed)
    records: list[dict[str, float]] = []
    labels: list[str] = []
    for cls, (cx, cy) in enumerate([(0.0, 0.0), (5.0, 5.0)]):
        for _ in range(n_per_class):
            records.append(
                {"x": cx + rng.gauss(0, 0.6), "y": cy + rng.gauss(0, 0.6)}
            )
            labels.append(str(cls))
    damaging: set[int] = set()
    # Lone far-out points on the WRONG side of the decision boundary (gap
    # isolation only catches LONE outliers — a tight outlier cluster is
    # internally dense and invisible to it).  Deliberately mislabeled w.r.t.
    # position -> hinge-bad/ugly ground truth.
    damaging_spots = [
        ((13.0, 6.0), "0"), ((12.0, 9.0), "0"), ((14.0, 3.0), "0"),
        ((-9.0, -2.0), "1"), ((-7.0, -8.0), "1"),
    ]
    for (sx, sy), label in damaging_spots[: max(n_noise, 0)]:
        index = len(records)
        records.append({"x": sx + rng.gauss(0, 0.05), "y": sy + rng.gauss(0, 0.05)})
        labels.append(label)
        damaging.add(index)
    benign: set[int] = set()
    # Lone 4-5-sigma extrapolations ALONG each cluster's diagonal (correct
    # side, full margin): genuinely outlying yet harmless -> hinge-good.
    benign_spots = [
        ((-5.0, -5.0), "0"), ((-6.5, -6.0), "0"),
        ((10.0, 10.5), "1"), ((11.5, 11.0), "1"),
    ]
    for (sx, sy), label in benign_spots[: max(n_mild, 0)]:
        index = len(records)
        records.append({"x": sx + rng.gauss(0, 0.05), "y": sy + rng.gauss(0, 0.05)})
        labels.append(label)
        benign.add(index)
    return records, labels, damaging, benign


class DetectorTests(unittest.TestCase):
    def test_gap_isolation_flags_distant_point(self) -> None:
        records = [{"v": float(v)} for v in (10, 11, 12, 11, 10, 12, 11, 10, 100)]
        detector = GapIsolationDetector(theta=10.0)
        flags = detector.flag(records)
        self.assertTrue(flags[-1])
        self.assertFalse(any(flags[:-1]))

    def test_gap_isolation_dense_column_no_flags(self) -> None:
        records = [{"v": float(v)} for v in range(20)]
        detector = GapIsolationDetector(theta=10.0)
        self.assertFalse(any(detector.flag(records)))

    def test_mad_detector_flags_extreme(self) -> None:
        records = [{"v": float(v)} for v in (5, 6, 5, 6, 5, 6, 5, 50)]
        detector = MADZScoreDetector(z_threshold=3.0)
        flags = detector.flag(records)
        self.assertTrue(flags[-1])
        self.assertFalse(any(flags[:-1]))


@unittest.skipUnless(_SKLEARN, "requires sklearn")
class HingeClassificationTests(unittest.TestCase):
    def test_hinge_flags_damaging_as_bad_or_ugly(self) -> None:
        from hufu.scenarios import GapIsolationDetector, classify_outliers_hinge

        records, labels, damaging, benign = _make_dataset()
        reports = classify_outliers_hinge(
            records, labels, detector=GapIsolationDetector(theta=10.0)
        )
        categories = {r.index: r.category for r in reports}
        flagged = {i for i, c in categories.items() if c != "normal"}
        # The detector must catch most injected (far-out) noise...
        self.assertGreaterEqual(len(flagged & damaging), len(damaging) * 0.7)
        # ...and flagged damaging rows are mostly bad/ugly (margin-disputed),
        # while clean core records are essentially never flagged.
        core = set(range(2 * 60))
        self.assertGreaterEqual(
            sum(1 for i in flagged & damaging if categories[i] in {"bad", "ugly"})
            / max(len(flagged & damaging), 1),
            0.6,
        )
        self.assertLessEqual(len(flagged & core), len(core) * 0.1)

    def test_hinge_benign_outliers_are_good(self) -> None:
        # Mild 3-sigma outliers on the correct side carry full margin.
        from hufu.scenarios import GapIsolationDetector, classify_outliers_hinge

        records, labels, damaging, benign = _make_dataset(n_noise=0)
        reports = classify_outliers_hinge(
            records, labels, detector=GapIsolationDetector(theta=10.0)
        )
        categories = {r.index: r.category for r in reports}
        self.assertFalse([i for i in categories if categories[i] == "bad"])
        flagged_benign = [i for i in benign if categories[i] != "normal"]
        self.assertTrue(flagged_benign, "benign outliers should be flagged")
        self.assertTrue(
            all(categories[i] == "good" for i in flagged_benign),
            f"benign outliers misclassified: {[categories[i] for i in flagged_benign]}",
        )


@unittest.skipUnless(_SKLEARN, "requires sklearn")
class RepairTests(unittest.TestCase):
    def test_knn_label_repair_recovers_accuracy(self) -> None:
        from sklearn.linear_model import LogisticRegression

        from hufu.scenarios import repair_outliers

        records, labels, damaging, _benign = _make_dataset(n_noise=15)
        held_records, held_labels, _, _ = _make_dataset(
            n_per_class=40, n_noise=0, n_mild=0, seed=99
        )

        def accuracy(train_records, train_labels) -> float:
            model = LogisticRegression()
            model.fit(
                [[r["x"], r["y"]] for r in train_records],
                [str(label) for label in train_labels],
            )
            predictions = model.predict([[r["x"], r["y"]] for r in held_records])
            return sum(
                p == str(t) for p, t in zip(predictions, held_labels)
            ) / len(held_labels)

        baseline = accuracy(records, labels)
        outcome = repair_outliers(records, labels, sorted(damaging), method="knn_label")
        repaired = accuracy(list(outcome.records), list(outcome.labels))
        self.assertGreaterEqual(len(outcome.repaired_labels), 1)
        self.assertGreaterEqual(repaired, baseline)
        self.assertGreaterEqual(repaired, 0.95)

    def test_delete_and_median_paths(self) -> None:
        from hufu.scenarios import repair_outliers

        records, labels, damaging, _ = _make_dataset(n_noise=6)
        deleted = repair_outliers(records, labels, sorted(damaging), method="delete")
        self.assertEqual(len(deleted.records), len(records) - len(damaging))
        self.assertEqual(len(deleted.deleted), len(damaging))
        medianed = repair_outliers(records, labels, sorted(damaging), method="median")
        self.assertEqual(len(medianed.records), len(records))
        self.assertEqual(sorted(medianed.repaired_values), sorted(damaging))
        # Median repair keeps row count and replaces target features.
        for i in damaging:
            self.assertNotEqual(medianed.records[i], records[i])

    def test_empty_targets_noop(self) -> None:
        from hufu.scenarios import repair_outliers

        records, labels, _, _ = _make_dataset(n_noise=0, n_mild=0)
        outcome = repair_outliers(records, labels, [], method="delete")
        self.assertEqual(len(outcome.records), len(records))
        self.assertFalse(outcome.deleted)


@unittest.skipUnless(_SKLEARN, "requires sklearn")
class DiscoveryTests(unittest.TestCase):
    def test_discovery_beats_baseline_and_reports_grid(self) -> None:
        from sklearn.linear_model import LogisticRegression

        from hufu.scenarios import discover_outlier_pipeline

        records, labels, _damaging, _benign = _make_dataset()
        held_records, held_labels, _, _ = _make_dataset(
            n_per_class=40, n_noise=0, n_mild=0, seed=99
        )

        def accuracy(train_records, train_labels) -> float:
            model = LogisticRegression()
            model.fit(
                [[r["x"], r["y"]] for r in train_records],
                [str(label) for label in train_labels],
            )
            predictions = model.predict([[r["x"], r["y"]] for r in held_records])
            return sum(
                p == str(t) for p, t in zip(predictions, held_labels)
            ) / len(held_labels)

        result = discover_outlier_pipeline(records, labels, accuracy)
        self.assertGreaterEqual(result.best_accuracy, result.baseline_accuracy)
        self.assertEqual(len(result.leaderboard), 15)  # default grid size
        # The leaderboard is sorted best-first and the best heads it.
        self.assertEqual(result.leaderboard[0][0], result.best)
        accuracies = [score for _config, score in result.leaderboard]
        self.assertEqual(accuracies, sorted(accuracies, reverse=True))


if __name__ == "__main__":
    unittest.main()
