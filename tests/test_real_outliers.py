"""Real-data outlier suites: sklearn's bundled datasets + noise injection.

Iris / wine / breast-cancer ship with sklearn (no download), giving the
OHunt port a real-data workout under the paper's noise-injection protocol:
lone far-out points with wrong-side labels are the damaging ground truth;
the pipeline must flag most of them and repair must not degrade accuracy.

sklearn-gated; deterministic.
"""

from __future__ import annotations

import importlib.util
import random
import unittest

_SKLEARN = importlib.util.find_spec("sklearn") is not None

_REAL_DATASETS = ["iris", "wine", "breast_cancer"]


def _load(name: str):
    from sklearn import datasets

    loader = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
    }[name]
    bunch = loader()
    keys = [f"f{i}" for i in range(bunch.data.shape[1])]
    records = [
        {key: float(value) for key, value in zip(keys, row)} for row in bunch.data
    ]
    labels = [str(y) for y in bunch.target]
    return records, labels


def _inject_noise(
    records: list[dict[str, float]],
    labels: list[str],
    *,
    rate: float = 0.05,
    seed: int = 5,
) -> tuple[list[dict[str, float]], list[str], set[int]]:
    """Add lone far-out mislabeled points (damaging ground truth)."""

    rng = random.Random(seed)
    keys = sorted(records[0].keys())
    columns = {key: [float(r[key]) for r in records] for key in keys}
    medians = {key: sorted(col)[len(col) // 2] for key, col in columns.items()}
    spans = {
        key: max(max(col) - min(col), 1e-9) for key, col in columns.items()
    }
    classes = sorted(set(labels))
    n_noise = max(3, int(len(records) * rate))
    out_records = list(records)
    out_labels = list(labels)
    noisy: set[int] = set()
    for k in range(n_noise):
        # Push each feature far below the column minimum (lone in most dims);
        # label from the class that is (cyclically) wrong.
        record = {
            key: medians[key] - (1.5 + 0.3 * k) * spans[key] for key in keys
        }
        out_records.append(record)
        out_labels.append(classes[(k + 1) % len(classes)])
        noisy.add(len(out_records) - 1)
    return out_records, out_labels, noisy


@unittest.skipUnless(_SKLEARN, "requires sklearn")
class RealDataOutlierTests(unittest.TestCase):
    def test_real_datasets_pipeline(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        from hufu.scenarios import (
            GapIsolationDetector,
            classify_outliers_hinge,
            repair_outliers,
        )

        summaries: list[str] = []
        failures: list[str] = []
        for name in _REAL_DATASETS:
            base_records, base_labels = _load(name)
            keys = sorted(base_records[0].keys())
            records, labels, noisy = _inject_noise(base_records, base_labels)

            scaler = StandardScaler()
            matrix = scaler.fit_transform(
                [[r[k] for k in keys] for r in records]
            )
            scaled = [
                {k: float(v) for k, v in zip(keys, row)} for row in matrix
            ]

            reports = classify_outliers_hinge(
                scaled, labels, detector=GapIsolationDetector(theta=10.0)
            )
            flagged = {r.index for r in reports if r.category != "normal"}
            recall = len(flagged & noisy) / len(noisy)

            def accuracy(tr_x, tr_y) -> float:
                model = LogisticRegression(max_iter=2000)
                model.fit([[r[k] for k in keys] for r in tr_x], tr_y)
                clean_matrix = scaler.transform(
                    [[r[k] for k in keys] for r in base_records]
                )
                predictions = model.predict(clean_matrix)
                return sum(
                    p == t for p, t in zip(predictions, base_labels)
                ) / len(base_labels)

            baseline = accuracy(scaled, labels)
            targets = [r.index for r in reports if r.category in {"ugly", "bad"}]
            outcome = repair_outliers(
                scaled, labels, targets, method="knn_label"
            )
            repaired = accuracy(list(outcome.records), list(outcome.labels))
            summaries.append(
                f"{name}: n={len(records)} noise={len(noisy)} "
                f"detection_recall={recall:.2f} acc {baseline:.3f}→{repaired:.3f}"
            )
            if recall < 0.6:
                failures.append(f"{name}: detection recall {recall:.2f} < 0.6")
            if repaired < baseline - 0.02:
                failures.append(
                    f"{name}: repair degraded accuracy {baseline:.3f}→{repaired:.3f}"
                )
        print("\nreal-data outliers:")
        for line in summaries:
            print("  " + line)
        self.assertFalse(failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main(verbosity=2)
