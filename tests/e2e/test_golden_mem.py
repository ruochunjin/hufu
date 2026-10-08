"""Runner for the golden memory-bank corpus (CDR scenario, v0.5).

Rules-only by default: supplies explicit attributes, so vec_sim uses the
embedding service and the NLI fallback rule needs the NLI model — both are
available in the e2e environment.  Runs every bank through
``detect_memory_conflicts`` and compares the conflict pair sets exactly.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
import unittest
from pathlib import Path

from tests.e2e.golden_mem_corpus import CASES

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = importlib.util.find_spec("spacy") is not None

REPORT_PATH = Path(__file__).parent / "golden_mem_report.json"


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and spacy")
class GoldenMemTests(unittest.TestCase):
    def test_corpus_shape(self) -> None:
        self.assertGreaterEqual(len(CASES), 25)
        with_conflicts = [c for c in CASES if c.expected_pairs]
        without = [c for c in CASES if not c.expected_pairs]
        self.assertGreaterEqual(len(with_conflicts), 15)
        self.assertGreaterEqual(len(without), 8)

    def test_memory_conflicts_on_corpus(self) -> None:
        from hufu.scenarios import detect_memory_conflicts

        report: list[dict] = []
        failures: list[str] = []
        total_seconds = 0.0
        for case in CASES:
            start = time.perf_counter()
            conflicts = detect_memory_conflicts(list(case.memories))
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            actual = {
                tuple(sorted((c.index_a, c.index_b))) for c in conflicts
            }
            expected = set(case.expected_pairs)
            missing = expected - actual
            extra = actual - expected
            report.append(
                {
                    "id": case.id,
                    "family": case.family,
                    "expected": sorted(map(list, expected)),
                    "actual": sorted(map(list, actual)),
                    "missing": sorted(map(list, missing)),
                    "extra": sorted(map(list, extra)),
                    "seconds": round(elapsed, 2),
                    "note": case.note,
                }
            )
            if missing or extra:
                failures.append(
                    f"{case.id} ({case.family}) missing={sorted(missing)} extra={sorted(extra)}"
                )

        REPORT_PATH.write_text(
            json.dumps({"cases": report}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        exact = sum(
            1 for r in report if not r["missing"] and not r["extra"]
        )
        by_family: dict[str, list[int]] = {}
        for r in report:
            bucket = by_family.setdefault(r["family"], [0, 0])
            bucket[1] += 1
            bucket[0] += int(not r["missing"] and not r["extra"])
        summary = {fam: f"{ok}/{tot}" for fam, (ok, tot) in sorted(by_family.items())}
        print(
            f"\ngolden-mem: {exact}/{len(report)} exact pair sets "
            f"({total_seconds:.0f}s total); by family: {summary}"
        )
        self.assertFalse(
            failures,
            f"{len(failures)} corpus failures:\n" + "\n".join(failures[:25]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
