"""Doc300 corpus runner: 300 documents through the star-graph engine.

Same protocol as the v100 runner; reports to
``tests/e2e/golden_doc300_report.json``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
import unittest
from pathlib import Path

from tests.e2e.golden_doc300 import CASES

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = importlib.util.find_spec("spacy") is not None

REPORT_PATH = Path(__file__).parent / "golden_doc300_report.json"


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and spacy")
class GoldenDoc300Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("HUFU_SPACY_MODEL", "en_core_web_sm")

    def test_corpus_shape(self) -> None:
        self.assertEqual(len(CASES), 300)
        contra = [c for c in CASES if c.expected]
        clean = [c for c in CASES if not c.expected]
        self.assertGreaterEqual(len(contra), 130)
        self.assertGreaterEqual(len(clean), 160)

    def test_star_engine_on_corpus(self) -> None:
        from hufu.engine.star_graph import check_document_text
        from hufu.llm.***REMOVED*** import ***REMOVED***

        backend = os.environ.get("***REMOVED***", "local")
        ***REMOVED*** = os.environ.get("***REMOVED***", "0") == "1"
        if ***REMOVED*** and backend == "***REMOVED***" and not ***REMOVED***():
            ***REMOVED*** = False

        report: list[dict] = []
        failures: list[str] = []
        total_seconds = 0.0
        for case in CASES:
            start = time.perf_counter()
            try:
                has_contra, _secs, evidence, conflicts, _chain = check_document_text(
                    case.text, ***REMOVED***=***REMOVED***
                )
                error = None
            except Exception as exc:  # noqa: BLE001 - corpus runner records, not raises
                has_contra, evidence, conflicts, error = None, "", [], f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            report.append(
                {
                    "id": case.id,
                    "family": case.family,
                    "expected": case.expected,
                    "actual": has_contra,
                    "seconds": round(elapsed, 2),
                    "error": error,
                    "evidence_head": (evidence or "").splitlines()[0][:140] if evidence else "",
                    "rules": [c.rule for c in (conflicts or [])][:4],
                    "note": case.note,
                }
            )
            if error is not None:
                failures.append(f"{case.id} raised {error}")
            elif has_contra != case.expected:
                failures.append(
                    f"{case.id} ({case.family}) expected={case.expected} got={has_contra}"
                )

        REPORT_PATH.write_text(
            json.dumps(
                {"llm_enabled": ***REMOVED***, "backend": backend, "cases": report},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        correct = [r for r in report if r["expected"] == r["actual"]]
        by_family: dict[str, list[int]] = {}
        for r in report:
            bucket = by_family.setdefault(r["family"], [0, 0])
            bucket[1] += 1
            bucket[0] += int(r["expected"] == r["actual"])
        summary = {fam: f"{ok}/{tot}" for fam, (ok, tot) in sorted(by_family.items())}
        tp = sum(1 for r in report if r["expected"] and r["actual"])
        fn = sum(1 for r in report if r["expected"] and not r["actual"])
        fp = sum(1 for r in report if not r["expected"] and r["actual"])
        tn = sum(1 for r in report if not r["expected"] and not r["actual"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(
            f"\ngolden-doc300: {len(correct)}/{len(report)} correct "
            f"(P={precision:.3f} R={recall:.3f} F1={f1:.3f}), "
            f"{total_seconds:.0f}s total; by family: {summary}"
        )
        self.assertFalse(
            failures,
            f"{len(failures)} corpus failures:\n" + "\n".join(failures[:25]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
