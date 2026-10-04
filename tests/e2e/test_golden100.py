"""Golden-100 runner: 100 diverse cases through the reference engine.

Pinned cases (``expected`` is True/False) assert the verdict; observation
cases (``expected=None``) are reported but never fail.  All runs force the
structural path (``distance_threshold=-1``) so the HyperMatch check — not
the embedding gate — decides.  A JSON report lands in
``tests/e2e/golden100_report.json`` for the working record.

Run:
    HUFU_RUN_E2E=1 HF_HUB_OFFLINE=1 .venv-e2e/Scripts/python.exe \\
        -m pytest tests/e2e/test_golden100.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
import unittest
from pathlib import Path

from tests.e2e.golden100 import CASES

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = all(
    importlib.util.find_spec(name) for name in ("spacy", "sentence_transformers")
)

REPORT_PATH = Path(__file__).parent / "golden100_report.json"


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and models extra")
class Golden100Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("HUFU_SPACY_MODEL", "en_core_web_sm")

    def test_corpus_shape(self) -> None:
        self.assertEqual(len(CASES), 100)
        ids = [case.id for case in CASES]
        self.assertEqual(len(ids), len(set(ids)), "duplicate case ids")
        pinned = [c for c in CASES if c.expected is not None]
        self.assertGreaterEqual(len(pinned), 70, "corpus should pin most cases")

    def test_reference_engine_on_corpus(self) -> None:
        from hufu.engine.reference_check import reference_consistent_detection

        report: list[dict] = []
        failures: list[str] = []
        total_seconds = 0.0
        for case in CASES:
            start = time.perf_counter()
            try:
                has_contradiction, distance, evidence = reference_consistent_detection(
                    case.query,
                    case.document,
                    distance_threshold=-1.0,
                    query_is_question=case.is_question,
                )
                consistent = not has_contradiction
                error = None
            except Exception as exc:  # noqa: BLE001 - corpus runner records, not raises
                consistent, distance, evidence, error = None, None, "", f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            report.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "expected": case.expected,
                    "actual": consistent,
                    "seconds": round(elapsed, 2),
                    "distance": None if distance is None else round(distance, 4),
                    "error": error,
                    "evidence_head": (evidence or "").splitlines()[0][:140] if evidence else "",
                    "note": case.note,
                }
            )
            if error is not None:
                failures.append(f"{case.id} raised {error}")
            elif case.expected is not None and consistent != case.expected:
                failures.append(
                    f"{case.id} ({case.category}) expected={case.expected} got={consistent}: "
                    f"{(evidence or '').splitlines()[0][:120]}"
                )

        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pinned = [r for r in report if r["expected"] is not None]
        correct = [r for r in pinned if r["actual"] == r["expected"]]
        print(
            f"\ngolden100: {len(correct)}/{len(pinned)} pinned correct, "
            f"{sum(1 for r in report if r['error'])} errors, "
            f"{total_seconds:.0f}s total; report → {REPORT_PATH}"
        )
        self.assertFalse(
            failures,
            f"{len(failures)} golden failures:\n" + "\n".join(failures[:20]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
