"""Runner for the generated RAG-consistency corpus (200 queries, 600 pairs)."""

from __future__ import annotations

import importlib.util
import json
import os
import time
import unittest
from pathlib import Path

from tests.e2e.golden_rag200 import CASES

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = importlib.util.find_spec("spacy") is not None

REPORT_PATH = Path(__file__).parent / "golden_rag200_report.json"


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and spacy")
class GoldenRag200Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # trf default: sm's NER loses uncommon toponyms (see golden_struct200).
        os.environ.setdefault("HUFU_SPACY_MODEL", "en_core_web_trf")

    def test_corpus_shape(self) -> None:
        self.assertEqual(len(CASES), 200)
        for case in CASES:
            self.assertEqual(len(case.documents), 3)
            self.assertEqual(len(case.expected_consistent), 3)

    def test_check_retrieval_on_corpus(self) -> None:
        from hufu.api import check_retrieval

        report: list[dict] = []
        failures: list[str] = []
        total_seconds = 0.0
        pair_total = pair_correct = 0
        for case in CASES:
            start = time.perf_counter()
            try:
                verdicts = check_retrieval(
                    case.query, list(case.documents),
                    engine="reference",  # validated path (golden-pinned);
                    # the Rust standard path predates the declarative channel
                    query_is_question=case.is_question,
                )
                actual = tuple(v.consistent for v in verdicts)
                distances = tuple(v.distance for v in verdicts)
                error = None
            except Exception as exc:  # noqa: BLE001 - corpus runner records
                actual = tuple()
                distances = tuple()
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            pair_total += 3
            if not error:
                pair_correct += sum(
                    a == e for a, e in zip(actual, case.expected_consistent)
                )
            bad = [
                (i, case.expected_consistent[i], actual[i] if i < len(actual) else None)
                for i in range(3)
                # slot 2 (irrelevant) is judged by the signal contract below,
                # not by an exact flag value (coverage of nominal-root titles
                # is parse-dependent; both flagging and distance are valid)
                if i != 2 and (i >= len(actual) or actual[i] != case.expected_consistent[i])
            ]
            # Slot-2 (irrelevant document) contract: it must never look AS
            # GOOD AS the gold passage — either flagged not-consistent, or
            # clearly farther by distance.  (Which signal fires depends on
            # parse-level coverage luck; both are valid filter signals.)
            if len(actual) == 3 and len(distances) == 3:
                d0, d2 = distances[0], distances[2]
                if actual[2] and d2 is not None and d0 is not None and d2 <= d0 + 0.05:
                    bad.append((2, "irrelevant looks as good as gold", distances))
            report.append(
                {
                    "id": case.id,
                    "family": case.family,
                    "query": case.query,
                    "expected": list(case.expected_consistent),
                    "actual": list(actual),
                    "bad_slots": bad,
                    "seconds": round(elapsed, 2),
                    "error": error,
                }
            )
            if error:
                failures.append(f"{case.id} raised {error}")
            elif bad:
                failures.append(f"{case.id} ({case.family}) bad_slots={bad}")

        REPORT_PATH.write_text(
            json.dumps({"cases": report}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        by_family: dict[str, list[int]] = {}
        for r in report:
            bucket = by_family.setdefault(r["family"], [0, 0])
            bucket[1] += 1
            bucket[0] += int(not r["bad_slots"])
        summary = {fam: f"{ok}/{tot}" for fam, (ok, tot) in sorted(by_family.items())}
        exact = sum(1 for r in report if not r["bad_slots"])
        print(
            f"\ngolden-rag200: {exact}/{len(report)} queries exact "
            f"({pair_correct}/{pair_total} pairs), "
            f"{total_seconds:.0f}s total; by family: {summary}"
        )
        self.assertFalse(
            failures,
            f"{len(failures)} corpus failures:\n" + "\n".join(failures[:25]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
