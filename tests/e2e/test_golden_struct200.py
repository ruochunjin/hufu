"""Runner for the generated structured-document corpus (~200 docs)."""

from __future__ import annotations

import importlib.util
import json
import os
import time
import unittest
from pathlib import Path

from tests.e2e.golden_struct200 import CASES

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = importlib.util.find_spec("spacy") is not None

REPORT_PATH = Path(__file__).parent / "golden_struct200_report.json"


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and spacy")
class GoldenStruct200Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # en_core_web_trf: sm's NER misses uncommon toponyms (Tampere,
        # Curitiba, Graz...), dropping GPE typing and with it the typed
        # value-slot fires — measured 13/20 on bio-loc with sm vs 20/20
        # with trf.  The engine's own default is trf for the same reason.
        os.environ["HUFU_SPACY_MODEL"] = "en_core_web_trf"  # pinned (model-aware get_nlp reloads per suite)

    def test_corpus_shape(self) -> None:
        self.assertEqual(len(CASES), 186)
        contra = [c for c in CASES if c.expected]
        clean = [c for c in CASES if not c.expected]
        self.assertEqual(len(contra), 106)
        self.assertEqual(len(clean), 80)

    def test_star_engine_on_structured_corpus(self) -> None:
        from hufu.engine.star_graph import check_document_text

        report: list[dict] = []
        failures: list[str] = []
        total_seconds = 0.0
        for case in CASES:
            start = time.perf_counter()
            try:
                has_contra, _secs, evidence, conflicts, _chain = check_document_text(
                    case.text
                )
                error = None
            except Exception as exc:  # noqa: BLE001 - corpus runner records
                has_contra, evidence, conflicts, error = (
                    None, "", [], f"{type(exc).__name__}: {exc}",
                )
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
                }
            )
            if error is not None:
                failures.append(f"{case.id} raised {error}")
            elif has_contra != case.expected:
                failures.append(
                    f"{case.id} ({case.family}) expected={case.expected} got={has_contra} "
                    f"rules={[c.rule for c in (conflicts or [])][:4]}"
                )

        REPORT_PATH.write_text(
            json.dumps({"cases": report}, ensure_ascii=False, indent=2),
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
            f"\ngolden-struct200: {len(correct)}/{len(report)} correct "
            f"(P={precision:.3f} R={recall:.3f} F1={f1:.3f}; "
            f"tp={tp} fn={fn} fp={fp} tn={tn}), "
            f"{total_seconds:.0f}s total; by family: {summary}"
        )
        self.assertFalse(
            failures,
            f"{len(failures)} corpus failures:\n" + "\n".join(failures[:25]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
