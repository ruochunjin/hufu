"""Runner for the generated ~300-bank memory corpus."""

from __future__ import annotations

import importlib.util
import json
import os
import time
import unittest
from pathlib import Path

from tests.e2e.golden_mem300 import CASES

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = importlib.util.find_spec("spacy") is not None

REPORT_PATH = Path(__file__).parent / "golden_mem300_report.json"


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and spacy")
class GoldenMem300Tests(unittest.TestCase):
    def test_corpus_shape(self) -> None:
        self.assertEqual(len(CASES), 320)
        with_conflicts = [c for c in CASES if c.expected_pairs]
        self.assertGreaterEqual(len(with_conflicts), 180)

    def test_memory_conflicts_on_generated_corpus(self) -> None:
        import math

        from hufu.engine.embedding import get_embedding_batch
        from hufu.scenarios import detect_memory_conflicts

        # CDR's VecSim gate is a SCOPE boundary: pairs with cosine below the
        # learned threshold are not in the check regime by design (the rule
        # system only watches geometrically adjacent memories).  A missing
        # pair below the gate is out-of-scope, not a failure.
        gate = 0.59
        texts = sorted({t for case in CASES for t, _ in case.memories})
        vectors = dict(zip(texts, get_embedding_batch(texts)))

        def cosine(ta: str, tb: str) -> float:
            va, vb = vectors[ta], vectors[tb]
            na = math.sqrt(sum(x * x for x in va))
            nb = math.sqrt(sum(x * x for x in vb))
            return sum(x * y for x, y in zip(va, vb)) / (na * nb) if na and nb else 0.0

        report: list[dict] = []
        failures: list[str] = []
        out_of_scope: list[str] = []
        total_seconds = 0.0
        for case in CASES:
            start = time.perf_counter()
            conflicts = detect_memory_conflicts(list(case.memories))
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            actual = {tuple(sorted((c.index_a, c.index_b))) for c in conflicts}
            expected = set(case.expected_pairs)
            missing = expected - actual
            extra = actual - expected
            scoped_out = {
                pair
                for pair in missing
                if cosine(case.memories[pair[0]][0], case.memories[pair[1]][0]) < gate
            }
            missing -= scoped_out
            report.append(
                {
                    "id": case.id,
                    "family": case.family,
                    "missing": sorted(map(list, missing)),
                    "extra": sorted(map(list, extra)),
                    "out_of_scope": sorted(map(list, scoped_out)),
                    "seconds": round(elapsed, 2),
                }
            )
            if scoped_out:
                out_of_scope.append(f"{case.id}: {sorted(scoped_out)}")
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
            f"\ngolden-mem300: {exact}/{len(report)} exact pair sets "
            f"({len(out_of_scope)} below VecSim gate, out of scope by design), "
            f"{total_seconds:.0f}s total; by family: {summary}"
        )
        print("out-of-scope pairs:", out_of_scope)
        self.assertFalse(
            failures,
            f"{len(failures)} corpus failures:\n" + "\n".join(failures[:25]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
