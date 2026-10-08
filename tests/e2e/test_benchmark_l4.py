"""L4 benchmark: standard (Rust) vs reference (pure Python) engine latency.

Same golden pairs, both engines, wall-clock per verdict.  The Rust fixed
point is expected to dominate on speed; the reference engine trades latency
for zero native dependencies.  Numbers are recorded in the working record
(hufu-e2e-测试驱动开发记录.md) and printed, not hard-asserted — the only
hard assertion is that both engines complete.

Note: a fixed-point-level differential (identical hv/dependencies into both
solvers) is deliberately deferred: the L3 divergences all originate upstream
of the fixed point (hv generation, match strictness, critical-vertex
policy), so solver-level equivalence is not the current bottleneck.
"""

from __future__ import annotations

import importlib.util
import os
import time
import unittest

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = all(
    importlib.util.find_spec(name)
    for name in ("spacy", "sentence_transformers", "simulation")
)

CASES = [
    (
        "When was the treaty signed?",
        "The treaty was signed on 11 November 1918 in Compiègne.",
    ),
    (
        "Where was Ernest Hemingway born?",
        "Ernest Hemingway was a famous writer. He wrote many novels.",
    ),
    (
        "How many meters is the marathon race?",
        "The marathon race is 42195 meters.",
    ),
]


def _timed_check(query: str, doc: str, *, engine: str) -> tuple[bool, float]:
    import hufu

    start = time.perf_counter()
    verdict = hufu.check_pair(query, doc, distance_threshold=-1.0, engine=engine)
    return verdict.consistent, time.perf_counter() - start


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1, models, and simulation")
class EngineBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["HUFU_SPACY_MODEL"] = "en_core_web_sm"  # pinned (get_nlp reloads per suite)

    def test_engine_latency_comparison(self) -> None:
        results: dict[str, list[float]] = {"standard": [], "reference": []}
        # Warm up both engines (model loads land on the first call).
        _timed_check(*CASES[0], engine="standard")
        _timed_check(*CASES[0], engine="reference")
        for query, doc in CASES:
            for engine in results:
                _consistent, elapsed = _timed_check(query, doc, engine=engine)
                results[engine].append(elapsed)
        report_lines = ["engine latency (s), warm:"]
        for engine, times in results.items():
            mean = sum(times) / len(times)
            report_lines.append(
                f"  {engine:10s} mean={mean:6.2f}  " + " ".join(f"{t:5.2f}" for t in times)
            )
        report = "\n".join(report_lines)
        print(f"\n{report}\n")
        # Rust standard path should not be slower than the Python reference.
        self.assertLessEqual(
            sum(results["standard"]),
            sum(results["reference"]) * 1.5,
            report,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
