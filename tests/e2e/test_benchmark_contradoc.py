"""Benchmark smoke test: ContraDoc contradiction pairs through the reference engine.

Dataset: ContraDoc (ddhruvkr/CONTRADOC, Apache-2.0), committed at
``benchmarks/data/ContraDoc.json``.  Each positive document with
``ref sentences`` yields a (ref, evidence) pair where the injected evidence
sentence contradicts the reference sentence — a golden contradiction case
for ``check_pair``.

Findings pinned here (2026-10-03 probe):
- Some ``ref sentences`` are truncated to a few characters ("A"); pairs whose
  reference has < 6 words are skipped — they carry no structural signal.
- Clearly contradictory pairs (e.g. "mourn" vs "celebrates" a death) are
  flagged by the pure-Python reference engine.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_MODELS_OK = all(
    importlib.util.find_spec(name) for name in ("spacy", "sentence_transformers")
)

_MIN_REF_WORDS = 6
_SAMPLE_SIZE = 4
_MIN_FLAGGED = 3


def _contradiction_pairs(limit: int) -> list[tuple[str, str]]:
    from nli_loaders import load_contradoc

    pairs: list[tuple[str, str]] = []
    for example in load_contradoc():
        if example.label != "contradiction":
            continue
        refs = [
            ref
            for ref in example.meta.get("ref_sentences", [])
            if len(ref.split()) >= _MIN_REF_WORDS
        ]
        if not refs or not example.hypothesis.strip():
            continue
        pairs.append((refs[0], example.hypothesis))
        if len(pairs) >= limit:
            break
    return pairs


@unittest.skipUnless(_ENABLED and _MODELS_OK, "requires HUFU_RUN_E2E=1 and models extra")
class ContraDocSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["HUFU_SPACY_MODEL"] = "en_core_web_sm"  # pinned (get_nlp reloads per suite)

    def test_reference_engine_flags_contradiction_pairs(self) -> None:
        from hufu.engine.reference_check import reference_consistent_detection

        pairs = _contradiction_pairs(_SAMPLE_SIZE)
        self.assertGreaterEqual(
            len(pairs), _MIN_FLAGGED, "not enough usable ContraDoc pairs"
        )
        flagged = 0
        verdicts = []
        for ref, evidence in pairs:
            # ContraDoc refs/evidence are declarative claims: build without
            # query placeholders (F8) and engage the NLI polarity gate, which
            # is what catches "mourn" vs "celebrate" style conflicts.
            has_contradiction, distance, evidence_text = reference_consistent_detection(
                ref, evidence, distance_threshold=-1.0, query_is_question=False
            )
            verdicts.append((ref, evidence, has_contradiction, evidence_text))
            flagged += int(has_contradiction)
        details = "\n".join(
            f"  contra={flag} ref={ref[:60]!r} evi={evi[:60]!r}"
            for ref, evi, flag, _ in verdicts
        )
        self.assertGreaterEqual(
            flagged,
            _MIN_FLAGGED,
            f"expected ≥{_MIN_FLAGGED}/{len(verdicts)} flagged contradictions:\n{details}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
