"""CDRs prototype fixture: 85-entry robot knowledge bank through the memory
conflict scenario (robustness smoke on real prototype data).

The prototype's conflict ground truth lives in the GOAL logic forms
(`On_Drinks_Table1` vs `~On_Drinks_Table1`), not in the instruction texts,
so this suite does NOT assert pair-level accuracy.  It asserts the tool
runs a real 85-entry bank end-to-end, produces explainable conflicts, and
stays deterministic.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

_FIXTURE = Path(
    r"D:\tools\LLM-data-management\CDRs-main\robot_retr_result\easy_instr_goal.jsonl"
)


def _load_bank() -> list[tuple[str, dict]]:
    bank: dict[str, tuple[str, str]] = {}
    with open(_FIXTURE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for ctx in row["ctxs"]:
                bank[ctx["id"]] = (ctx["title"].strip(), ctx["text"].strip())
    return [(title, {}) for _id, (title, _goal) in sorted(bank.items(), key=lambda kv: int(kv[0]))]


@unittest.skipUnless(
    os.environ.get("HUFU_RUN_E2E") == "1" and _FIXTURE.exists(),
    "requires HUFU_RUN_E2E=1 and the local CDRs prototype fixture",
)
class CdrsFixtureTests(unittest.TestCase):
    def test_robot_bank_smoke(self) -> None:
        from hufu.scenarios import detect_memory_conflicts

        bank = _load_bank()
        self.assertEqual(len(bank), 85)
        conflicts = detect_memory_conflicts(bank)
        print(
            f"\ncdrs-fixture: {len(bank)} memories -> {len(conflicts)} "
            "conflict pairs reported (unstructured NLI fallback path)"
        )
        # Every reported conflict must carry a rule explanation.
        for c in conflicts:
            self.assertTrue(c.explanations)
        # Determinism: same run twice, same pair set.
        again = detect_memory_conflicts(bank)
        self.assertEqual(
            {(c.index_a, c.index_b) for c in conflicts},
            {(c.index_a, c.index_b) for c in again},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
