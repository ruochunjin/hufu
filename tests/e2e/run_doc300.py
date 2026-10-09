"""Doc300 corpus driver with progress output (mirrors test_golden_doc300.py).

Usage:
    python run_doc300.py <report_path>

Env: HUFU_RUN_E2E=1 HF_HUB_OFFLINE=1.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.e2e.golden_doc300 import CASES  # noqa: E402


def main() -> None:
    report_path = Path(sys.argv[1])
    limit = 0
    for arg in sys.argv[2:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])
    os.environ["HUFU_SPACY_MODEL"] = "en_core_web_sm"  # pinned (get_nlp reloads per suite)

    from hufu.engine.star_graph import check_document_text

    cases = list(CASES[:limit]) if limit else list(CASES)
    report: list[dict] = []
    total_seconds = 0.0
    for index, case in enumerate(cases, 1):
        start = time.perf_counter()
        try:
            has_contra, _secs, evidence, conflicts, _chain = check_document_text(
                case.text,
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - corpus runner records, not raises
            has_contra, evidence, conflicts, error = None, "", [], f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        ok = "ok" if has_contra == case.expected else "MISS"
        print(
            f"[{index}/300] {case.id} exp={case.expected} got={has_contra} {ok} "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        if os.environ.get("HUFU_DEBUG_CASE") == case.id:
            from hufu.engine.build_hypergraph import text_to_hypergraph
            from hufu.engine.star_graph import extract_stars
            dbg_local = text_to_hypergraph(case.text)
            for s in extract_stars(dbg_local):
                print(f"    DBG-STAR center={s.center.text()!r}", flush=True)
                for f in s.facts:
                    print(f"      pred={f.predicate!r} role={f.center_role} others={f.others}", flush=True)
            for c in conflicts or []:
                print(f"    DBG {c.rule} direct={c.direct} :: {c.explanation[:150]}", flush=True)
                print(f"      A: {c.fact_a.sentence[:100]}", flush=True)
                print(f"      B: {c.fact_b.sentence[:100]}", flush=True)
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

    report_path.write_text(
        json.dumps(
            {"cases": report},
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
        f"(P={precision:.3f} R={recall:.3f} F1={f1:.3f}; "
        f"tp={tp} fn={fn} fp={fp} tn={tn}), "
        f"{total_seconds:.0f}s total; by family: {summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
