"""Sentence-pair NLI robustness eval: HuFu strict checker vs open NLI benchmarks.

Datasets (via ``nli_loaders``): contract_nli, docnli, control.  The hypothesis
is checked against the premise as a declarative claim
(``query_is_question=False``), reference engine, distance channel pinned off
(``distance_threshold=-1.0``) per the golden-suite semantics: the tool
certifies *structural* consistency, so only structural contradictions are
expected to be caught — this harness measures robustness (no crashes,
bounded runtime) plus the achievable contradiction recall / clean-pass rate
on each label.

Sampling is stratified: up to ``--per-label`` examples per raw label
(contradiction / entailment / neutral), seeded for reproducibility.

Usage (repo root, e2e venv)::

    HF_ENDPOINT=https://hf-mirror.com .venv-e2e/Scripts/python.exe \
        benchmarks/run_nli_eval.py --dataset control --per-label 30
    # cached reruns:
    HF_HUB_OFFLINE=1 .venv-e2e/Scripts/python.exe \
        benchmarks/run_nli_eval.py --dataset contract_nli --per-label 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nli_loaders import NLIExample, load_hf_nli  # noqa: E402

REPORT_DIR = Path(__file__).parent / "data" / "cache"


def stratified_sample(name: str, per_label: int, seed: int, split: str, scan_limit: int | None) -> list[NLIExample]:
    buckets: dict[str, list[NLIExample]] = {}
    # ``scan_limit`` caps how many rows are scanned before sampling — large
    # datasets (DocNLI) would otherwise load hundreds of thousands of
    # document-length premises into memory.
    for ex in load_hf_nli(name, split=split, limit=scan_limit):
        buckets.setdefault(ex.label, []).append(ex)
    rng = random.Random(seed)
    picked: list[NLIExample] = []
    for label, rows in sorted(buckets.items()):
        rows = rows[:]
        rng.shuffle(rows)
        picked.extend(rows[:per_label])
    rng.shuffle(picked)
    return picked


def _write_report(out: Path, args, rows: list[dict], n_error: int, t0: float) -> None:
    """Write the (possibly partial) report — scoring is over completed rows."""

    ok = [r for r in rows if r["predicted_contradiction"] is not None]

    def _scores(positive_labels: set[str]) -> dict:
        tp = sum(1 for r in ok if r["label"] in positive_labels and r["predicted_contradiction"])
        fn = sum(1 for r in ok if r["label"] in positive_labels and not r["predicted_contradiction"])
        fp = sum(1 for r in ok if r["label"] not in positive_labels and r["predicted_contradiction"])
        tn = sum(1 for r in ok if r["label"] not in positive_labels and not r["predicted_contradiction"])
        tot = tp + fn + fp + tn
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "accuracy": (tp + tn) / tot if tot else None,
        }

    report = {
        "dataset": args.dataset,
        "per_label": args.per_label,
        "seed": args.seed,
        "n": len(rows),
        "errors": n_error,
        "contradiction_vs_rest": _scores({"contradiction"}),
        "not_covered_vs_entailed": _scores({"neutral", "contradiction"}),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "rows": rows,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["contract_nli", "docnli", "control"])
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scan-limit", type=int, default=None,
                        help="cap rows scanned before stratified sampling (large datasets)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from hufu import check_pair

    examples = stratified_sample(args.dataset, args.per_label, args.seed, args.split, args.scan_limit)
    print(f"[{args.dataset}] {len(examples)} examples "
          f"({ {k: sum(1 for e in examples if e.label == k) for k in {e.label for e in examples}} })")

    # Resume: an existing report at --out contributes completed rows; ids
    # already done are skipped so killed long runs restart cheaply.
    out = Path(args.out) if args.out else REPORT_DIR / f"nli_eval_{args.dataset}_{len(examples)}.json"
    done_rows: list[dict] = []
    done_ids: set[str] = set()
    if out.exists():
        try:
            prior = json.loads(out.read_text(encoding="utf-8"))
            done_rows = [r for r in prior.get("rows", []) if r.get("predicted_contradiction") is not None]
            done_ids = {r["id"] for r in done_rows}
            if done_ids:
                print(f"resuming: {len(done_ids)} rows already done")
        except (json.JSONDecodeError, OSError):
            pass

    rows = []
    n_error = 0
    t0 = time.perf_counter()
    total = len(examples) - len(done_ids)
    seen = 0
    for i, ex in enumerate(examples):
        if ex.example_id in done_ids:
            continue
        seen += 1
        start = time.perf_counter()
        try:
            verdict = check_pair(
                ex.hypothesis,
                ex.premise,
                engine="reference",
                query_is_question=False,
                distance_threshold=-1.0,
            )
            predicted_contra = not verdict.consistent
            evidence = verdict.evidence[:200]
            error = None
        except Exception as exc:  # robustness: count, never abort the batch
            n_error += 1
            predicted_contra = None
            evidence = ""
            error = f"{type(exc).__name__}: {exc}"[:300]
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "id": ex.example_id,
                "label": ex.label,
                "predicted_contradiction": predicted_contra,
                "seconds": round(elapsed, 2),
                "evidence": evidence,
                "error": error,
            }
        )
        print(f"  [{seen}/{total}] {ex.label:>13} -> {predicted_contra} "
              f"({elapsed:.1f}s){' ERROR ' + error if error else ''}", flush=True)
        # incremental flush: partial results survive a killed shell
        _write_report(out, args, done_rows + rows, n_error, t0)

    rows = done_rows + rows
    ok = [r for r in rows if r["predicted_contradiction"] is not None]

    def _scores(positive_labels: set[str]) -> dict:
        """Binary P/R/F1/accuracy with ``positive_labels`` as the positive class."""

        tp = sum(1 for r in ok if r["label"] in positive_labels and r["predicted_contradiction"])
        fn = sum(1 for r in ok if r["label"] in positive_labels and not r["predicted_contradiction"])
        fp = sum(1 for r in ok if r["label"] not in positive_labels and r["predicted_contradiction"])
        tn = sum(1 for r in ok if r["label"] not in positive_labels and not r["predicted_contradiction"])
        total = tp + fn + fp + tn
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "accuracy": (tp + tn) / total if total else None,
        }

    # Two scorings, matching the two semantics the tool certifies:
    # - contradiction_vs_rest: classic NLI contradiction detection;
    # - not_covered_vs_entailed: hufu's pinned coverage semantics — an
    #   entailment hypothesis IS covered by the premise (consistent=True),
    #   neutral/contradiction are not (consistent=False).
    contradiction = _scores({"contradiction"})
    not_covered = _scores({"neutral", "contradiction"})
    report = {
        "dataset": args.dataset,
        "per_label": args.per_label,
        "seed": args.seed,
        "n": len(rows),
        "errors": n_error,
        "contradiction_vs_rest": contradiction,
        "not_covered_vs_entailed": not_covered,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "rows": rows,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[{args.dataset}] errors={n_error}")
    print(f"  contradiction-vs-rest:  acc={contradiction['accuracy']} recall={contradiction['recall']}")
    print(f"  not-covered-vs-entailed: acc={not_covered['accuracy']} recall={not_covered['recall']}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
