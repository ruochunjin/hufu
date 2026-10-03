"""ContraDoc evaluation harness: contradiction detection accuracy.

For every positive document with ``ref sentences``, the pair
``(ref_sentence, evidence)`` is a golden contradiction case: the checker
should return ``consistent=False``.  For negative documents a sentence pair
is sampled from the same clean document (first and middle sentence), which
should return ``consistent=True``.

Hypergraph construction dominates runtime on long documents (coreference on
300-2200 token texts), so built graphs are cached as pickles keyed by text
hash under ``benchmarks/data/cache/hypergraphs/`` — reruns skip the build.

Usage (from the repo root, e2e venv):

    HUFU_SPACY_MODEL=en_core_web_sm HF_HUB_OFFLINE=1 \
        .venv-e2e/Scripts/python.exe benchmarks/run_contradoc_eval.py \
        --engine reference --limit 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nli_loaders import load_contradoc  # noqa: E402

CACHE_DIR = Path(__file__).parent / "data" / "cache" / "hypergraphs"
_MIN_REF_WORDS = 6


def _text_key(text: str, is_query: bool) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    return f"{'q' if is_query else 'd'}-{digest}"


def _cached_hypergraph(text: str, *, is_query: bool, spacy_model: str):
    """Build or load a pickled LocalHypergraph for one text."""

    from hufu.engine.build_hypergraph import text_to_hypergraph

    key = f"{spacy_model}-{_text_key(text, is_query)}"
    target = CACHE_DIR / f"{key}.pkl"
    if target.exists():
        from hufu.hypergraph.hypergraph import Hypergraph as LocalHypergraph

        try:
            return LocalHypergraph.load(str(target))
        except Exception:
            target.unlink(missing_ok=True)
    graph = text_to_hypergraph(text, is_query=is_query)
    target.parent.mkdir(parents=True, exist_ok=True)
    graph.save(str(target))
    return graph


def _neg_sentence_pair(text: str) -> tuple[str, str] | None:
    """Pick two well-separated sentences from a clean document."""

    sentences = [s.strip() for s in text.split(". ") if len(s.split()) >= _MIN_REF_WORDS]
    if len(sentences) < 3:
        return None
    return sentences[0], sentences[len(sentences) // 2]


def evaluate(engine: str, limit: int, threshold: float) -> dict:
    from hufu.engine.reference_check import reference_consistent_detection

    spacy_model = os.environ.get("HUFU_SPACY_MODEL", "en_core_web_sm")
    results = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "skip": 0}
    details: list[dict] = []

    pos_seen = neg_seen = 0
    for example in load_contradoc():
        if example.label == "contradiction" and pos_seen >= limit:
            continue
        if example.label == "entailment" and neg_seen >= limit:
            continue
        if example.label == "contradiction":
            refs = [
                ref
                for ref in example.meta.get("ref_sentences", [])
                if len(ref.split()) >= _MIN_REF_WORDS
            ]
            if not refs:
                results["skip"] += 1
                continue
            query_text, doc_text, expected = refs[0], example.hypothesis, False
            pos_seen += 1
        else:
            pair = _neg_sentence_pair(example.premise)
            if pair is None:
                results["skip"] += 1
                continue
            query_text, doc_text, expected = pair[0], pair[1], True
            neg_seen += 1

        start = time.perf_counter()
        if engine == "reference":
            query_graph = _cached_hypergraph(query_text, is_query=True, spacy_model=spacy_model)
            data_graph = _cached_hypergraph(doc_text, is_query=False, spacy_model=spacy_model)
            has_contradiction, _dist, evidence = reference_consistent_detection(
                query_text,
                doc_text,
                distance_threshold=threshold,
                query_local=query_graph,
                data_local=data_graph,
            )
            consistent = not has_contradiction
        else:
            import hufu

            verdict = hufu.check_pair(
                query_text, doc_text, distance_threshold=threshold, engine=engine
            )
            consistent = verdict.consistent
            evidence = verdict.evidence
        elapsed = time.perf_counter() - start

        # Confusion accounting: expected False = contradiction (positive class).
        if not expected and not consistent:
            results["tp"] += 1
        elif not expected and consistent:
            results["fn"] += 1
        elif expected and consistent:
            results["tn"] += 1
        else:
            results["fp"] += 1
        details.append(
            {
                "id": example.example_id,
                "label": example.label,
                "consistent": consistent,
                "expected_consistent": expected,
                "seconds": round(elapsed, 2),
                "evidence_head": evidence.splitlines()[0][:120],
            }
        )
        done = results["tp"] + results["fn"] + results["tn"] + results["fp"]
        print(f"[{done}] {example.example_id} label={example.label} consistent={consistent} ({elapsed:.1f}s)", flush=True)
        if pos_seen >= limit and neg_seen >= limit:
            break

    tp, fn, tn, fp = results["tp"], results["fn"], results["tn"], results["fp"]
    total = tp + fn + tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "engine": engine,
        "threshold": threshold,
        "confusion": results,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["standard", "reference"], default="reference")
    parser.add_argument("--limit", type=int, default=30, help="max pos and max neg examples")
    parser.add_argument("--threshold", type=float, default=-1.0, help="distance gate (-1 forces structural)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate(args.engine, args.limit, args.threshold)
    summary = {key: value for key, value in report.items() if key != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = args.out or (
        Path(__file__).parent / "data" / "cache" / f"contradoc_eval_{args.engine}_{args.limit}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
