"""Threshold calibration for the document-level self-contradiction protocol.

Collects per-candidate-pair contradiction probabilities for a pos/neg sample
ONCE (the expensive NLI calls), then sweeps the decision threshold offline
and reports accuracy/precision/recall/F1 per threshold, so the operating
point is chosen from data rather than guessed.

    HF_HUB_OFFLINE=1 .venv-e2e/Scripts/python.exe \
        benchmarks/calibrate_doc_threshold.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
from nli_loaders import load_contradoc  # noqa: E402
from run_contradoc_doc_eval import _conflict_cues, _split_sentences  # noqa: E402

from hufu.engine.embedding import get_embedding_batch  # noqa: E402
from hufu.engine.nli import get_nli_contradiction_probs_batch  # noqa: E402

_TOP_K = 3


def collect(limit: int, top_k: int) -> list[dict]:
    """Per-document feature rows for offline threshold sweeps."""

    docs: list[dict] = []
    pos_seen = neg_seen = 0
    for example in load_contradoc():
        positive = example.label == "contradiction"
        if positive and pos_seen >= limit:
            continue
        if not positive and neg_seen >= limit:
            continue
        pos_seen += int(positive)
        neg_seen += int(not positive)
        sentences = _split_sentences(example.premise)
        if len(sentences) < 2:
            continue
        vectors = get_embedding_batch(sentences)
        matrix = np.array([v / (np.linalg.norm(v) or 1.0) for v in vectors])
        sims = matrix @ matrix.T
        np.fill_diagonal(sims, -1.0)
        pairs: set[tuple[int, int]] = set()
        for index in range(len(sentences)):
            for other in np.argsort(-sims[index])[:top_k]:
                if sims[index, other] > 0:
                    pairs.add((min(index, int(other)), max(index, int(other))))
        ordered = sorted(pairs)
        probs = get_nli_contradiction_probs_batch(
            [(sentences[i], sentences[j]) for i, j in ordered]
        )
        pair_rows = []
        for (i, j), prob in zip(ordered, probs):
            numeric, _neg_raw = _conflict_cues(sentences[i], sentences[j], overlap_t=0.0)
            overlap_levels = {
                f"neg_o{t}": _conflict_cues(sentences[i], sentences[j], overlap_t=t)[1]
                for t in (0.1, 0.2, 0.3, 0.4)
            }
            pair_rows.append(
                {
                    "cos": round(float(sims[i, j]), 4),
                    "nli_p": round(prob, 4),
                    "numeric": numeric,
                    "negation": _neg_raw,
                    **overlap_levels,
                }
            )
        docs.append({"positive": positive, "pairs": pair_rows})
        best = max((p["nli_p"] for p in pair_rows), default=0.0)
        print(f"[{len(docs)}] {example.example_id} pos={positive} pairs={len(pair_rows)} max_p={best:.3f}", flush=True)
        if pos_seen >= limit and neg_seen >= limit:
            break
    return docs


def sweep(docs: list[dict]) -> list[dict]:
    """Sweep joint thresholds: flag doc when ∃ pair with cue ∧ p≥τ."""

    rows = []
    cue_modes = ("negation", "neg_o0.1", "neg_o0.2", "neg_o0.3", "neg_o0.4", "numeric", "any")
    for p_t in [0.5, 0.7, 0.9]:
        for cue in cue_modes:
            def flagged(doc):
                return any(
                    pair["nli_p"] >= p_t
                    and (cue == "any" and (pair["numeric"] or pair["negation"]) or pair.get(cue, False))
                    for pair in doc["pairs"]
                )
            tp = sum(1 for d in docs if d["positive"] and flagged(d))
            fn = sum(1 for d in docs if d["positive"] and not flagged(d))
            fp = sum(1 for d in docs if not d["positive"] and flagged(d))
            tn = sum(1 for d in docs if not d["positive"] and not flagged(d))
            total = tp + fn + fp + tn
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "p_t": p_t, "cue": cue,
                    "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                    "accuracy": round((tp + tn) / total, 4) if total else 0.0,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                }
            )
    rows.sort(key=lambda row: -row["f1"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=_TOP_K)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = collect(args.limit, args.top_k)
    rows = sweep(records)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    out = args.out or (
        Path(__file__).parent / "data" / "cache" / f"doc_threshold_calibration_{args.limit}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"records": records, "sweep": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"written to {out}")


if __name__ == "__main__":
    main()
