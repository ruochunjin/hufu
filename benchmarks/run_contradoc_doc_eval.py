"""Document-level self-contradiction evaluation on ContraDoc.

**Status: EXPERIMENTAL v2 baseline — see the calibration findings below.**

Protocol:
  1. Split the document into sentences.
  2. Embed sentences; every sentence takes its top-K most similar *other*
     sentences as conflict candidates (contradictions live between
     topically-related sentences).
  3. Rule cues (numeric difference / refined negation) pre-filter, then the
     NLI contradiction probability confirms.

Calibration findings (20 pos + 20 neg, 2026-10-05):
- NLI alone floods: clean narrative docs get max contradiction prob ≈ 1.0
  just like contradictory ones — no separable threshold (precision 0.51).
- Cue gates × NLI thresholds trade precision/recall continuously with no
  knee; best F1 ≈ 0.55.  Count features (pairs above p) overlap, too.
- Root cause: ContraDoc conflicts are "same slot, different value"
  (number/date/place/person) phrased in lexically diverse ways; sentence
  NLI cannot tell value conflict from narrative tension.

The validated boundary of the current stack is the SENTENCE-PAIR channel
(see run_contradoc_eval.py, F1 0.9474).  The v3 path for documents is the
typed-value slot model over the document hypergraph (design recorded in the
working record); not built in this round.

Usage:
    HF_HUB_OFFLINE=1 .venv-e2e/Scripts/python.exe \
        benchmarks/run_contradoc_doc_eval.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nli_loaders import load_contradoc  # noqa: E402

_MIN_SENT_WORDS = 5
_TOP_K = 3
_NLI_BATCH = 32

_NEGATION = re.compile(r"\b(not|n't|never|no|none|nothing|nobody|nowhere|neither|nor|without)\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_TOKEN = re.compile(r"[a-z]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "was", "were", "are", "be", "been",
    "being", "has", "have", "had", "do", "does", "did", "that", "this",
    "these", "those", "it", "its", "his", "her", "their", "he", "she",
    "they", "we", "you", "i", "who", "which", "when", "where", "what", "how",
}


def _content_tokens(text: str) -> set[str]:
    """Lowercased content tokens minus stopwords and negation markers."""

    return {
        token
        for token in _TOKEN.findall(text.lower())
        if token not in _STOPWORDS
        and token
        not in {"not", "nt", "never", "no", "none", "nothing", "nobody", "nowhere", "neither", "nor", "without"}
    }


def _conflict_cues(a: str, b: str, overlap_t: float = 0.4) -> tuple[bool, bool]:
    """Cheap rule cues that a topically-similar pair can be a true conflict.

    NLI-only doc-level detection fires on narrative tension in clean docs
    (all-max-prob=1.0 on both classes, calibration 2026-10-05).  The cues:

    - C1 numeric: both sentences state numbers and the value sets differ
      ("300 men" vs "3000 men");
    - C2 refined negation: exactly one side carries a negation marker AND,
      after negation is stripped, the content tokens still overlap at least
      ``overlap_t`` (Jaccard) — the same event negated, not ordinary
      narrative negation in unrelated clauses.  The operating point is
      swept in benchmarks/calibrate_doc_threshold.py.
    """

    numbers_a = set(_NUMBER.findall(a))
    numbers_b = set(_NUMBER.findall(b))
    numeric = bool(numbers_a and numbers_b and numbers_a != numbers_b)
    negation_asym = bool(_NEGATION.search(a)) != bool(_NEGATION.search(b))
    if not negation_asym or overlap_t <= 0.0:
        return numeric, negation_asym
    tokens_a = _content_tokens(a)
    tokens_b = _content_tokens(b)
    if not tokens_a or not tokens_b:
        return numeric, False
    overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    return numeric, overlap >= overlap_t


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence split (ContraDoc docs are plain prose)."""

    import re

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if len(part.split()) >= _MIN_SENT_WORDS]


def detect_self_contradiction(
    text: str,
    *,
    top_k: int = _TOP_K,
    nli_threshold: float = 0.5,
) -> tuple[bool, float, str]:
    """Return ``(has_contradiction, seconds, evidence)`` for one document.

    Cue-first, NLI-confirmed: a candidate pair is a true conflict only when a
    rule cue (numeric difference or negation asymmetry) is present AND the
    NLI contradiction probability clears ``nli_threshold``.  The cues carry
    the precision; NLI alone fires on narrative tension in clean documents.
    """

    start = time.perf_counter()
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return False, time.perf_counter() - start, "[CLEAN] too few sentences"

    import numpy as np

    from hufu.engine.embedding import get_embedding_batch
    from hufu.engine.nli import get_nli_contradiction_probs_batch

    vectors = get_embedding_batch(sentences)
    matrix = np.array([v / (np.linalg.norm(v) or 1.0) for v in vectors])
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -1.0)

    candidate_pairs: set[tuple[int, int]] = set()
    for index in range(len(sentences)):
        nearest = np.argsort(-sims[index])[:top_k]
        for other in nearest:
            if sims[index, other] <= 0.0:
                continue
            pair = (min(index, int(other)), max(index, int(other)))
            candidate_pairs.add(pair)

    cued_pairs = [
        (i, j)
        for i, j in sorted(candidate_pairs)
        if any(_conflict_cues(sentences[i], sentences[j]))
    ]
    if not cued_pairs:
        return (
            False,
            time.perf_counter() - start,
            f"[CLEAN] {len(sentences)} sentences, {len(candidate_pairs)} pairs, none with conflict cues",
        )
    probs = get_nli_contradiction_probs_batch(
        [(sentences[i], sentences[j]) for i, j in cued_pairs]
    )
    for (i, j), prob in zip(cued_pairs, probs):
        if prob >= nli_threshold:
            cues = _conflict_cues(sentences[i], sentences[j])
            cue_names = "+".join(
                name for name, on in (("numeric", cues[0]), ("negation", cues[1])) if on
            )
            evidence = (
                f"[CONTRADICTION] sentences #{i + 1} and #{j + 1} "
                f"(cues={cue_names}, NLI p={prob:.2f}):\n"
                f"  • {sentences[i][:160]}\n"
                f"  • {sentences[j][:160]}"
            )
            return True, time.perf_counter() - start, evidence
    return (
        False,
        time.perf_counter() - start,
        f"[CLEAN] {len(sentences)} sentences, {len(cued_pairs)} cued pairs below threshold",
    )


def evaluate(limit: int, top_k: int) -> dict:
    results = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "skip": 0}
    details: list[dict] = []
    pos_seen = neg_seen = 0
    for example in load_contradoc():
        positive = example.label == "contradiction"
        if positive and pos_seen >= limit:
            continue
        if not positive and neg_seen >= limit:
            continue
        pos_seen += int(positive)
        neg_seen += int(not positive)
        has_contra, seconds, evidence = detect_self_contradiction(
            example.premise, top_k=top_k
        )
        if positive:
            results["tp" if has_contra else "fn"] += 1
        else:
            results["fp" if has_contra else "tn"] += 1
        details.append(
            {
                "id": example.example_id,
                "label": example.label,
                "flagged": has_contra,
                "seconds": round(seconds, 2),
                "evidence_head": evidence.splitlines()[0][:140],
            }
        )
        done = results["tp"] + results["fn"] + results["tn"] + results["fp"]
        print(f"[{done}] {example.example_id} label={example.label} flagged={has_contra} ({seconds:.1f}s)", flush=True)
        if pos_seen >= limit and neg_seen >= limit:
            break

    tp, fn, tn, fp = results["tp"], results["fn"], results["tn"], results["fp"]
    total = tp + fn + tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "protocol": f"similarity-top{top_k}+nli",
        "confusion": results,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50, help="max pos and max neg documents")
    parser.add_argument("--top-k", type=int, default=_TOP_K)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate(args.limit, args.top_k)
    summary = {key: value for key, value in report.items() if key != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = args.out or (
        Path(__file__).parent / "data" / "cache" / f"contradoc_doc_eval_{args.limit}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
