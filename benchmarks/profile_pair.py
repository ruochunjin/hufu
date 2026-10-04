"""Per-stage latency profile: cold-start model loads vs warm per-pair costs.

Sequences the pipeline stages explicitly so every timer is exact:
build (spaCy+coref), bridge, hv, run_hypermatch (embedding instrumented),
NLI gate.  A warmup pass loads all models first; stage times below are the
steady-state per-pair costs that dominate batch evaluation.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("HUFU_SPACY_MODEL", "en_core_web_sm")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import hufu.engine.embedding as emb  # noqa: E402
from hufu.engine.bridge import bridge_allowed_pairs, bridge_hypergraph, bridge_type_compatible, whole_query_clusters  # noqa: E402
from hufu.engine.build_hypergraph import get_nlp, text_to_hypergraph  # noqa: E402
from hufu.engine.fixed_point import FixedPointMode  # noqa: E402
from hufu.engine.hyper_simulation import run_hypermatch  # noqa: E402
from hufu.backends import EmbeddingRoleCellScorer, SentenceTransformerBackend  # noqa: E402

_encode = {"calls": 0, "texts": 0, "seconds": 0.0}
_orig = emb.get_embedding_batch


def _timed(texts, batch_size=256, cache=None):
    start = time.perf_counter()
    out = _orig(texts, batch_size, cache)
    _encode["calls"] += 1
    _encode["texts"] += len(set(texts))
    _encode["seconds"] += time.perf_counter() - start
    return out


emb.get_embedding_batch = _timed

CASES = [
    ("short-Q", "Where was Ada Lovelace born?", "Ada Lovelace was born in London in 1815.", True),
    ("declarative", "The reservoir holds 3.2 million cubic meters of water.", "The reservoir holds 3.3 million cubic meters of water.", False),
    (
        "medium-doc",
        "How many goals did the striker score in the 1998 season?",
        "The 1998 season was the club's best in a decade. The midfielder controlled the tempo "
        "throughout the spring. The striker scored 23 goals that season, including a hat-trick "
        "in the final derby, before a knee injury ended his campaign early.",
        True,
    ),
]


def timed(fn, times, name):
    start = time.perf_counter()
    result = fn()
    times[name] = time.perf_counter() - start
    return result


def run_case(name: str, query: str, doc: str, is_question: bool) -> None:
    times: dict[str, float] = {}
    _encode.update(calls=0, texts=0, seconds=0.0)
    total = time.perf_counter()
    q_local = timed(lambda: text_to_hypergraph(query, is_query=is_question), times, "build_query")
    d_local = timed(lambda: text_to_hypergraph(doc), times, "build_data")
    q_core = timed(lambda: bridge_hypergraph(q_local, side="query").hypergraph, times, "bridge_q")
    d_core = timed(lambda: bridge_hypergraph(d_local, side="data").hypergraph, times, "bridge_d")
    hv = timed(lambda: bridge_allowed_pairs(q_local, d_local, strategy="typed"), times, "hv")
    backend = SentenceTransformerBackend()
    scorer = EmbeddingRoleCellScorer(backend)
    clusters = whole_query_clusters(q_core)
    out = timed(
        lambda: run_hypermatch(
            q_core, d_core,
            hv_allowed_pairs=hv, hc_backend=backend, role_scorer=scorer,
            query_clusters=clusters, type_compatible=bridge_type_compatible,
            fixed_point_mode=FixedPointMode.WITNESS_OR,
        ),
        times, "hypermatch",
    )
    if not is_question:
        from hufu.engine.nli import get_nli_label

        timed(lambda: get_nli_label(query, doc), times, "nli_gate")
    total_s = time.perf_counter() - total
    print(f"\n=== {name}: total={total_s:.2f}s")
    for stage, secs in sorted(times.items(), key=lambda kv: -kv[1]):
        print(f"  {stage:14} {secs:7.2f}s  {100 * secs / total_s:5.1f}%")
    print(f"  embedding: {_encode['calls']} batches, {_encode['texts']} unique texts, {_encode['seconds']:.2f}s")


if __name__ == "__main__":
    print("--- cold start ---")
    t = time.perf_counter()
    get_nlp()
    print(f"spacy+coref pipeline load: {time.perf_counter() - t:.2f}s")
    t = time.perf_counter()
    emb.get_embedding_batch(["warmup"])
    print(f"embedding model load+first encode: {time.perf_counter() - t:.2f}s")
    print("--- warm per-pair ---")
    for name, q, d, is_q in CASES:
        run_case(name, q, d, is_q)
    print("\n--- second pass (all warm incl. cached embedding texts) ---")
    for name, q, d, is_q in CASES:
        run_case(name, q, d, is_q)
