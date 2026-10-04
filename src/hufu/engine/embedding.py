"""Lazy sentence-embedding and cosine-similarity helpers.

The embedding model is loaded on first use, never at import time.  The model
name is configurable through the ``HUFU_EMBED_MODEL`` environment variable;
any Hugging Face hub identifier or local path accepted by
``sentence_transformers.SentenceTransformer`` works.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer

# 2026-10-04 A/B (CPU, golden suites): all-MiniLM-L6-v2 (22M) is 3.7x faster
# than Qwen/Qwen3-Embedding-0.6B with identical golden verdicts, and 39x on
# the F9 worst case.  Engineering default per the lab's decision; set
# HUFU_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B to reproduce paper-family runs.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model_cache: dict[str, "SentenceTransformer"] = {}
# Process-wide text->vector cache shared by ALL get_embedding_batch callers
# (HC scoring, D-match cells, distance gates): the same role/cluster texts
# recur across call sites within one pair and across pairs in a batch.
_default_text_cache: dict = {}


def _get_sentence_transformer() -> "SentenceTransformer":
    """Return the lazily initialized sentence-transformer model."""

    name = os.environ.get("HUFU_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    if name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name)
        model.eval()
        _model_cache[name] = model
    return _model_cache[name]


def init_embedding_model() -> None:
    """Eagerly initialize the configured sentence-transformer model."""

    _get_sentence_transformer()


def get_embedding_batch(
    texts: list[str],
    batch_size: int = 256,
    cache: None | dict[str, np.ndarray] = None,
) -> list[np.ndarray]:
    """Encode texts in batches, reusing caller-supplied normalized vectors.

    When ``HUFU_EMBED_CACHE`` points at a directory, embeddings are
    additionally persisted as ``.npy`` files keyed by (model, text) hash —
    benchmark reruns then skip encoding entirely.
    """

    model = _get_sentence_transformer()
    if cache is None:
        cache = _default_text_cache
    disk_dir = os.environ.get("HUFU_EMBED_CACHE")
    model_name = os.environ.get("HUFU_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    unique_texts = list(set(texts))
    missing_texts = [t for t in unique_texts if t not in cache]
    disk_hits: dict[str, np.ndarray] = {}
    if disk_dir and missing_texts:
        import hashlib
        from pathlib import Path

        model_dir = Path(disk_dir) / hashlib.sha256(model_name.encode()).hexdigest()[:12]
        for text in list(missing_texts):
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
            path = model_dir / f"{key}.npy"
            if path.exists():
                disk_hits[text] = np.load(path)
        if disk_hits:
            cache.update(disk_hits)
            missing_texts = [t for t in missing_texts if t not in disk_hits]
    if missing_texts:
        new_embeddings = model.encode(
            missing_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        cache.update(zip(missing_texts, new_embeddings))
        if disk_dir:
            import hashlib
            from pathlib import Path

            model_dir = Path(disk_dir) / hashlib.sha256(model_name.encode()).hexdigest()[:12]
            model_dir.mkdir(parents=True, exist_ok=True)
            for text, vector in zip(missing_texts, new_embeddings):
                key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
                np.save(model_dir / f"{key}.npy", vector)
    return [cache[t] for t in texts]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity for two dense vectors."""

    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


def get_similarity_batch(query: list[str], data: list[str], N: int = 8) -> list[float]:
    """Score the Cartesian product of query and data strings in row-major order."""

    query_embeddings = get_embedding_batch(query, N)
    data_embeddings = get_embedding_batch(data, N)
    pairs = [(q_emb, d_emb) for q_emb in query_embeddings for d_emb in data_embeddings]
    return get_cosine_similarity_batch(pairs, is_normalized=True)


def get_similarity(text1: str, text2: str) -> float:
    """Encode and compare one text pair."""

    emb1 = get_embedding_batch([text1])[0]
    emb2 = get_embedding_batch([text2])[0]
    return cosine_similarity(emb1, emb2)


def get_cosine_similarity_batch(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    is_normalized: bool = False,
) -> list[float]:
    """Compute pairwise cosine scores, optionally trusting pre-normalized inputs."""

    if not pairs:
        return []
    A_list, B_list = zip(*pairs)
    A = np.stack(A_list)
    B = np.stack(B_list)
    if is_normalized:
        similarities = np.einsum("ij,ij->i", A, B)
        return similarities.tolist()
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)
    A_norms[A_norms == 0] = 1e-10
    B_norms[B_norms == 0] = 1e-10
    A_normalized = A / A_norms
    B_normalized = B / B_norms
    similarities = np.einsum("ij,ij->i", A_normalized, B_normalized)
    return similarities.tolist()
