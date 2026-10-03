"""Optional model-backed adapters for the HyperMatch reference core.

Everything here is lazy: importing this module never loads torch,
sentence-transformers, or any model weights.  Models are resolved on first
use through ``hufu.engine.embedding`` / ``hufu.engine.nli`` (configurable via
``HUFU_EMBED_MODEL`` / ``HUFU_NLI_MODEL``).
"""

from __future__ import annotations

import math
from typing import Sequence

from hufu.engine.d_match import RoleCell, RoleScoringResult
from hufu.engine.scoring import EmbeddingBackend


class SentenceTransformerBackend(EmbeddingBackend):
    """``EmbeddingBackend`` protocol implementation over sentence-transformers.

    Vectors are L2-normalized by the underlying helper, matching the scoring
    assumptions of the reference core.
    """

    def encode(
        self, texts: Sequence[str], *, prompt: str | None = None
    ) -> Sequence[Sequence[float]]:
        from hufu.engine.embedding import get_embedding_batch

        if prompt is not None:
            texts = [f"{prompt}{text}" for text in texts]
        vectors = get_embedding_batch(list(texts))
        return [vector.tolist() for vector in vectors]


class EmbeddingRoleCellScorer:
    """Score D-match role cells by embedding cosine of the target surfaces.

    First-cut model-backed scorer for the reference core, calibrated against
    the standard path in the L3 dual-track tests.  Cosine similarity of the
    two target-surface embeddings is squashed to a probability with
    ``(cos + 1) / 2``; identical surfaces short-circuit to 1.0.
    """

    def __init__(self, backend: EmbeddingBackend | None = None) -> None:
        self._backend = backend or SentenceTransformerBackend()

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        norm_l = math.sqrt(sum(a * a for a in left))
        norm_r = math.sqrt(sum(b * b for b in right))
        if not norm_l or not norm_r:
            return 0.0
        return dot / (norm_l * norm_r)

    def score(self, cells: Sequence[RoleCell]) -> RoleScoringResult:
        texts: list[str] = []
        for cell in cells:
            texts.append(cell.query_target_text)
            texts.append(cell.data_target_text)
        vectors = list(self._backend.encode(texts)) if texts else []
        scores: dict[tuple[str, str], float] = {}
        for index, cell in enumerate(cells):
            same_surface = (
                cell.query_target_text.strip().casefold()
                == cell.data_target_text.strip().casefold()
            )
            if same_surface:
                scores[cell.pair] = 1.0
                continue
            cos = self._cosine(vectors[2 * index], vectors[2 * index + 1])
            scores[cell.pair] = (cos + 1.0) / 2.0
        return RoleScoringResult(scores)


__all__ = ["EmbeddingRoleCellScorer", "SentenceTransformerBackend"]
