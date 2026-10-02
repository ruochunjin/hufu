"""Optional model-backed adapters for the HyperMatch reference core.

Everything here is lazy: importing this module never loads torch,
sentence-transformers, or any model weights.  Models are resolved on first
use through ``hufu.engine.embedding`` / ``hufu.engine.nli`` (configurable via
``HUFU_EMBED_MODEL`` / ``HUFU_NLI_MODEL``).
"""

from __future__ import annotations

from typing import Sequence

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


__all__ = ["SentenceTransformerBackend"]
