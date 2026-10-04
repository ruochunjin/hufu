"""Lazy cross-encoder helpers for NLI decisions and scores.

The cross-encoder model is loaded on first use, never at import time.  The
model name is configurable through the ``HUFU_NLI_MODEL`` environment
variable; any Hugging Face hub identifier or local path accepted by
``sentence_transformers.CrossEncoder`` works.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import CrossEncoder

# 2026-10-04 A/B on the declarative polarity channel (golden100 I-family +
# ContraDoc smoke): xsmall keeps identical verdicts at ~4x lower latency
# (0.11s vs 0.40s per call on CPU).  HUFU_NLI_MODEL overrides.
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
BATCH_SIZE = 1024

_LABELS = ["contradiction", "entailment", "neutral"]
_ENTAILMENT_INDEX = 1
_CONTRADICTION_INDEX = 0

_model_cache: dict[str, "CrossEncoder"] = {}


def _get_nli_model() -> "CrossEncoder":
    """Return the lazily initialized NLI cross-encoder."""

    name = os.environ.get("HUFU_NLI_MODEL", DEFAULT_NLI_MODEL)
    if name not in _model_cache:
        from sentence_transformers import CrossEncoder

        _model_cache[name] = CrossEncoder(name)
    return _model_cache[name]


def init_nli_model() -> None:
    """Eagerly initialize the configured NLI cross-encoder."""

    _get_nli_model()


def get_nli_labels_batch(pairs: list[tuple[str, str]]) -> list[str]:
    """Predict contradiction, entailment, or neutral for each ordered text pair."""

    model = _get_nli_model()
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    return [_LABELS[argmax] for argmax in scores.argmax(axis=1)]


def get_nli_labels_with_score_batch(pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Return each predicted label with its raw entailment score."""

    model = _get_nli_model()
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    return [
        (_LABELS[argmax], scores[i][_ENTAILMENT_INDEX])
        for i, argmax in enumerate(scores.argmax(axis=1))
    ]


def get_nli_label(text1: str, text2: str) -> str:
    """Predict one ordered text pair's NLI label."""

    return get_nli_labels_batch([(text1, text2)])[0]


def get_nli_entailment_score_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """Return raw entailment scores for ordered text pairs."""

    model = _get_nli_model()
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    return [score[_ENTAILMENT_INDEX] for score in scores]


def get_nli_contradiction_score_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """Return raw contradiction scores for ordered text pairs."""

    model = _get_nli_model()
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    return [score[_CONTRADICTION_INDEX] for score in scores]


def get_nli_remix_score_batch(pairs: list[tuple[str, str]], to_refine: bool = False) -> list[float]:
    """Return entailment probabilities, optionally zeroing predicted contradictions."""

    if not pairs:
        return []
    import torch

    model = _get_nli_model()
    logits = model.predict(pairs, convert_to_tensor=True, batch_size=BATCH_SIZE)
    probs = torch.softmax(logits, dim=1)
    labels_idx = torch.argmax(probs, dim=1)
    entailment_probs = probs[:, _ENTAILMENT_INDEX]
    if to_refine:
        pair_scores = torch.where(
            labels_idx == _CONTRADICTION_INDEX,
            torch.zeros_like(entailment_probs),
            entailment_probs,
        )
    else:
        pair_scores = entailment_probs
    return pair_scores.detach().cpu().tolist()
