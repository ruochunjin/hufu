"""Text normalization helpers for hypergraph construction."""

import re


def clean_text_for_spacy(text: str) -> str:
    """Normalize whitespace, citation markers, and dash variants for spaCy."""

    if not text:
        return ""
    text = re.sub(r'\[n\s+\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[–—―]', '-', text)
    return text
