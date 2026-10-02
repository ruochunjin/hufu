"""Text-level RAG post-checker example (requires the `models` extra).

Setup:
    pip install -e ".[models]"
    python -m spacy download en_core_web_trf
    pip install ./rust/simulation        # Rust fixed-point accelerator

Run from the repository root:  python examples/quickstart_text.py
"""

import hufu

QUERY = "Where was the author of 'The Old Man and the Sea' born?"

# Two passages that a vector retriever might rank as highly similar:
DOCUMENTS = [
    # Consistent: actually answers the query.
    "Ernest Hemingway wrote The Old Man and the Sea. He was born in Oak Park, Illinois, in 1899.",
    # Geometrically close but semantically off: about the book, not the birthplace.
    "The Old Man and the Sea is a 1952 novella set mostly on the sea off the coast of Cuba.",
]

verdicts = hufu.check_retrieval(QUERY, DOCUMENTS)
for verdict in verdicts:
    status = "CONSISTENT " if verdict.consistent else "INCONSISTENT"
    print(f"[{status}] distance={verdict.distance:.4f}")
    print(f"  doc: {verdict.document[:80]}...")
    print(f"  evidence: {verdict.evidence.splitlines()[0] if verdict.evidence else '-'}")
