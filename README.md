# HuFu 虎符

> *虎符相合，方为可信。* — In ancient China, a tiger tally was split in two;
> only when the halves matched was an order authentic. **HuFu** does the same
> for data: it checks whether two pieces of data *fit together* — a query and
> a retrieved passage, a memory and a memory, a statement and a document.

HuFu is a **lightweight consistency checker**. Vector similarity says two
texts are *close*; HuFu decides whether they are *consistent* — with
explainable evidence and provable polynomial-time complexity. Symbolic
structure (linguistic hypergraphs, hypergraph simulation, quality rules)
makes the call; small neural models (embeddings, NLI) serve only as
measurable inputs to the rules.

## Quick start

```bash
pip install hufu            # dependency-free reference core
pip install "hufu[models]"  # text-level checking (spaCy, embeddings, NLI)
python -m spacy download en_core_web_trf
```

```python
import hufu

# 1. RAG retrieval post-check: does each retrieved passage fit the query?
verdicts = hufu.check_retrieval(
    "Where was the author of 'The Old Man and the Sea' born?",
    [
        "Ernest Hemingway, who wrote The Old Man and the Sea, was born in Oak Park, Illinois.",
        "Paris is the capital of France and a major European city.",
    ],
)
for v in verdicts:
    print(v.consistent, v.distance)

# 2. Vector-memory conflicts: same object, opposing actions?
conflicts = hufu.detect_memory_conflicts([
    ("Turn on the light",  {"object": "light", "action": "on"}),
    ("Turn off the light", {"object": "light", "action": "off"}),
    ("Water the plants",   {"object": "plants", "action": "water"}),
])
print(conflicts[0].labels, conflicts[0].explanations)

# 3. Training-data outliers: good / bad / ugly by measured impact
from hufu.scenarios import classify_outliers
reports = classify_outliers(records, labels, evaluator=train_and_score)
```

Every verdict carries evidence provenance and the full decision chain —
HuFu explains *why* it flags something.

## What it does

- **RAG retrieval post-checker** — sits between retrieval and generation in
  any pipeline, vector-database agnostic; labels every retrieved document
  `consistent / conflict` with per-pair evidence and distance.
- **Document contradiction detection** — star-graph rules over a document's
  own statements: typed value-slot swaps, numeric mismatches (with unit
  conversion), role reversals, antonym pairs, polarity flips.
- **Vector-memory conflict detection & trimming** — CDR-style rules for
  memory banks (same object, opposing actions/states/values), conflict
  inference from retrieval judgments, and evidence/index-level trimming so
  retrieval stops surfacing conflicts.
- **Training-data outlier management** — detectors, the good / bad / ugly
  classification by measured downstream impact, and a repair family
  (KNN/RF label repair, value imputation, deletion) with a small pipeline
  search.
- **Unified quality-rule language** — one JSON rule schema (logic, NLI,
  similarity, numeric/temporal predicates), pluggable via a registry.
  Rules are data: hand-write, version, or discover them offline.

## Scope and limitations

HuFu is deliberately **precision-first**: it certifies structural
contradictions with explainable evidence rather than chasing every semantic
nuance.

- **Reliable subdomain (default `profile="strict"`)**: conflicts provable
  from structure alone — typed value-slot swaps (born in X vs Y, 300 vs
  3000 men, May vs June 1883), role reversals on asymmetric verbs, antonym
  pairs, and polarity flips on an otherwise identical frame.
- **Out of scope**: narrative tension, perspective/emotion shifts,
  paraphrase-level disagreement, and temporal *progressions* (a story
  legitimately giving the same attribute different values at different
  times). Real long-form narrative contains the same surface signatures as
  injected conflicts — a methods boundary, not an engineering gap.

In short: use HuFu to *confirm* contradictions on structured /
semi-structured content (knowledge bases, fact tables, records, curated
documents), as a consistency post-checker for RAG retrieval, and for
vector-memory conflict detection. Long-form narrative contradiction
hunting is explicitly out of scope.

## Installation

```bash
pip install hufu               # dependency-free reference core (Python ≥ 3.10)
pip install "hufu[models]"     # + text-level checking (spaCy, sentence-transformers, fastcoref)
pip install "hufu[outliers]"   # + sklearn-backed outlier classification/repair
python -m spacy download en_core_web_trf
```

Text-level checking can also use the Rust fixed-point accelerator
(`simulation-alg`) when a Rust toolchain is available:

```bash
pip install ./rust/simulation
```

### Model configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `HUFU_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | sentence embedding model |
| `HUFU_NLI_MODEL` | `cross-encoder/nli-deberta-v3-xsmall` | NLI cross-encoder |
| `HUFU_COREF_MODEL` | `biu-nlp/lingmess-coref` | coreference model (`HUFU_COREF=off` disables) |
| `HUFU_SPACY_MODEL` | `en_core_web_trf` | spaCy pipeline |

## Repository layout

```text
src/hufu/
  api.py            Public facade: check_pair / check_retrieval + reference core
  backends.py       Model-backed EmbeddingBackend adapters (lazy)
  engine/           HyperMatch core + star-graph document engine + bridge
  rules/            Unified quality-rule DSL: model, registry, predicates, engine
  scenarios/        Ready-made scenarios (memory, documents, outliers, trim)
  hypergraph/       Linguistic hypergraph construction, traversal, fusion
  core/             QueryInstance, logging, text utilities
rust/
  graph-simulation/ Rust graph-simulation crate
  simulation/       maturin Python binding (simulation-alg)
tests/              Dependency-free engine and rule tests (runnable anywhere)
```

## Development

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -v   # no models or network needed
ruff check src tests
```

## Citation

HuFu integrates several peer-reviewed works. If you use it, please cite:

```bibtex
@inproceedings{huang2027catching,
  title={Catching Inconsistencies in Vector Similarity Retrieval},
  author={Huang, Zixiao and Jin, Ruochun and Wang, Haotian and Zhang, Meng and Tang, Yuhua},
  booktitle={Proceedings of the ACM on Management of Data (SIGMOD 2027)},
  year={2027},
  note={Corresponding author: Ruochun Jin}
}

@inproceedings{chen2027detecting,
  title={Detecting Contradictions Within and Across Documents},
  author={Chen, Shenglin and Fan, Wenfei and Jin, Ruochun and Ma, Kexin},
  booktitle={Proceedings of the ACM on Management of Data (SIGMOD 2027)},
  year={2027},
  note={Corresponding author: Ruochun Jin}
}

@inproceedings{ma2026conflict,
  title={Conflict-Aware Memory for Embodied Agents: Enhancing Vector Data Quality via Detection Rules},
  author={Ma, Kexin and Wang, Haotian and Chen, Shenglin and Cai, Yishuai and Jin, Ruochun and others},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={28328--28347},
  year={2026}
}

@article{chen2025outliers,
  title={Outliers: The Good, the Bad and the Ugly},
  author={Chen, Shenglin and Fan, Wenfei and Jin, Ruochun},
  journal={Proceedings of the ACM on Management of Data},
  volume={3},
  number={4},
  pages={1--30},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```

## License

Apache-2.0 (see LICENSE). Existing source copyright notices are retained.
