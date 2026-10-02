# HuFu 虎符

> *虎符相合，方为可信。* — In ancient China, a tiger tally was split in two;
> only when the halves matched was an order authentic. **HuFu** does the same
> for data in the LLM era: it checks whether two pieces of data *fit
> together* — a query and a retrieved passage, a memory and a memory, a
> statement and a document.

HuFu is a **neuro-symbolic data-quality toolkit**. Vector similarity says two
texts are *close*; HuFu decides whether they are *consistent* — with
explainable evidence, provable polynomial-time complexity, and no LLM
required on the decision path. Neural models (embeddings, NLI, optional LLMs)
are used only where they help; symbolic structure (linguistic hypergraphs,
hypergraph simulation, quality rules) makes the final call.

## What it does today

**Unified quality-rule language (v0.2).** One JSON rule schema covering the
CDR / RICO / OMR rule systems — logic predicates, NLI predicates, similarity
predicates, numeric/temporal predicates, all pluggable via a registry. Rules
are data: hand-write them, version them, or (upcoming) discover them:

```python
import hufu

conflicts = hufu.detect_memory_conflicts([
    ("Turn on the light",  {"object": "light", "action": "on"}),
    ("Turn off the light", {"object": "light", "action": "off"}),
    ("Water the plants",   {"object": "plants", "action": "water"}),
])
for c in conflicts:
    print(c.labels, c.explanations)
# → ('conflict',) ("cdr.action_conflict → conflict [attr_equal=1.000(pass), …]",)
```

Attributes can be supplied directly, or extracted by an LLM endpoint
(`enrich_with_llm=True`); an NLI fallback rule covers fully unstructured
text. Write your own rules in the same schema — see
`src/hufu/scenarios/rulesets/memory_conflicts.json`.

**RAG retrieval post-checker (v0.1).** High-cosine retrieved passages can still
semantically contradict the query (temporal, entity, relational, cross-document
conflicts). HuFu sits between retrieval and generation in any RAG pipeline —
it is vector-database agnostic — and labels every retrieved document
`consistent / conflict / useless / irrelevant`, with per-pair evidence:

```python
import hufu

verdicts = hufu.check_retrieval(
    "Where was the author of 'The Old Man and the Sea' born?",
    [
        "Ernest Hemingway, who wrote The Old Man and the Sea, was born in Oak Park, Illinois.",
        "Paris is the capital of France and a major European city.",
    ],
)
for v in verdicts:
    print(v.consistent, v.distance, v.evidence)
```

**Verified reference core, dependency-free.** The typed `run_hypermatch`
pipeline (HC enumeration → D-match → fixed point → document marking) runs
with caller-injected backends and has zero third-party dependencies:

```python
from hufu import (
    CallableRoleCellScorer, Cluster, Hyperedge, Hypergraph,
    Role, RoleScoringResult, Vertex, run_hypermatch,
)
# Build typed query/data hypergraphs, inject any embedding backend and role
# scorer (a callable suffices), and get an immutable HyperMatchOutput back.
# See tests/test_engine_core.py for a complete, runnable example.
```

## Roadmap

- **v0.1** — HyperMatch engine: reference core + text-level checking + Rust accelerator.
- **v0.2** — Unified quality-rule DSL (CDR / RICO / OMR syntax), pluggable predicate library, memory-conflict scenario. *(this release)*
- **v0.3** — Vector-store / agent-memory index trimming (HNSW); LLM-guided rule discovery.
- **v0.4** — Intra-/cross-document contradiction detection with evidence provenance; training-data outlier management (good / bad / ugly).

## Installation

The reference core needs nothing but Python ≥ 3.10:

```bash
pip install -e .            # dependency-free reference core
pip install -e ".[models]"  # + text-level checking (spaCy, sentence-transformers, fastcoref)
pip install -e ".[llm]"     # + LLM-assisted features (OpenAI-compatible endpoint)
python -m spacy download en_core_web_trf
```

Text-level checking also uses the Rust fixed-point accelerator
(`simulation-alg`); with a Rust toolchain:

```bash
pip install ./rust/simulation
```

### Model configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `HUFU_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | sentence embedding model |
| `HUFU_NLI_MODEL` | `cross-encoder/nli-deberta-v3-base` | NLI cross-encoder |
| `HUFU_COREF_MODEL` | `biu-nlp/lingmess-coref` | coreference model |
| `HUFU_SPACY_MODEL` | `en_core_web_trf` | spaCy pipeline |
| `HUFU_API_KEY` / `HUFU_BASE_URL` / `HUFU_MODEL` | — | OpenAI-compatible endpoint for LLM-assisted features |

## Repository layout

```text
src/hufu/
  api.py            Public facade: check_pair / check_retrieval + reference core
  backends.py       Model-backed EmbeddingBackend adapters (lazy)
  engine/           HyperMatch algorithm core (contracts, HC, D-match, fixed point, marking)
  rules/            Unified quality-rule DSL: model, registry, predicates, engine
  scenarios/        Ready-made scenarios (memory conflicts) + bundled rule sets
  hypergraph/       Linguistic hypergraph construction, traversal, fusion
  graphgen/         LLM-assisted graph extraction (optional)
  core/             QueryInstance, logging, text utilities
  llm/              Unified OpenAI-compatible client (optional)
rust/
  graph-simulation/ Rust graph-simulation crate
  simulation/       maturin Python binding (simulation-alg)
tests/              Dependency-free engine and rule tests (runnable anywhere)
examples/           Quickstart scripts
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
% Catching Inconsistencies in Vector Similarity Retrieval (HyperMatch)
% Conflict-Aware Memory for Embodied Agents: Enhancing Vector Data Quality via Detection Rules (CDRs, ACL 2026)
% Detecting Contradictions Within and Across Documents (DocChecker)
% Outliers: The Good, the Bad and the Ugly (OHunt)
```

(Full BibTeX entries will be added with publication metadata.)

## License

Apache-2.0 (see LICENSE). Existing source copyright notices are retained.
