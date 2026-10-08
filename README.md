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

**Memory conflicts, conflict inference & calibration (v0.5).**
`detect_memory_conflicts` applies CDR-style rules (opposing actions/states
on the same object + VecSim) with explicit opposite-pair tables — pure
structure, no model on the decision path for attributed memories; the NLI
fallback is gated to unstructured text only.  Two CDR-paper algorithms join
the toolkit: `infer_conflicts_from_judgments` (the Trimming Theorem — items
whose consistency verdicts diverge against the same query conflict with
each other) and `learn_threshold_youden` (ROC/Youden threshold calibration
for similarity predicates, pure stdlib).

**Outlier detection, classification & repair (v0.5).** Beyond the v0.4
measured-impact classifier, HuFu now ports OHunt's exact semantics:
`GapIsolationDetector` (gap-of-gaps normalization — robust to clustered
data), `classify_outliers_hinge` (the paper's hinge three-way: full-margin
outliers are *good*, in-margin *ugly*, misclassified *bad*), the repair
family `repair_outliers` (KNN label repair — the paper's best —, RF label,
mean/median value, KNN feature vote, delete), and `discover_outlier_pipeline`
(OLearner-lite: grid search over detector θ × repair method with your
downstream metric as the objective).  sklearn/faiss are optional extras —
injected or lazily imported; the core stays dependency-free.

**Document contradictions & training-data outliers (v0.4).**
`detect_contradictions` checks statement pairs within and across documents
against RICO-style rules (entity/value/polarity conflicts + NLI fallback),
and every finding carries evidence provenance — document/sentence positions
plus the full rule trace. `classify_outliers` implements the
good / bad / ugly philosophy: outliers are classified by their *measured
effect* on your downstream model (injected as a plain callable); ugly ones
are repaired (never silently deleted), good ones (repairing would hurt) are
kept, harmless ones are left alone.

**Conflict-aware trimming & rule discovery (v0.3).** Once conflicts are
detected, HuFu resolves them — at evidence level (dependency-free greedy
cover, chains like A↔B↔C drop only B) and, for faiss HNSW indexes, at the
graph level by cutting edges between conflicting vectors so retrieval stops
surfacing conflicts. And rules don't have to be hand-written:
`hufu.rules.discovery.discover_rules` has an LLM propose candidate rules
from labeled pairs, then keeps only those passing coverage + accuracy
thresholds — offline discovery, LLM-free runtime.

```python
from hufu.scenarios import trim_evidence

result = trim_evidence(memories, [(0, 1), (1, 2)], strategy="keep_recent",
                       items_attrs=[{"timestamp": t} for t in timestamps])
clean_memories = result.apply(memories)
```

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

## Scope and limitations

HuFu's document checker is deliberately **precision-first**: it aims to
*certify* structural contradictions with explainable evidence, not to cover
every semantic nuance. What that means in practice, measured on the
300-document golden corpus (`tests/e2e/golden_doc300.py`: hand-built
batteries + stratified ContraDoc positives/negatives + synthetic hard
negatives):

- **Reliable subdomain (default `profile="strict"`)**: conflicts provable
  from structure alone — typed value-slot swaps (born in X vs Y, 300 vs
  3000 men, May vs June 1883), role reversals on asymmetric verbs
  (sold/hired/bit), antonym pairs (open/closed, increase/decrease), and
  polarity flips on an otherwise identical frame (attended vs did not
  attend). These rules never consult a model; every verdict ships with the
  full decision chain.
- **Deliberately out of scope for the strict verdict**: narrative tension,
  perspective/emotion shifts, paraphrase-level disagreement, and temporal
  *progressions* (a story legitimately giving the same attribute different
  values at different times). Real long-form narrative contains the same
  surface signatures as injected conflicts — clean ContraDoc stories trip
  structural rules at high rates — so those channels (similarity+NLI cues,
  weak negation cues) are computed and recorded in the decision chain but
  do not change the strict verdict.
- *****REMOVED***: deprecated (2026-10-08, measured).** The `full` profile +
  `***REMOVED***=True` path (local ***REMOVED*** or API ***REMOVED*** gating the cued channels)
  is kept for reproducibility but is **not recommended**: the local ***REMOVED***
  judge over-confirms on narrative (precision 0.523 — a recall amplifier,
  not a gate), the API judge is slow (~76 s/doc) yet still cannot fix the
  cue channels' signal overload on clean narrative, and neither generalizes
  to unseen domains. The strict, structure-only verdict is the supported
  decision path.

In short: use HuFu to *confirm* contradictions on structured /
semi-structured content (knowledge bases, fact tables, records, curated
documents), as a consistency post-checker for RAG retrieval, and for
vector-memory conflict detection — always with full provenance. Long-form
narrative contradiction hunting is explicitly out of scope.

Measured on the frozen 300-document corpus (2026-10-07):

| Mode | Correct | Precision | Recall | F1 | Wall time |
|---|---|---|---|---|---|
| rules-only (`strict`) | 203/300 | 0.636 | 0.669 | 0.652 | 12 min |
| local ***REMOVED*** ***REMOVED*** (`full`) | 176/300 | 0.523 | 0.985 | 0.684 | 3.4 h |
| Claude-API ***REMOVED*** (`full`) | 201/300 | 0.593 | 0.868 | 0.704 | 6.3 h |

Strict mode is the only configuration whose false-flag rate is controlled
(50/50 synthetic hard negatives, 14/15 hand-built clean docs); the local
***REMOVED*** judge over-confirms and is a recall amplifier, not a precision gate;
the API ***REMOVED*** gives the best recall/F1 but cannot fix the structural
rules' signal overload on clean narrative — that residual is a methods
boundary, not an engineering gap.

## Roadmap

- **v0.1** — HyperMatch engine: reference core + text-level checking + Rust accelerator.
- **v0.2** — Unified quality-rule DSL (CDR / RICO / OMR syntax), pluggable predicate library, memory-conflict scenario.
- **v0.3** — Conflict-aware trimming (evidence level + HNSW edges); LLM-guided rule discovery with coverage/accuracy validation.
- **v0.4** — Document contradiction detection with evidence provenance; training-data outlier management (good / bad / ugly).
- **v0.5** — Full CDRs/OHunt integration *(this release)*: opposite-pair rule semantics + NLI fallback gating + softmax-probability fix; Trimming-Theorem witness conflict inference; Youden threshold calibration; gap-isolation detector; hinge three-way classification; repair family (KNN/RF label, mean/median, KNN features, delete); OLearner-lite pipeline discovery. ***REMOVED*** path deprecated (measured: slow, imprecise, poor generalization).

Future directions: tighter coupling between the HyperMatch structural verdicts
and the rule layer (structural predicates as rule conditions), prebuilt
adapters for popular vector stores, and benchmark suites (ContraDoc, ECON,
ConTRoL, ContractNLI).

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
