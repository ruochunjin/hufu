"""Build and persist linguistic hypergraphs from text or query instances."""

import hashlib
import os
import re
from pathlib import Path
from time import time
from typing import List, Optional, Union

import spacy
from spacy.symbols import ORTH
from spacy.tokens import Doc
from tqdm import tqdm

from hufu.core.log import getLogger
from hufu.core.query_instance import QueryInstance
from hufu.core.text import clean_text_for_spacy
from hufu.hypergraph.abstraction import TokenEntityAdder
from hufu.hypergraph.combine import calc_correfs_str, combine, combine_links
from hufu.hypergraph.corref import CorrefCluster, mark_corref
from hufu.hypergraph.dependency import Dependency, LocalDoc, Node
from hufu.hypergraph.hypergraph import Hypergraph as LocalHypergraph

_NLP: Optional[spacy.Language] = None
def normalize_special_chars(text: str) -> str:
    """Replace escaped and literal control characters with normalized spaces."""

    special_chars = {
        '\\n': ' ',
        '\\r': ' ',
        '\\t': ' ',
        '\\v': ' ',
        '\\f': ' ',
        '\\b': ' ',
        '\\a': ' ',
        '\n': ' ',
        '\r': ' ',
        '\t': ' ',
        '\v': ' ',
        '\f': ' ',
        '\b': ' ',
        '\a': ' ',
    }
    result = text
    for char, replacement in special_chars.items():
        result = result.replace(char, replacement)
    result = re.sub(r' +', ' ', result)
    return result.strip()
def get_nlp(use_gpu: bool = False) -> spacy.Language:
    """Return the cached spaCy pipeline, loading fast coreference on first use."""

    global _NLP
    if _NLP is None:
        if use_gpu:
            spacy.require_gpu()
        _NLP = spacy.load('en_core_web_trf')
        try:
            import fastcoref  # noqa: F401 - registers the spaCy factory.
        except ImportError:
            fastcoref_available = False
        else:
            fastcoref_available = True
        if fastcoref_available and 'fastcoref' not in _NLP.pipe_names:
            model_path = os.environ.get("HUFU_COREF_MODEL", "biu-nlp/lingmess-coref")
            device = 'cuda' if use_gpu else 'cpu'
            _NLP.add_pipe('fastcoref', config={ 'model_architecture': 'LingMessCoref', 'model_path': model_path, 'device': device})
    _NLP.tokenizer.add_special_case("I.", [{ORTH: "I"}, {ORTH: "."}])
    ROMAN_NUMERALS = ["II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII"]
    for numeral in ROMAN_NUMERALS:
        _NLP.tokenizer.add_special_case(f"{numeral}.", [{ORTH: numeral}, {ORTH: "."}])
    return _NLP
def text_to_doc(text: str) -> Doc:
    """Parse text with coreference resolution and fall back to plain spaCy."""

    logger = getLogger(__name__)
    nlp = get_nlp()
    if "fastcoref" in nlp.pipe_names:
        try:
            cfg = {"fastcoref": {'resolve_text': True}}
            doc = nlp(text, component_cfg=cfg)
            if hasattr(doc._, "coref_clusters") and doc._.coref_clusters:
                for cluster in doc._.coref_clusters:
                    if cluster is None:
                        raise ValueError("coref_clusters contains None")
                    for span in cluster:
                        if span is None or not isinstance(span, (list, tuple)) or len(span) != 2:
                            raise ValueError(f"Invalid span in coref_clusters: {span}")
            return doc
        except Exception as e:
            logger.warning(
                f"[text_to_doc] fastcoref failed for text (len={len(text)}): {type(e).__name__}: {e}, "
                f"falling back to no coref"
            )
    doc = nlp(text)
    return doc
def doc_to_hypergraph(doc: Doc, text: str, is_query: bool = False) -> LocalHypergraph:
    """Convert a parsed document into the production linguistic hypergraph."""

    correfs = calc_correfs_str(doc) if hasattr(doc._, "coref_clusters") else set()
    abstractor = TokenEntityAdder("qwen_ontology_mapping.json")
    t1 = time()
    # Retokenization must precede dependency conversion so every selected span
    # becomes exactly one node in the downstream graph representation.
    links_to_merge = combine_links(doc)
    with doc.retokenize() as retokenizer:
        for link in links_to_merge:
            retokenizer.merge(link)
    t2 = time()
    print(f"combine_links and retokenize took {t2 - t1:.2f} seconds")
    corref_clusters = CorrefCluster.from_doc(doc)
    spans_to_merge = combine(doc, correfs, is_query=is_query, corefs_clusters=corref_clusters)
    abstractor.set_entity_from_spans(spans_to_merge, doc)
    with doc.retokenize() as retokenizer:
        for span in spans_to_merge:
            if span.start < span.end:
                retokenizer.merge(span)
    t3 = time()
    print(f"combine and retokenize took {t3 - t2:.2f} seconds")
    corref_clusters = CorrefCluster.update_by_doc(corref_clusters, doc)
    nodes, roots = Node.from_doc(doc, abstractor)
    nodes = mark_corref(nodes, corref_clusters)
    local_doc = LocalDoc(doc)
    dep = Dependency(nodes, roots, local_doc, is_query=is_query)
    t4 = time()
    print(f"Dependency construction took {t4 - t3:.2f} seconds")
    vertices, rels, id_map = (
        dep.solve_conjunctions().mark_pronoun_antecedents().mark_prefixes().mark_vertex().compress_dependencies().calc_relationships()
    )
    t5 = time()
    print(f"Dependency processing took {t5 - t4:.2f} seconds")
    hypergraph = LocalHypergraph.from_rels(vertices, rels, id_map, local_doc)
    hypergraph.original_text = text
    return hypergraph
def text_to_hypergraph(text: str, is_query: bool = False) -> LocalHypergraph:
    """Clean and parse text, then return its linguistic hypergraph."""

    text = clean_text_for_spacy(text)
    doc = text_to_doc(text)
    return doc_to_hypergraph(doc, text, is_query=is_query)
def generate_instance_id(query: str) -> str:
    """Return the historical whitespace-insensitive identifier for a query."""

    normalized = ''.join(query.split()).lower()
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]
def build_hypergraph_for_query_instance(
    query_instance: QueryInstance,
    dataset_name: str = "hotpotqa",
    base_dir: Union[str, Path] = "data/hypergraph",
    force_rebuild: bool = False
) -> str:
    """Build one query and its contexts under a deterministic instance directory."""

    instance_id = generate_instance_id(query_instance.query)
    instance_dir = Path(base_dir) / dataset_name / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = instance_dir / "metadata.json"
    if metadata_path.exists() and not force_rebuild:
        return str(instance_dir.resolve())
    query_path = instance_dir / "query.pkl"
    if not query_path.exists() or force_rebuild:
        query_hg = text_to_hypergraph(query_instance.query, is_query=True)
        query_hg.save(str(query_path))
    for idx, doc_text in enumerate(query_instance.data):
        data_path = instance_dir / f"data_{idx}.pkl"
        if data_path.exists() and not force_rebuild:
            continue
        data_hg = text_to_hypergraph(doc_text)
        data_hg.save(str(data_path))
    metadata = {
        "instance_id": instance_id,
        "num_data": len(query_instance.data),
        "data_lengths": [len(d) for d in query_instance.data],
    }
    with open(metadata_path, 'w') as f:
        import json
        json.dump(metadata, f)
    return str(instance_dir.resolve())
def test_build_hypergraph_for_query_instance(query_instance: QueryInstance) -> tuple[LocalHypergraph, List[LocalHypergraph]]:
    """Build an instance in memory and record untyped noun mentions for analysis."""

    query_hg = text_to_hypergraph(query_instance.query, is_query=True)
    data_list = [text_to_hypergraph(doc_text, is_query=False) for doc_text in query_instance.data]
    with open("missing.txt", "a") as f:
        for h in [query_hg] + data_list:
            for v in h.vertices:
                if not v.is_noun():
                    continue
                if v.has_entity():
                    continue
                f.write(f"{v.text().strip().lower()}\n")
    return query_hg, data_list
def build_hypergraph_batch(
    query_instances: List[QueryInstance],
    dataset_name: str = "hotpotqa",
    base_dir: Union[str, Path] = "data/hypergraph",
    force_rebuild: bool = False
) -> List[str]:
    """Build and persist a sequence of query instances on the default device."""

    instance_dirs = []
    for qi in tqdm(query_instances, desc="Building hypergraphs", position=1, leave=False):
        instance_dir = build_hypergraph_for_query_instance(
            qi, dataset_name, base_dir, force_rebuild
        )
        instance_dirs.append(instance_dir)
    return instance_dirs
def build_hypergraph_batch_gpu(
    query_instances: List[QueryInstance],
    dataset_name: str = "hotpotqa",
    base_dir: Union[str, Path] = "data/hypergraph",
    force_rebuild: bool = False,
    batch_size: int = 16
) -> List[str]:
    """Batch spaCy parsing on GPU while preserving per-instance output layout."""

    logger = getLogger(__name__)
    tasks = []
    instance_dirs = []
    instances_to_write_metadata = []
    for qi in query_instances:
        instance_id = generate_instance_id(qi.query)
        instance_dir = Path(base_dir) / dataset_name / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        instance_dirs.append(str(instance_dir.resolve()))
        metadata_path = instance_dir / "metadata.json"
        if metadata_path.exists() and not force_rebuild:
            continue
        instances_to_write_metadata.append((instance_dir, qi))
        query_path = instance_dir / "query.pkl"
        if not query_path.exists() or force_rebuild:
            tasks.append({
                'path': query_path,
                'text': clean_text_for_spacy(qi.query),
                'original_text': qi.query,
                'is_query': True
            })
        for idx, doc_text in enumerate(qi.data):
            data_path = instance_dir / f"data_{idx}.pkl"
            if not data_path.exists() or force_rebuild:
                tasks.append({
                    'path': data_path,
                    'text': clean_text_for_spacy(doc_text),
                    'original_text': doc_text,
                    'is_query': False
                })
    if not tasks:
        return instance_dirs
    nlp = get_nlp(use_gpu=True)
    texts = [t['text'] for t in tasks]
    cfg = {"fastcoref": {'resolve_text': True}} if "fastcoref" in nlp.pipe_names else {}
    logger.info(f"Processing {len(texts)} texts with batch_size={batch_size} on GPU...")
    docs = nlp.pipe(texts, component_cfg=cfg, batch_size=batch_size)
    import json
    for task, doc in tqdm(zip(tasks, docs), total=len(tasks), desc="Building hypergraphs (GPU)", position=1, leave=False):
        try:
            hg = doc_to_hypergraph(doc, task['original_text'], is_query=task['is_query'])
            hg.save(str(task['path']))
        except Exception as e:
            logger.error(f"Failed to build hypergraph for {task['path']}: {e}")
    for instance_dir, qi in instances_to_write_metadata:
        metadata_path = instance_dir / "metadata.json"
        instance_id = generate_instance_id(qi.query)
        metadata = {
            "instance_id": instance_id,
            "num_data": len(qi.data),
            "data_lengths": [len(d) for d in qi.data],
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    return instance_dirs
