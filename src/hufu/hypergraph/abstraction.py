"""Map spaCy spans and tokens onto the project's coarse entity ontology."""

import json
import os
from functools import lru_cache
from importlib.resources import files as resource_files

from spacy.tokens import Span, Token

from hufu.core.log import getLogger
from hufu.hypergraph.entity import ENT
from hufu.hypergraph.linguistic import Entity

logger = getLogger(__name__)


@lru_cache(maxsize=1)
def _wordnet():
    """Import WordNet lazily so minimal installs stay nltk-free.

    Returns ``None`` when nltk (or its wordnet corpus) is unavailable; callers
    then simply skip the hypernym fallback and rely on NER labels only.
    """

    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        logger.warning("nltk not installed; WordNet entity abstraction disabled")
        return None
    try:
        wn.ensure_loaded()
    except LookupError:
        logger.warning(
            "nltk wordnet corpus missing; run `python -m nltk.downloader wordnet`. "
            "WordNet entity abstraction disabled"
        )
        return None
    return wn


@lru_cache(maxsize=20000)
def _cached_synsets(word: str, wn_pos) -> tuple:
    """WordNet synsets for (word, pos) — the build pipeline asks this per
    span and per token, and hypernym_paths() below costs ~ms per call."""

    wn = _wordnet()
    if wn is None:
        return ()
    return tuple(wn.synsets(word, pos=wn_pos))


@lru_cache(maxsize=20000)
def _cached_hypernym_path(synset_name: str) -> tuple:
    """First hypernym path for a synset, as names root→synset."""

    wn = _wordnet()
    if wn is None:
        return ()
    try:
        synset = wn.synset(synset_name)
    except Exception:  # noqa: BLE001 - unknown synset name
        return ()
    paths = synset.hypernym_paths()
    if not paths:
        return (synset.name(),)
    return tuple(node.name() for node in paths[0])


def default_mapping_path() -> str:
    """Resolve the synset-to-ENT mapping path.

    ``HUFU_ONTOLOGY_MAPPING`` wins; otherwise the bundled seed mapping is
    used.  The upstream LLM-generated ``qwen_ontology_mapping.json`` is not
    redistributable, so the seed (top-level WordNet synsets) ships instead.
    """

    override = os.environ.get("HUFU_ONTOLOGY_MAPPING")
    if override:
        return override
    return str(resource_files("hufu.hypergraph").joinpath("ontology_mapping_seed.json"))


def _cosine_lesk(context: str, word: str, *, pos):
    """Run optional PyWSD disambiguation without import-time corpus loading."""

    try:
        from pywsd.lesk import cosine_lesk
    except (ImportError, LookupError):
        return None
    try:
        return cosine_lesk(context, word, pos=pos)
    except LookupError:
        return None


class TokenEntityAdder:
    """Assign coarse entity types using NER labels and WordNet hypernyms."""

    def __init__(self, path: str | None = None):
        self.char_index_to_entity: dict[int, ENT] = {}
        self.mapping = {}
        resolved = path or default_mapping_path()
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                self.mapping = {
                    key: value
                    for key, value in json.load(f).items()
                    if not key.startswith("_")
                }
        except FileNotFoundError:
            logger.warning(
                f"ontology mapping {resolved!r} not found; WordNet typing disabled"
            )
    def _spacy_to_wn_pos(self, spacy_pos):
        wn = _wordnet()
        if wn is None:
            return None
        if spacy_pos in ['NOUN', 'PROPN']: return wn.NOUN
        if spacy_pos == 'VERB': return wn.VERB
        if spacy_pos == 'ADJ': return wn.ADJ
        if spacy_pos == 'ADV': return wn.ADV
        return None
    def _get_contextual_synset_path_span(self, span: Span, doc) -> list[str]:
        wn_pos = self._spacy_to_wn_pos(span.root.pos_)
        if not wn_pos:
            return []
        try:
            synset = _cosine_lesk(doc.text, span.text, pos=wn_pos)
        except Exception:
            synset = None
        if not synset:
            synsets = _cached_synsets(span.text.lower().strip(), wn_pos)
            if synsets:
                synset = synsets[0]
        if not synset:
            return []
        return list(_cached_hypernym_path(synset.name()))
    def _get_contextual_synset_path_token(self, token: Token, doc) -> list[str]:
        wn_pos = self._spacy_to_wn_pos(token.pos_)
        if not wn_pos:
            return []
        try:
            synset = _cosine_lesk(doc.text, token.text, pos=wn_pos)
        except Exception:
            synset = None
        if not synset:
            synsets = _cached_synsets(token.lemma_, wn_pos)
            if synsets:
                synset = synsets[0]
        if not synset:
            return []
        return list(_cached_hypernym_path(synset.name()))
    def set_entity_from_spans(self, spans: list[Span], doc):
        """Resolve and cache an entity type for every selected span start."""

        # Explicit spaCy NER labels take precedence; WordNet abstraction is the
        # fallback only for spans that have no named-entity annotation.
        for span in spans:
            if span.label_:
                self.char_index_to_entity[span.start_char] = ENT.from_entity(Entity[span.label_])
                continue
            synset = self._get_contextual_synset_path_span(span, doc)
            is_concept = False
            loop_end = False
            for syn in reversed(synset):
                if syn in self.mapping:
                    mapped_entity = self.mapping[syn]
                    if mapped_entity == "CONCEPT":
                        is_concept = True
                        continue
                    elif mapped_entity == "NOT_ENT":
                        continue
                    self.char_index_to_entity[span.start_char] = ENT[mapped_entity]
                    loop_end = True
                    break
            if loop_end:
                continue
            loop_end = False
            synset = self._get_contextual_synset_path_token(span.root, doc)
            for syn in reversed(synset):
                if syn in self.mapping:
                    mapped_entity = self.mapping[syn]
                    if mapped_entity == "CONCEPT":
                        is_concept = True
                        continue
                    elif mapped_entity == "NOT_ENT":
                        continue
                    self.char_index_to_entity[span.start_char] = ENT[mapped_entity]
                    loop_end = True
                    break
            if loop_end:
                continue
            if is_concept:
                self.char_index_to_entity[span.start_char] = ENT.CONCEPT
                continue
            self.char_index_to_entity[span.start_char] = ENT.NOT_ENT
    def get_entity_for_char_index(self, char_index: int) -> ENT | None:
        """Return the cached entity type at a span's starting character."""

        return self.char_index_to_entity.get(char_index, None)
    def get_entity_for_token(self, token: Token, doc) -> ENT | None:
        """Infer a coarse entity type for one token from its hypernym path."""

        synset = self._get_contextual_synset_path_token(token, doc)
        is_concept = False
        for syn in reversed(synset):
            if syn in self.mapping:
                mapped_entity = self.mapping[syn]
                if mapped_entity == "CONCEPT":
                    is_concept = True
                    continue
                elif mapped_entity == "NOT_ENT":
                    return None
                else:
                    return ENT[mapped_entity]
        if is_concept:
            return ENT.CONCEPT
        return None
