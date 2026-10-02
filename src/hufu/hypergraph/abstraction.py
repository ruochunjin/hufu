"""Map spaCy spans and tokens onto the project's coarse entity ontology."""

import json

from nltk.corpus import wordnet as wn
from spacy.tokens import Span, Token

from hufu.hypergraph.entity import ENT
from hufu.hypergraph.linguistic import Entity


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

    def __init__(self, path: str):
        self.char_index_to_entity: dict[int, ENT] = {}
        self.mapping = {}
        with open(path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)
    def _spacy_to_wn_pos(self, spacy_pos):
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
            synsets = wn.synsets(span.text.lower().strip(), pos=wn_pos)
            if synsets:
                synset = synsets[0]
        if not synset:
            return []
        paths = synset.hypernym_paths()
        if not paths:
            return [synset.name()]
        return [node.name() for node in paths[0]]
    def _get_contextual_synset_path_token(self, token: Token, doc) -> list[str]:
        wn_pos = self._spacy_to_wn_pos(token.pos_)
        if not wn_pos:
            return []
        try:
            synset = _cosine_lesk(doc.text, token.text, pos=wn_pos)
        except Exception:
            synset = None
        if not synset:
            synsets = wn.synsets(token.lemma_, pos=wn_pos)
            if synsets:
                synset = synsets[0]
        if not synset:
            return []
        paths = synset.hypernym_paths()
        if not paths:
            return [synset.name()]
        return [node.name() for node in paths[0]]
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
