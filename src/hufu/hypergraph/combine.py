"""Select non-overlapping linguistic spans for spaCy retokenization."""

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans

from hufu.hypergraph.corref import CorrefCluster


def get_level_order(doc: Doc, reversed=False) -> list[Token]:
    """Return tokens grouped by dependency-tree depth."""

    levels: dict[int, list[Token]] = {}
    max_level = 0
    for token in doc:
        level = 0
        current = token
        while current.head != current:
            level += 1
            current = current.head
        if level not in levels:
            levels[level] = []
        levels[level].append(token)
        if level > max_level:
            max_level = level
    ordered_tokens: list[Token] = []
    if reversed:
        for level in range(max_level, -1, -1):
            ordered_tokens.extend(levels.get(level, []))
    else:
        for level in range(0, max_level + 1):
            ordered_tokens.extend(levels.get(level, []))
    return ordered_tokens
def _restrict_correfs(clusters: list[list[tuple[int, int]]], level: int=0) -> list[list[tuple[int, int]]]:
    restricted = []
    for cluster in clusters:
        if cluster is None:
            continue
        cluster = [span for span in cluster if span is not None and isinstance(span, (list, tuple)) and len(span) == 2]
        if not cluster:
            continue
        if level == 0:
            restricted.append(cluster)
        elif level == 1:
            is_sub = False
            for span in cluster:
                if is_sub:
                    break
                for other_span in cluster:
                    if span == other_span:
                        continue
                    if span[0] >= other_span[0] and span[1] <= other_span[1]:
                        is_sub = True
                        break
            if not is_sub:
                restricted.append(cluster)
        elif level == 2:
            has_intersection = False
            for span in cluster:
                if has_intersection:
                    break
                for other_span in cluster:
                    if span == other_span:
                        continue
                    if not (span[1] <= other_span[0] or span[0] >= other_span[1]):
                        has_intersection = True
                        break
            if not has_intersection:
                restricted.append(cluster)
    return restricted
def _left_descendants(token: Token) -> list[Token]:
    descendants = []
    for left in reversed(list(token.lefts)):
        if left.i != token.i - 1:
            break
        descendants.append(left)
        descendants.extend(_left_descendants(left))
    return descendants
def _right_descendants(token: Token) -> list[Token]:
    descendants = []
    for right in token.rights:
        if right.i != token.i + 1:
            break
        descendants.append(right)
        descendants.extend(_right_descendants(right))
    return descendants
def calc_correfs_str(doc: Doc) -> set[str]:
    """Collect valid mention strings from the document's coreference clusters."""

    correfs: set[str] = set()
    clusters = getattr(doc._, "coref_clusters", None)
    if not clusters:
        return correfs
    clusters = [c for c in clusters if c is not None]
    if not clusters:
        return correfs
    text = doc.text
    clusters = _restrict_correfs(clusters, level=1)
    for cluster in clusters:
        for (start, end) in cluster:
            if start is None or end is None or start < 0 or end > len(text):
                continue
            correfs.add(text[start:end])
    correfs.update(_build_coref_span_map(doc).values())
    return correfs
def _build_coref_span_map(doc: Doc) -> dict[tuple[int, int], str]:
    span_map: dict[tuple[int, int], str] = {}
    clusters = getattr(doc._, "coref_clusters", None)
    if not clusters:
        return span_map
    clusters = [c for c in clusters if c is not None]
    if not clusters:
        return span_map
    clusters = _restrict_correfs(clusters, level=1)
    for cluster in clusters:
        if not cluster:
            continue
        if len(cluster) == 0 or cluster[0] is None or len(cluster[0]) != 2:
            continue
        canonical_span = doc.char_span(*cluster[0], alignment_mode="expand")
        if canonical_span is None:
            continue
        canonical_text = canonical_span.text
        for start_char, end_char in cluster:
            if start_char is None or end_char is None:
                continue
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is None:
                continue
            span_map[(span.start, span.end)] = canonical_text
    return span_map
def _calc_same_tokens(doc: Doc, correfs: set[str]) -> dict[str, list[tuple[int, int]]]:
    token_map: dict[str, set[tuple[int, int]]] = {}
    resolved_span_map = _build_coref_span_map(doc)
    n = len(doc)
    for i in range(n):
        for k in range(i + 1, n):
            max_len = 0
            while i + max_len < n and k + max_len < n:
                if doc[i + max_len].text != doc[k + max_len].text:
                    break
                max_len += 1
            if max_len <= 1:
                continue
            left_trim = 0
            right_trim = 0
            while left_trim < max_len and (doc[i + left_trim].pos_ in {"SPACE", "PUNCT"} or doc[k + left_trim].pos_ in {"SPACE", "PUNCT"}):
                left_trim += 1
            while right_trim < max_len - left_trim and (doc[i + max_len - 1 - right_trim].pos_ in {"SPACE", "PUNCT"} or doc[k + max_len - 1 - right_trim].pos_ in {"SPACE", "PUNCT"}):
                right_trim += 1
            span_len = max_len - left_trim - right_trim
            if span_len <= 1:
                continue
            start_i = i + left_trim
            end_i = start_i + span_len
            start_k = k + left_trim
            end_k = start_k + span_len
            span_i = doc[start_i:end_i]
            span_k = doc[start_k:end_k]
            span_text_i = resolved_span_map.get((span_i.start, span_i.end), span_i.text)
            span_text_k = resolved_span_map.get((span_k.start, span_k.end), span_k.text)
            if span_text_i != span_text_k:
                continue
            if len(correfs) > 0 and span_text_i not in correfs:
                continue
            token_map.setdefault(span_text_i, set()).add((span_i.start, span_i.end))
            token_map.setdefault(span_text_i, set()).add((span_k.start, span_k.end))
    token_map_filtered: dict[str, list[tuple[int, int]]] = {}
    for text, positions in token_map.items():
        pos_list = sorted(positions)
        if len(pos_list) > 1 and (pos_list[0][1] - pos_list[0][0]) > 1:
            token_map_filtered[text] = pos_list
    return token_map_filtered
def _calc_bigram_likelihood_scores(doc: Doc) -> dict[tuple[str, str], float]:
    tokens = [token.text.lower() for token in doc]
    if len(tokens) < 2:
        return {}
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(BigramAssocMeasures.likelihood_ratio)
    return {pair: score for pair, score in scored}
def combine_links(doc: Doc) -> list[Span]:
    """Select hyphen-linked spans and expand overlapping named entities."""

    links_to_merge: list[Span] = []
    links_token_idxs: set[int] = set()
    for token in doc:
        if token.text == "-" :
            if token.i - 1 >= 0 and token.i + 1 < len(doc):
                left_token, right_token = doc[token.i - 1], doc[token.i + 1]
                if left_token.dep_ == "conj" or right_token.dep_ == "conj":
                    continue
                span = doc[token.i - 1:token.i + 2]
                if links_token_idxs.intersection(range(span.start, span.end)):
                    continue
                links_to_merge.append(span)
                links_token_idxs.update(range(span.start, span.end))
        elif token.text.endswith("-") and token.i + 1 < len(doc):
            span = doc[token.i:token.i + 2]
            if links_token_idxs.intersection(range(span.start, span.end)):
                continue
            links_to_merge.append(span)
            links_token_idxs.update(range(span.start, span.end))
        elif token.text.startswith("-") and token.i - 1 >= 0:
            span = doc[token.i - 1:token.i + 1]
            if links_token_idxs.intersection(range(span.start, span.end)):
                continue
            links_to_merge.append(span)
            links_token_idxs.update(range(span.start, span.end))
    new_ents = []
    for ent in doc.ents:
        span_start = ent.start
        span_end = ent.end
        for link_span in links_to_merge:
            if not (link_span.end <= span_start or link_span.start >= span_end):
                new_start = min(span_start, link_span.start)
                new_end = max(span_end, link_span.end)
                span_start = new_start
                span_end = new_end
        new_ent = Span(doc, span_start, span_end, label=ent.label_)
        new_ents.append(new_ent)
    doc.set_ents(filter_spans(new_ents), default="unmodified")
    return links_to_merge
def combine(doc: Doc, correfs: set[str]=set(), is_query: bool = False, corefs_clusters: list[CorrefCluster] = []) -> list[Span]:
    """Select entity, phrase, predicate, modifier, and coreference spans to merge."""

    # Span categories are accumulated into a shared token-occupancy set so the
    # final reverse-sorted list is safe for one spaCy retokenization pass.
    spans_to_merge: list[Span] = []
    ent_token_idxs: set[int] = set()
    bigram_lr_scores = _calc_bigram_likelihood_scores(doc)
    lr_threshold = 7.5
    new_ents = []
    for ent in doc.ents:
        if ent.label_ in {"ORDINAL", "CARDINAL"} and len(ent) == 1:
            continue
        if ent_token_idxs.intersection(range(ent.start, ent.end)):
            continue
        span_start = ent.start
        span_end = ent.end
        def is_compound_to_ent(token: Token, ent: Span):
            if token.dep_ == "compound" and token.head in ent:
                return True
            return False
        for i in range(ent.start - 1, -1, -1):
            if is_compound_to_ent(doc[i], ent) and not (is_query and doc[i].pos_ == "PROPN"):
                span_start = i
            else:
                break
        for i in range(ent.end, len(doc)):
            if is_compound_to_ent(doc[i], ent) and not (is_query and doc[i].pos_ == "PROPN"):
                span_end = i + 1
            else:
                break
        if ent_token_idxs.intersection(range(span_start, span_end)) or (span_start == ent.start and span_end == ent.end):
            spans_to_merge.append(ent)
            ent_token_idxs.update(range(ent.start, ent.end))
            continue
        new_ent = Span(doc, span_start, span_end, label=ent.label_)
        new_ents.append(new_ent)
        spans_to_merge.append(new_ent)
        ent_token_idxs.update(range(span_start, span_end))
    doc.set_ents(filter_spans(new_ents), default="unmodified")
    naive_dets = { "the", "a", "an" }
    wh_dets = {"what", "which", "whose", "whichever", "whatever"}
    noun_token_idxs: set[int] = set()
    max_span_tokens = 5
    spans_to_merge_on_noun: dict[tuple[int,int], Span] = {}
    doc_by_level = list(doc)
    for token in doc_by_level:
        if token.pos_ == "NOUN":
            span_start = token.i
            span_end = token.i + 1
            for left in reversed(list(token.lefts)):
                if left.i != span_start - 1:
                    break
                if left.dep_ == "amod":
                    pair = (left.text.lower(), left.head.text.lower())
                    score = bigram_lr_scores.get(pair, 0.0)
                    if score >= lr_threshold:
                        span_start = left.i
                    else:
                        break
                elif left.dep_ in {"advmod", "neg", "nummod", "quantmod", "npadvmod", "compound"} or (left.dep_ == "det" and  left.text.lower() not in naive_dets):
                    if is_query and left.dep_ == "compound" and left.pos_ == "PROPN":
                        break
                    if is_query and left.dep_ == "det" and left.tag_ == "WDT":
                        break
                    span_start = left.i
                else:
                    break
            for right in token.rights:
                if right.i != span_end:
                    break
                if right.dep_ in {"case", "advmod", "neg", "nummod", "quantmod", "npadvmod"}:
                    span_end = right.i + 1
                else:
                    break
            if span_start + 1 == span_end or (span_end - span_start) > max_span_tokens:
                continue
            span = doc[span_start:span_end]
            if noun_token_idxs.intersection(range(span.start, span.end)):
                for start, end in spans_to_merge_on_noun.keys():
                    if not (span.end <= start or span.start >= end):
                        new_start = min(span.start, start)
                        new_end = max(span.end, end)
                        if new_end - new_start > max_span_tokens:
                            break
                        new_span = doc[new_start:new_end]
                        spans_to_merge_on_noun.pop((start, end))
                        spans_to_merge_on_noun[(new_start, new_end)] = new_span
                        noun_token_idxs.update(range(new_start, new_end))
                        break
                continue
            spans_to_merge_on_noun[(span.start, span.end)] = span
            noun_token_idxs.update(range(span.start, span.end))
    for span in spans_to_merge_on_noun.values():
        if ent_token_idxs.intersection(range(span.start, span.end)):
            continue
        spans_to_merge.append(span)
        ent_token_idxs.update(range(span.start, span.end))
    for token in doc:
        if token.pos_ == "VERB":
            span_start = token.i
            span_end = token.i + 1
            for left in _left_descendants(token):
                if left.i != span_start - 1:
                    break
                if left.dep_ in {"aux", "auxpass", "neg", "advmod"}:
                    span_start = left.i
                else:
                    break
            for right in _right_descendants(token):
                if right.i != span_end:
                    break
                if right.dep_ in {"prt", "advmod", "acomp", "xcomp", "ccomp"}:
                    span_end = right.i + 1
                else:
                    break
            if span_start + 1 == span_end:
                continue
            span = doc[span_start:span_end]
            if ent_token_idxs.intersection(range(span.start, span.end)):
                continue
            spans_to_merge.append(span)
            ent_token_idxs.update(range(span.start, span.end))
    for token in doc:
        if token.pos_ == "ADJ" and token.dep_ in {"amod", "acomp"}:
            span_start = token.i
            span_end = token.i + 1
            for left in reversed(list(token.lefts)):
                if left.i != span_start - 1:
                    break
                if left.dep_ in {"advmod", "neg"}:
                    span_start = left.i
                else:
                    break
            for right in token.rights:
                if right.i != span_end:
                    break
                if right.dep_ in {"advmod", "acomp", "prep", "det"}:
                    span_end = right.i + 1
                else:
                    break
            if span_start + 1 == span_end:
                continue
            span = doc[span_start:span_end]
            if ent_token_idxs.intersection(range(span.start, span.end)):
                continue
            spans_to_merge.append(span)
            ent_token_idxs.update(range(span.start, span.end))
    for token in doc:
        if token.pos_ == "ADV" and token.dep_ in {"advmod"}:
            span_start = token.i
            span_end = token.i + 1
            for left in reversed(list(token.lefts)):
                if left.i != span_start - 1:
                    break
                if left.dep_ in {"advmod", "neg"}:
                    span_start = left.i
                else:
                    break
            for right in token.rights:
                if right.i != span_end:
                    break
                if right.dep_ in {"advmod", "prep", "det"}:
                    span_end = right.i + 1
                else:
                    break
            if span_start + 1 == span_end:
                continue
            span = doc[span_start:span_end]
            if ent_token_idxs.intersection(range(span.start, span.end)):
                continue
            spans_to_merge.append(span)
            ent_token_idxs.update(range(span.start, span.end))
    # Coreference indices must be repaired against all earlier merges before
    # their mentions are appended to the retokenization plan.
    corefs_clusters, corref_to_merge = CorrefCluster.fixup_clusters(corefs_clusters, spans_to_merge)
    spans_to_merge.extend(corref_to_merge)
    spans_to_merge = sorted(spans_to_merge, key=lambda s: s.start, reverse=True)
    return spans_to_merge
