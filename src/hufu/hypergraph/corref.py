"""Track coreference mentions across retokenization and annotate graph nodes."""

from dataclasses import dataclass
from typing import List, Tuple

from spacy.tokens import Doc, Span

from hufu.hypergraph.dependency import Node
from hufu.hypergraph.linguistic import Entity, Pos


class CorrefCluster:
    """Mutable mention cluster whose token spans survive document retokenization."""

    def __init__(self, cluster_id: int, mentions: list[Span]):
        self.cluster_id: int = cluster_id
        self.mentions: list[Span] = mentions
        self.is_primary_mention: set[Span] = set()
        self.changed_ranges: list[Tuple[int, int]] = []
        self.changed_primary_ranges: list[Tuple[int, int]] = []
        self.dropped: bool = False
        self.covered_nodes_if_dropped: list[list[Node]] = []
    def drop(self):
        """Mark this ambiguous cluster as unavailable for node linking."""

        self.dropped = True
    def is_dropped(self) -> bool:
        """Return whether the cluster was excluded during overlap repair."""

        return self.dropped
    def get_current_index(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Return current token ranges for all mentions and primary mentions."""

        mention_indices = [(mention.start, mention.end) for mention in self.mentions]
        primary_indices = [(mention.start, mention.end) for mention in self.is_primary_mention]
        return mention_indices, primary_indices
    def _calc_changed_ranges(self):
        self.changed_ranges = []
        for mention in self.mentions:
            char_start = mention.start_char
            char_end = mention.end_char
            self.changed_ranges.append((char_start, char_end))
        self.changed_primary_ranges = [
            (mention.start_char, mention.end_char) for mention in self.is_primary_mention
        ]
    @staticmethod
    def from_doc(doc: Doc) -> List['CorrefCluster']:
        """Convert fastcoref character ranges into spaCy token-span clusters."""

        clusters: list['CorrefCluster'] = []
        resolved_text = doc._.resolved_text
        coref_clusters = doc._.coref_clusters
        if not coref_clusters or len(coref_clusters) == 0 or not resolved_text:
            return clusters
        for cluster in coref_clusters:
            mentions: list[Span] = []
            primary_mentions: list[Span] = []
            for char_start, char_end in cluster:
                token_start = None
                token_end = None
                for token in doc:
                    if token.idx <= char_start < token.idx + len(token):
                        token_start = token.i
                    if token.idx < char_end <= token.idx + len(token):
                        token_end = token.i + 1
                    if token_start is not None and token_end is not None:
                        break
                if token_start is not None and token_end is not None:
                    span = Span(doc, token_start, token_end)
                    mentions.append(span)
                    if span.text in resolved_text:
                        primary_mentions.append(span)
                else:
                    assert False, f"Failed to find token span for char span ({char_start}, {char_end}) in cluster {cluster}"
            cluster_id = len(clusters)
            cluster = CorrefCluster(cluster_id, mentions)
            for mention in primary_mentions:
                cluster.is_primary_mention.add(mention)
            clusters.append(cluster)
        return clusters
    @staticmethod
    def update_by_doc(old_clusters: List['CorrefCluster'], doc: Doc) -> List['CorrefCluster']:
        """Remap saved character ranges after the document was retokenized."""

        clusters: list['CorrefCluster'] = []
        for cluster in old_clusters:
            cluster._calc_changed_ranges()
        def _char_range_to_span(char_start: int, char_end: int) -> Span | None:
            token_start = None
            token_end = None
            for token in doc:
                if token.idx <= char_start < token.idx + len(token):
                    token_start = token.i
                if token.idx < char_end <= token.idx + len(token):
                    token_end = token.i + 1
                if token_start is not None and token_end is not None:
                    break
            if token_start is None or token_end is None:
                print(
                    f"[Corref] Failed to map char span ({char_start}, {char_end}) "
                    f"to tokens in updated doc (len={len(doc)}), skipping"
                )
                return None
            return Span(doc, token_start, token_end)
        for old_cluster in old_clusters:
            mentions: list[Span] = []
            primary_mentions: list[Span] = []
            for char_start, char_end in old_cluster.changed_ranges:
                span = _char_range_to_span(char_start, char_end)
                if span is not None:
                    mentions.append(span)
            for p_start, p_end in old_cluster.changed_primary_ranges:
                p_span = _char_range_to_span(p_start, p_end)
                if p_span is None:
                    continue
                if not any(m.start == p_span.start and m.end == p_span.end for m in mentions):
                    mentions.append(p_span)
                if not any(m.start == p_span.start and m.end == p_span.end for m in primary_mentions):
                    primary_mentions.append(p_span)
            new_cluster = CorrefCluster(old_cluster.cluster_id, mentions)
            new_cluster.is_primary_mention = set(primary_mentions)
            if old_cluster.is_dropped():
                new_cluster.drop()
            clusters.append(new_cluster)
        return clusters
    @staticmethod
    def fixup_clusters(clusters: List['CorrefCluster'], spans_to_merge: List[Span]) -> Tuple[List['CorrefCluster'], List[Span]]:
        """Repair overlapping mentions and return additional safe merge spans."""

        # Retokenization cannot accept overlapping spans.  Resolve ownership by
        # dependency root while retaining each surviving mention's primary flag.
        @dataclass
        class _KeptEntry:
            cluster_idx: int
            mention_idx: int
            root_index: int
            span: Span
        root_to_cluster_spans: dict[int, list[tuple[int, int]]] = {}
        for cluster_idx, cluster in enumerate(clusters):
            for mention in cluster.mentions:
                root_idx = mention.root.i
                span_len = mention.end - mention.start
                if root_idx not in root_to_cluster_spans:
                    root_to_cluster_spans[root_idx] = []
                root_to_cluster_spans[root_idx].append((span_len, cluster_idx))
        dropped_cidxs: set[int] = set()
        for root_idx, entries in root_to_cluster_spans.items():
            unique_cidxs = {cidx for _, cidx in entries}
            if len(unique_cidxs) <= 1:
                continue
            best_cidx = min(entries, key=lambda e: (e[0], e[1]))[1]
            for _, cidx in entries:
                if cidx != best_cidx:
                    dropped_cidxs.add(cidx)
        for cidx in dropped_cidxs:
            clusters[cidx].drop()
        if not clusters or not spans_to_merge:
            return clusters, []
        sorted_merges = sorted(spans_to_merge, key=lambda s: (s.start, s.end))
        def _has_overlap(a: Span, b: Span) -> bool:
            return a.start < b.end and b.start < a.end
        def _is_subset_of_any_merge(span: Span) -> bool:
            for m in sorted_merges:
                if m.end <= span.start:
                    continue
                if m.start > span.start:
                    break
                if m.start <= span.start and span.end <= m.end:
                    return True
            return False
        def _merge_owner_of_root(root_index: int) -> Span | None:
            owners: list[Span] = []
            for m in sorted_merges:
                if m.end <= root_index:
                    continue
                if m.start > root_index:
                    break
                if m.start <= root_index < m.end:
                    owners.append(m)
            if not owners:
                return None
            if len(owners) != 1:
                print(
                    f"[Corref] Root {root_index} belongs to {len(owners)} merge spans "
                    f"(expected 1): {[o.text for o in owners]}, using first"
                )
                return owners[0]
            return owners[0]
        def _partition_by_merges(a: Span) -> list[Span]:
            boundaries = {a.start, a.end}
            for m in sorted_merges:
                if m.end <= a.start:
                    continue
                if m.start >= a.end:
                    break
                if not _has_overlap(a, m):
                    continue
                boundaries.add(max(a.start, m.start))
                boundaries.add(min(a.end, m.end))
            sorted_bounds = sorted(boundaries)
            parts: list[Span] = []
            for i in range(len(sorted_bounds) - 1):
                s = sorted_bounds[i]
                e = sorted_bounds[i + 1]
                if s < e:
                    parts.append(Span(a.doc, s, e))
            return parts
        def _keep_left_to_root(span: Span, root_index: int) -> Span:
            end = max(span.start + 1, min(span.end, root_index + 1))
            return Span(span.doc, span.start, end)
        def _keep_right_from_root(span: Span, root_index: int) -> Span:
            start = min(span.end - 1, max(span.start, root_index))
            return Span(span.doc, start, span.end)
        rewritten_mentions: list[list[Span]] = []
        primary_source_indices: list[set[int]] = []
        kept_entries: list[_KeptEntry] = []
        for cluster_idx, cluster in enumerate(clusters):
            if cluster.is_dropped() or not cluster.mentions:
                rewritten_mentions.append([])
                primary_source_indices.append(set())
                continue
            fixed_mentions: list[Span] = []
            primary_source_idx_set: set[int] = set()
            primary_mentions_in_cluster = set(cluster.is_primary_mention)
            for mention_idx, mention in enumerate(cluster.mentions):
                root_index = mention.root.i
                pieces = _partition_by_merges(mention)
                fixed: Span | None = None
                for piece in pieces:
                    if piece.start <= root_index < piece.end:
                        fixed = piece
                        break
                if fixed is None:
                    fixed = mention
                if _is_subset_of_any_merge(fixed):
                    owner = _merge_owner_of_root(root_index)
                    if owner is not None:
                        fixed = owner
                else:
                    kept_entries.append(
                        _KeptEntry(
                            cluster_idx=cluster_idx,
                            mention_idx=mention_idx,
                            root_index=root_index,
                            span=fixed,
                        )
                    )
                fixed_mentions.append(fixed)
                if any(
                    mention.start == p.start and mention.end == p.end
                    for p in primary_mentions_in_cluster
                ):
                    primary_source_idx_set.add(mention_idx)
            rewritten_mentions.append(fixed_mentions)
            primary_source_indices.append(primary_source_idx_set)
        changed = True
        while changed and len(kept_entries) > 1:
            changed = False
            order = sorted(range(len(kept_entries)), key=lambda idx: (kept_entries[idx].span.start, kept_entries[idx].span.end))
            for pos, i in enumerate(order):
                a_entry = kept_entries[i]
                a_span = a_entry.span
                a_root = a_entry.root_index
                if a_span.start >= a_span.end:
                    continue
                for j in order[pos + 1 :]:
                    b_entry = kept_entries[j]
                    b_span = b_entry.span
                    if b_span.start >= b_span.end:
                        continue
                    if b_span.start >= a_span.end:
                        break
                    if not _has_overlap(a_span, b_span):
                        continue
                    b_root = b_entry.root_index
                    if a_root == b_root:
                        print(
                            f"[Corref] Overlapping spans with same root {a_root}: "
                            f"'{a_span.text}' (cluster {a_entry.cluster_idx}) vs "
                            f"'{b_span.text}' (cluster {b_entry.cluster_idx}), "
                            f"keeping longer span"
                        )
                        if a_span.end - a_span.start <= b_span.end - b_span.start:
                            a_entry.span = Span(a_span.doc, a_span.start, a_span.start)
                        else:
                            b_entry.span = Span(b_span.doc, b_span.start, b_span.start)
                        changed = True
                        break
                    if a_root < b_root:
                        new_a = _keep_left_to_root(a_span, a_root)
                        new_b = _keep_right_from_root(b_span, b_root)
                    else:
                        new_a = _keep_right_from_root(a_span, a_root)
                        new_b = _keep_left_to_root(b_span, b_root)
                    if new_a.start != a_span.start or new_a.end != a_span.end:
                        a_entry.span = new_a
                        changed = True
                    if new_b.start != b_span.start or new_b.end != b_span.end:
                        b_entry.span = new_b
                        changed = True
                    break
                if changed:
                    break
        for entry in kept_entries:
            cluster_idx = entry.cluster_idx
            mention_idx = entry.mention_idx
            span = entry.span
            rewritten_mentions[cluster_idx][mention_idx] = span
        for cluster_idx, cluster in enumerate(clusters):
            fixed_mentions = rewritten_mentions[cluster_idx] if cluster_idx < len(rewritten_mentions) else []
            if not fixed_mentions:
                continue
            dedup_mentions: list[Span] = []
            for m in fixed_mentions:
                if not any(x.start == m.start and x.end == m.end for x in dedup_mentions):
                    dedup_mentions.append(m)
            cluster.mentions = dedup_mentions
            source_idx_set = primary_source_indices[cluster_idx] if cluster_idx < len(primary_source_indices) else set()
            new_primary_mentions: set[Span] = set()
            for source_idx in source_idx_set:
                if 0 <= source_idx < len(fixed_mentions):
                    primary_candidate = fixed_mentions[source_idx]
                    mapped_primary = next(
                        (m for m in dedup_mentions if m.start == primary_candidate.start and m.end == primary_candidate.end),
                        None,
                    )
                    if mapped_primary is not None:
                        new_primary_mentions.add(mapped_primary)
            cluster.is_primary_mention = new_primary_mentions
        kept_a_minus_b_spans: list[Span] = []
        for entry in kept_entries:
            span = entry.span
            if not any(s.start == span.start and s.end == span.end for s in kept_a_minus_b_spans):
                kept_a_minus_b_spans.append(span)
        return clusters, kept_a_minus_b_spans
def mark_corref(nodes: List[Node], corref_clusters: List[CorrefCluster]) -> List[Node]:
    """Attach cluster ids, primary mentions, and resolved text to graph nodes."""

    node_map: dict[int, Node] = {node.index: node for node in nodes}
    def _coref_primary_rank(node: Node) -> tuple[int, int, int, int]:
        ent_score = 1 if node.ent != Entity.NOT_ENTITY else 0
        pos_priority: dict[Pos, int] = {
            Pos.VERB: 10,
            Pos.AUX: 10,
            Pos.NOUN: 8,
            Pos.PROPN: 8,
            Pos.ADJ: 6,
            Pos.NUM: 5,
            Pos.ADV: 4,
            Pos.ADP: 3,
            Pos.PART: 2,
            Pos.PRON: 0,
        }
        pos_score = pos_priority.get(node.pos, 1)
        length_score = len(node.text)
        return (ent_score, pos_score, length_score, -node.index)
    def extract_verb_text(node: Node) -> str:
        if node.pos not in {Pos.VERB, Pos.AUX}:
            return node.text
        text = node.text.strip()
        if text.lower().startswith("to ") and len(text) > 3:
            return text
        if node.lemma and node.lemma.lower() not in {"to", node.text.lower()}:
            return node.lemma
        return node.text
    # Dropped clusters retain their covered nodes for diagnostics but must not
    # create referential links in the dependency graph.
    for cluster in corref_clusters:
        if cluster.is_dropped():
            covered_nodes: list[list[Node]] = []
            for mention in cluster.mentions:
                mention_nodes: list[Node] = []
                mention_node_set: set[Node] = set()
                for token in mention:
                    node = node_map.get(token.i)
                    if not node or node in mention_node_set:
                        continue
                    mention_node_set.add(node)
                    mention_nodes.append(node)
                covered_nodes.append(mention_nodes)
            cluster.covered_nodes_if_dropped = covered_nodes
            continue
        cluster_tokens: list[Node] = []
        cluster_token_set: set[Node] = set()
        for mention in cluster.mentions:
            for token in mention:
                node = node_map.get(token.i)
                if not node:
                    continue
                if node not in cluster_token_set:
                    cluster_token_set.add(node)
                    cluster_tokens.append(node)
        if len(cluster_tokens) <= 1:
            continue
        for node in cluster_tokens:
            node.correfence_id = cluster.cluster_id
        primary_span_token_ids: set[int] = set()
        if cluster.is_primary_mention:
            for primary_span in cluster.is_primary_mention:
                primary_span_token_ids.update(token.i for token in primary_span)
        def _rank_with_span_preference(node: Node) -> tuple[int, int, int, int, int]:
            base_rank = _coref_primary_rank(node)
            in_span_bonus = 1 if node.index in primary_span_token_ids else 0
            particle_penalty = -100 if node.pos == Pos.PART and node.text.lower() == "to" else 0
            return (
                base_rank[0],
                base_rank[1] + particle_penalty,
                in_span_bonus,
                base_rank[2],
                base_rank[3],
            )
        primary_node = max(cluster_tokens, key=_rank_with_span_preference)
        primary_node.is_correfence_primary = True
        if primary_node.pos in {Pos.VERB, Pos.AUX}:
            primary_text_for_replacement = primary_node.lemma
        else:
            primary_text_for_replacement = primary_node.text
        for node in cluster_tokens:
            if node.is_correfence_primary:
                if node.pos in {Pos.VERB, Pos.AUX}:
                    node.resolved_text = extract_verb_text(node)
                else:
                    node.resolved_text = node.text
            elif node.pos == Pos.PRON and primary_text_for_replacement:
                node.resolved_text = primary_text_for_replacement
            node.coref_primary = primary_node
            if node.pos == Pos.PRON and not node.pronoun_antecedent:
                node.pronoun_antecedent = primary_node
        if not primary_node.resolved_text:
            if primary_node.pos in {Pos.VERB, Pos.AUX}:
                primary_node.resolved_text = extract_verb_text(primary_node)
            else:
                primary_node.resolved_text = primary_node.text
    return nodes
