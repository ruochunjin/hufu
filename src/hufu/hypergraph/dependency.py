"""Convert spaCy dependencies into normalized vertices and n-ary relations."""

try:
    from thefuzz import process
except ImportError:
    # Hypergraph types remain importable in minimal installations.  Experiment
    # environments still use thefuzz; this deterministic fallback implements
    # only the extractOne contract needed below.
    from difflib import SequenceMatcher

    class _FallbackProcess:
        @staticmethod
        def extractOne(query: str, choices):
            ranked = [
                (
                    choice,
                    round(
                        100
                        * SequenceMatcher(
                            None, str(query).casefold(), str(choice).casefold()
                        ).ratio()
                    ),
                )
                for choice in choices
            ]
            return max(ranked, key=lambda item: (item[1], str(item[0]))) if ranked else None

    process = _FallbackProcess()
from hufu.hypergraph.entity import ENT
from hufu.hypergraph.linguistic import Dep, Entity, Pos, QueryType, Tag

dead_dep = {Dep.dative, Dep.prt, Dep.parataxis}
solved_dep = {Dep.meta, Dep.poss, Dep.det, Dep.predet, Dep.intj}
def _restrict_correfs(clusters: list[list[tuple[int, int]]], level: int=0) -> list[list[tuple[int, int]]]:
    restricted = []
    for cluster in clusters:
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
class Node:
    """Mutable linguistic token record enriched during graph construction."""

    def __init__(self, text: str, pos: Pos, tag: Tag, dep: Dep, ent: Entity, lemma: str, index: int) -> None:
        self.text = text
        self.original_text = text
        self.pos: Pos = pos
        self.tag: Tag = tag
        self.dep: Dep = dep
        self.ent: Entity = ent
        self.lemma: str = lemma
        self.sentence: str = text
        self.covered_sentence: str = text
        self.sentence_start: int = -1
        self.sentence_end: int = -1
        self.index = index
        self.is_query = False
        self.query_type: QueryType | None = None
        self.query_attribute: str | None = None
        self.is_vertex = False
        self.former_nodes: list[Node] = []
        self.dominator = False
        self.pronoun_antecedent: Node | None = None
        self.prefix_prep: str | None = None
        self.suffix_prep: str | None = None
        self.prefix_agent: str | None = None
        self.suffix_agent: str | None = None
        self.prefix_index: int | None = None
        self.suffix_index: int | None = None
        self.correfence_id: int | None = None
        self.is_correfence_primary: bool = False
        self.coref_primary: Node | None = None
        self.resolved_text: str | None = None
        self.head: Node | None = None
        self.children: list[Node] = []
        self.lefts: list[Node] = []
        self.rights: list[Node] = []
        self.wn_abstraction: str | None = None
        self.wn_hypernym_path: list[str] = []
        self.entity: ENT | None = None
        self.wd_tags: dict[str, str] = {}
        self.source_id: str | int | None = None
    def set_sentence(self, sentence: str, start: int, end: int) -> None:
        """Attach the dependency-covered sentence and token bounds."""

        self.sentence = sentence
        self.covered_sentence = sentence
        self.sentence_start = start
        self.sentence_end = end
    def set_entity(self, entity: ENT) -> None:
        """Attach a project-level coarse entity type."""

        self.entity = entity
    def type_str(self) -> str | None:
        """Return the matcher-facing semantic type for this node, if any."""

        if self.pos in {Pos.VERB, Pos.AUX}:
            return None
        if self.is_query:
            query_map = {
                QueryType.LOCATION: "LOCATION",
                QueryType.TIME: "TEMPORAL",
                QueryType.ATTRIBUTE: "ATTRIBUTE",
                QueryType.PERSON: "PERSON",
                QueryType.BELONGS: "COMPONENTS",
                QueryType.REASON: "REASON",
            }
            if self.query_type in query_map:
                return query_map[self.query_type]
        if self.entity:
            entity_mapping = {
                ENT.CONCEPT: "CONCEPT",
                ENT.TEMPORAL: "TEMPORAL",
                ENT.NUMBER: "NUMBER",
                ENT.ORGANISM: "ORGANISM",
                ENT.FOOD: "FOOD",
                ENT.MEDICAL: "MEDICAL",
                ENT.ANATOMY: "ANATOMY",
                ENT.SUBSTANCE: "SUBSTANCE",
                ENT.ASTRO: "ASTRO",
                ENT.AWARD: "AWARD",
                ENT.VEHICLE: "VEHICLE",
                ENT.PERSON: "PERSON",
                ENT.COUNTRY: "COUNTRY",
                ENT.LOC: "LOCATION",
                ENT.ORG: "ORGANIZATION",
                ENT.FAC: "FACILITY",
                ENT.GPE: "Geopolitical",
                ENT.NORP: "NORP",
                ENT.PRODUCT: "PRODUCT",
                ENT.WORK_OF_ART: "WORK_OF_ART",
                ENT.LAW: "LAW",
                ENT.LANGUAGE: "LANGUAGE",
                ENT.OCCUPATION: "OCCUPATION",
                ENT.EVENT: "EVENT",
                ENT.THEORY: "THEORY",
                ENT.GROUP: "GROUP",
                ENT.FEATURE: "FEATURE",
                ENT.ECONOMIC: "ECONOMIC",
                ENT.SOCIOLOGY: "SOCIOLOGY",
                ENT.PHENOMENON: "PHENOMENON",
            }
            if self.entity in entity_mapping:
                return entity_mapping[self.entity]
        if self.pos == Pos.ADJ:
            return "ADJECTIVE"
        if self.pos == Pos.ADV:
            return "ADVERB"
    @staticmethod
    def from_doc(doc, abst) -> tuple[list['Node'], list['Node']]:
        """Build linked nodes and roots from a spaCy document and abstractor."""

        nodes: list[Node] = []
        node_map: dict[int, Node] = {}
        wildcard_tags = {',', '.', '-LRB-', '-RRB-', '``', ':', "''", 'PRP$', 'WP$', '$', 'AFX'}
        for token in doc:
            pos = token.pos_
            tag = "WILDCARD" if token.tag_ in wildcard_tags else token.tag_
            dep = token.dep_
            ent = token.ent_type_ if token.ent_type_ else "NOT_ENTITY"
            sentence = doc[token.left_edge.i : token.right_edge.i + 1].text
            node = Node(
                text=token.text,
                pos=Pos[pos],
                tag=Tag[tag],
                dep=Dep[dep],
                ent=Entity[ent],
                lemma=token.lemma_,
                index=token.i,
            )
            entity_by_span: ENT | None = abst.get_entity_for_char_index(token.idx)
            if entity_by_span:
                node.set_entity(entity_by_span)
            else:
                entity_by_token: ENT | None = abst.get_entity_for_token(token, doc)
                if entity_by_token:
                    node.set_entity(entity_by_token)
            node.set_sentence(sentence, token.left_edge.i, token.right_edge.i + 1)
            node_map[token.i] = node
            nodes.append(node)
        for token in doc:
            node = node_map.get(token.i)
            if not node:
                continue
            if token.head.i != token.i and token.head.i in node_map:
                node.head = node_map[token.head.i]
                node_map[token.head.i].children.append(node)
            for left in token.lefts:
                left_node = node_map.get(left.i)
                if left_node:
                    node.lefts.append(left_node)
            for right in token.rights:
                right_node = node_map.get(right.i)
                if right_node:
                    node.rights.append(right_node)
        roots = [node for node in nodes if node.head is None]
        for root in roots:
            assert root.dep == Dep.ROOT or root.dep == Dep.dep, f"Root node dep should be ROOT or _SP, got {root.dep.name}: '{root.text}'\nDOC: '{doc.text}'"
        return nodes, roots
    def has_entity(self) -> bool:
        """Return whether either entity taxonomy assigns a concrete type."""

        return self.ent != Entity.NOT_ENTITY or (self.entity is not None and self.entity != ENT.NOT_ENT)
    def __format__(self, format_spec: str) -> str:
        return f"Node(text='{self.text}', pos={self.pos.name}, tag={self.tag.name}, dep={self.dep.name}, ent={self.ent.name}, sentence='{self.sentence}')"
    def __repr__(self) -> str:
        return self.__format__('')
    def __display__(self) -> str:
        return self.__format__('')
    def __str__(self) -> str:
        return self.__format__('')
class Relationship:
    """One dependency-rooted n-ary relation and its source sentence span."""

    def __init__(self, entities: list[Node], sentence: str, relationship_sentence: str) -> None:
        self.root = entities[0]
        self.entities = entities
        self.sentence = sentence
        self.relationship_sentence = relationship_sentence
        start, end = entities[0].index, entities[0].index
        for entity in entities[1:]:
            if entity.index < start:
                start = entity.index
            if entity.index > end:
                end = entity.index
        self.start = start
        self.end = end + 1
        self.father: Relationship | None = None
    def position_text(self, node: Node) -> str:
        """Render a relation participant with its determiner-like children."""

        from hufu.hypergraph.hypergraph import Vertex
        res = Vertex.resolved_text(node)
        determiner_children: list[Node] = []
        for child in node.lefts:
            if child.dep in {Dep.det, Dep.poss, Dep.predet}:
                determiner_children.append(child)
        determiner_children.sort(key=lambda n: n.index)
        if determiner_children:
            prefix = " ".join(Vertex.resolved_text(child) for child in determiner_children)
            res = f"{prefix} {res}"
        return res
    def relationship_text_simple(self) -> str:
        """Render only the relation-bearing portion of the source sentence."""

        def calc_prefix_suffix():
            rel_start = self.sentence.find(self.relationship_sentence)
            if rel_start != -1:
                prefix = self.sentence[:rel_start].strip()
                suffix = self.sentence[rel_start + len(self.relationship_sentence):].strip()
            else:
                prefix = ""
                suffix = ""
            return prefix, suffix
        prefix, suffix = calc_prefix_suffix()
        from hufu.hypergraph.hypergraph import Vertex
        sentence = str(self.relationship_sentence)
        for entity in self.entities[1:]:
            new_text = self.position_text(entity)
            old_candidates = [entity.sentence, Vertex.resolved_text(entity)]
            for old in old_candidates:
                if old and old in sentence:
                    sentence = sentence.replace(old, new_text, 1)
                    break
        sentence = sentence.replace(prefix, "").replace(suffix, "").strip()
        return sentence
    def __format__(self, format_spec: str) -> str:
        return f"[root: {self.position_text(self.root)}] ({', '.join([self.position_text(entity) for entity in self.entities])})\n\tIn Sentence: '{self.sentence}'\n\tSimple: '{self.relationship_text_simple()}'"
    def __repr__(self) -> str:
        return self.__format__('')
    def __str__(self) -> str:
        return self.__format__('')
    def __display__(self) -> str:
        return self.__format__('')
class LocalDoc:
    """Pickle-friendly token-text view supporting integer and slice access."""

    def __init__(self, doc) -> None:
        self.tokens = [token.text for token in doc]
    def __getitem__(self, index) -> str:
        if isinstance(index, slice):
            return ' '.join(self.tokens[index])
        else:
            return self.tokens[index]
class Dependency:
    """Stateful normalization pipeline from dependency nodes to relationships."""

    def __init__(self, nodes: list[Node], roots: list[Node], doc: LocalDoc, is_query: bool=False) -> None:
        self.nodes = nodes
        self.roots = roots
        self.doc = doc
        self.vertexes: list[Node] = []
        self.links_succ: dict[Node, list[Node]] = {}
        self.links_pred: dict[Node, Node] = {}
        self.relationship_sentences: dict[Node, str] = {}
        self.correfence_map: dict[Node, Node] = {}
        self.is_query = is_query
    def _fixup_lefts_rights_sentences(self, node: Node) -> None:
        node.children.sort(key=lambda n: n.index)
        node.lefts = [child for child in node.children if child.index < node.index]
        node.rights = [child for child in node.children if child.index > node.index]
        node.sentence_start = node.lefts[0].index if node.lefts else node.index
        node.sentence_end = node.rights[-1].index + 1 if node.rights else node.index + 1
        node.sentence = self.doc[node.sentence_start : node.sentence_end]
    def _calc_relationship_sentence(self, root: Node):
        left_edge = right_edge = root.index
        for succ in self.links_succ.get(root, []):
            if succ.index < left_edge:
                left_edge = succ.index
            if succ.index > right_edge:
                right_edge = succ.index
        return self.doc[left_edge : right_edge + 1]
    def solve_conjunctions(self):
        """Lift conjuncts and appositions to the governing dependency level."""

        if self.is_query:
            wh_dets = {"what", "which"}
            for node in self.nodes:
                if node.dep == Dep.det and node.text.lower() in wh_dets and node.head:
                    node.head.is_query = True
                    for child in node.head.children:
                        if child.dep == Dep.conj and child.head == node.head:
                            child.is_query = True
        queue = self.roots.copy()
        next_level: list[Node] = []
        while queue:
            node = queue.pop(0)
            remove_children: list[Node] = []
            for child in node.children:
                if child.dep == Dep.conj or (child.dep == Dep.appos and node.head):
                    child.dep = node.dep
                    child.head = node.head
                    remove_children.append(child)
                    queue.append(child)
                else:
                    next_level.append(child)
            for child in remove_children:
                node.children.remove(child)
                if node.head:
                    node.head.children.append(child)
            if not queue:
                queue = next_level
                next_level = []
            if not remove_children:
                continue
            self._fixup_lefts_rights_sentences(node)
            if node.head:
                self._fixup_lefts_rights_sentences(node.head)
        self.roots = [node for node in self.nodes if node.head is None]
        return self
    def mark_pronoun_antecedents(self):
        """Resolve supported relative and query-clause pronoun patterns."""

        for node in self.nodes:
            if node.dep != Dep.relcl or not node.head:
                continue
            is_pronoun_antecedent = False
            antecedent = node.head
            for child in node.children:
                if child.pos == Pos.PRON and child.ent == Entity.NOT_ENTITY:
                    child.pronoun_antecedent = antecedent
                    is_pronoun_antecedent = True
            if not is_pronoun_antecedent:
                continue
            node.head.children.remove(node)
            self._fixup_lefts_rights_sentences(node.head)
            node.dep = Dep.ROOT
            node.head = None
        if not self.is_query:
            return self
        for node in self.nodes:
            if node.dep != Dep.ccomp or not node.head:
                continue
            head = node.head
            for child in head.children:
                if child.dep not in {Dep.nsubj, Dep.nsubjpass}:
                    continue
                if child.pos != Pos.PRON:
                    continue
                for ccomp_child in node.children:
                    if (ccomp_child.dep in {Dep.nsubj, Dep.nsubjpass}) and (ccomp_child.pos in {Pos.PRON}):
                        child.pronoun_antecedent = ccomp_child
        return self
    def mark_prefixes(self):
        """Record preposition and agent markers adjacent to participant nodes."""

        for node in self.nodes:
            if node.dep == Dep.agent and node.head:
                if node.index < node.head.index:
                    node.head.prefix_agent = node.text
                    node.head.prefix_index = node.index
                else:
                    node.head.suffix_agent = node.text
                    node.head.suffix_index = node.index
            if node.dep == Dep.prep and node.head:
                if node.index < node.head.index:
                    node.head.prefix_prep = node.text
                    node.head.prefix_index = node.index
                else:
                    node.head.suffix_prep = node.text
                    node.head.suffix_index = node.index
            if node.dep == Dep.pobj and node.head:
                if node.index > node.head.index:
                    node.prefix_prep = node.head.text
                    node.prefix_index = node.head.index
                else:
                    node.suffix_prep = node.head.text
                    node.suffix_index = node.head.index
        return self
    def mark_vertex(self):
        """Select semantic vertices and annotate query placeholders."""

        for node in self.nodes:
            if node.dep in {Dep.nsubj, Dep.nsubjpass, Dep.csubj, Dep.csubjpass}:
                node.dominator = True
        correfence_primary_map: dict[int, Node] = {}
        for node in self.nodes:
            if node.correfence_id is not None and node.is_correfence_primary:
                correfence_primary_map[node.correfence_id] = node
        self.correfence_map = {
            node: correfence_primary_map[node.correfence_id]
            for node in self.nodes
            if node.correfence_id is not None
            and not node.is_correfence_primary
            and node.correfence_id in correfence_primary_map
        }
        pronoun_antecedent_map: dict[Node, Node] = {}
        for node in self.nodes:
            if node.pos == Pos.PRON and node.pronoun_antecedent:
                antecedent = node.pronoun_antecedent
                while antecedent.coref_primary and antecedent != antecedent.coref_primary:
                    antecedent = antecedent.coref_primary
                pronoun_antecedent_map[node] = antecedent
        for node, antecedent in pronoun_antecedent_map.items():
            if node not in self.correfence_map:
                self.correfence_map[node] = antecedent
            if not node.coref_primary:
                node.coref_primary = antecedent
        qualifying_pos = {Pos.NOUN, Pos.PROPN, Pos.VERB, Pos.AUX, Pos.ADJ, Pos.NUM, Pos.PRON, Pos.ADV}
        self.vertexes = []
        for node in self.nodes:
            node.is_vertex = False
            if node.pos in {Pos.SPACE, Pos.PUNCT} and node.ent == Entity.NOT_ENTITY:
                continue
            if self.is_query and node.pos == Pos.AUX and node.dep == Dep.aux and node.head and node.head.pos == Pos.VERB:
                continue
            if self.is_query and node.pos == Pos.PRON and node.dep == Dep.det:
                continue
            if self.is_query:
                normalized_how = node.text.lower().replace("-", " ").strip(" \t\n\r\f\v.,?!;:")
                normalized_how = " ".join(normalized_how.split())
                if normalized_how.startswith("how"):
                    parts = normalized_how.split()
                    how_tail = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
                    how_quantifiers = {"many", "much", "few", "little", "long", "often", "far"}
                    if how_tail in how_quantifiers:
                        node.is_query = True
                        node.query_type = QueryType.NUMBER
                    elif node.pos in {Pos.ADJ, Pos.ADV}:
                        node.is_query = True
                        node.query_type = QueryType.ATTRIBUTE
                        node.query_attribute = how_tail or None
            def is_relative_pronoun(node: Node) -> bool:
                return node.pronoun_antecedent is not None
            if self.is_query and node.pos == Pos.PRON and (not is_relative_pronoun(node)):
                wh_pronouns = {"what", "which", "who", "whom", "whose"}
                if node.text.lower() in wh_pronouns or node.tag in {"WP", "WP$"}:
                    node.is_query = True
                    pronoun = node.text.lower()
                    if pronoun == "which":
                        node.query_type = QueryType.WHICH
                    elif pronoun == "what":
                        node.query_type = QueryType.WHAT
                    elif pronoun in {"who", "whom"}:
                        node.query_type = QueryType.PERSON
            def is_clause_sconj(node: Node) -> bool:
                head = node.head
                if not head:
                    return False
                if head.dep in {Dep.acl, Dep.relcl}:
                    return True
                for child in head.children:
                    if child.dep in {Dep.acl, Dep.relcl} and child.head == head:
                        return True
                return False
            if self.is_query and (node.pos == Pos.ADV or (node.pos == Pos.SCONJ and (not is_clause_sconj(node)))):
                wh_adverbs = {"when", "where", "why"}
                if node.text.lower() in wh_adverbs or node.tag in {"WRB"}:
                    node.is_query = True
                    adverb = node.text.lower()
                    if adverb == "when":
                        node.query_type = QueryType.TIME
                    elif adverb == "where":
                        node.query_type = QueryType.LOCATION
                    elif adverb == "why":
                        node.query_type = QueryType.REASON
                    self.vertexes.append(node)
                    continue
            if self.is_query and node.pos == Pos.DET and node.dep == Dep.poss and (node.text.lower() == "whose" or node.tag == "WP$"):
                node.is_vertex = True
                node.is_query = True
                node.query_type = QueryType.BELONGS
                self.vertexes.append(node)
                continue
            if (
                node.head is None
                or node.ent != Entity.NOT_ENTITY
                or node.pos in qualifying_pos
            ):
                if node.ent != Entity.NOT_ENTITY and node.pos in {Pos.DET, Pos.PART, Pos.PUNCT}:
                    continue
                if node.pos in {Pos.AUX} and not node.children:
                    continue
                node.is_vertex = True
                self.vertexes.append(node)
        return self
    def compress_dependencies(self):
        """Connect selected vertices across intervening non-vertex tokens."""

        for node in self.vertexes:
            if not node.head:
                continue
            pred = node.head
            while pred and not pred.is_vertex:
                node.former_nodes.insert(0, pred)
                pred = pred.head
            if pred:
                self.links_pred[node] = pred
                if pred not in self.links_succ:
                    self.links_succ[pred] = []
                self.links_succ[pred].append(node)
        return self
    def calc_relationships(self) -> tuple[list[Node], list[Relationship], dict[Node, int]]:
        """Build n-ary relationships and assign stable within-document vertex ids."""

        def _match_same(
            best_match,
            score,
            node: Node,
            choices_map: dict[str, int],
            pos_map: dict[int, Pos],
            entity_map: dict[int, Entity],
            ent_map: dict[int, ENT],
        ) -> bool:
            candidate_id = choices_map[best_match]
            virtual = {Pos.PRON, Pos.AUX, Pos.VERB}
            if pos_map[candidate_id] in virtual or node.pos in virtual:
                return False
            node_ent = node.entity if node.entity is not None else ENT.NOT_ENT
            if score == 100:
                return True
            elif score >= 90 and (pos_map[candidate_id] == node.pos) and (entity_map[candidate_id] == node.ent) and (ent_map[candidate_id] == node_ent):
                return True
            return False
        saved_rels: set[tuple[str, str]] = set()
        relationships: list[Relationship] = []
        vertex_id_map: dict[Node, int] = {}
        root_to_relationship: dict[Node, Relationship] = {}
        for node in self.vertexes:
            if node in self.links_succ:
                node_key_text = (node.resolved_text or node.text)
                if (node_key_text, node.sentence) in saved_rels:
                    continue
                relational_sentence = self._calc_relationship_sentence(node)
                saved_rels.add((node_key_text, node.sentence))
                rel = Relationship(entities=[node] + self.links_succ[node], sentence=node.sentence, relationship_sentence=relational_sentence)
                root_to_relationship[node] = rel
                relationships.append(rel)
        relationship_trees: dict[Relationship, Relationship] = {}
        for rel in relationships:
            node = rel.root
            for succ in self.links_succ.get(node, []):
                if succ in root_to_relationship:
                    child_rel = root_to_relationship[succ]
                    if (succ.sentence_start >= node.sentence_start and succ.sentence_end <= node.sentence_end):
                        relationship_trees[child_rel] = rel
        for rel, father_rel in relationship_trees.items():
            rel.father = father_rel
        choices = []
        choices_map: dict[str, int] = {}
        pos_map: dict[int, Pos] = {}
        entity_map: dict[int, Entity] = {}
        ent_map: dict[int, ENT] = {}
        cnt = 1
        deferred_coref_nodes: list[Node] = []
        # Exact and near-exact lexical identity is accepted only when POS and
        # both entity taxonomies agree; coreferent nodes are assigned afterward.
        for node in self.vertexes:
            if node.coref_primary:
                deferred_coref_nodes.append(node)
                continue
            base_text = node.resolved_text or node.text
            text = base_text.lower()
            extraction = process.extractOne(text, choices) if choices else None
            match extraction:
                case (best_match, score) if _match_same(best_match, score, node, choices_map, pos_map, entity_map, ent_map):
                    vertex_id_map[node] = choices_map[best_match]
                    pos_map[vertex_id_map[node]] = node.pos
                case _:
                    choices.append(text)
                    choices_map[text] = cnt
                    vertex_id_map[node] = cnt
                    pos_map[cnt] = node.pos
                    entity_map[cnt] = node.ent
                    ent_map[cnt] = node.entity if node.entity is not None else ENT.NOT_ENT
                    cnt += 1
        def _assign_or_match_id(node: Node) -> None:
            nonlocal cnt
            base_text = node.resolved_text or node.text
            text = base_text.lower()
            extraction = process.extractOne(text, choices) if choices else None
            match extraction:
                case (best_match, score) if _match_same(best_match, score, node, choices_map, pos_map, entity_map, ent_map):
                    vertex_id_map[node] = choices_map[best_match]
                    pos_map[vertex_id_map[node]] = node.pos
                case _:
                    choices.append(text)
                    choices_map[text] = cnt
                    vertex_id_map[node] = cnt
                    pos_map[cnt] = node.pos
                    entity_map[cnt] = node.ent
                    ent_map[cnt] = node.entity if node.entity is not None else ENT.NOT_ENT
                    cnt += 1
        for node in deferred_coref_nodes:
            primary: Node | None = node.coref_primary
            if primary is not node:
                continue
            if node in vertex_id_map:
                continue
            _assign_or_match_id(node)
        for node in deferred_coref_nodes:
            if node in vertex_id_map:
                continue
            primary: Node | None = node.coref_primary
            if primary and primary in vertex_id_map:
                vertex_id_map[node] = vertex_id_map[primary]
                pos_map[vertex_id_map[node]] = primary.pos
                continue
            _assign_or_match_id(node)
        return self.vertexes, relationships, vertex_id_map
