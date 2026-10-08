"""L1 golden tests for hypergraph construction (no Rust, no embedding models).

Gate: ``HUFU_RUN_E2E=1`` plus spaCy + a small English model.  The suite pins
structural invariants and a handful of semantic anchors so that:

* bridge development has an observable target, and
* spaCy/model upgrades fail loudly instead of silently shifting labels.

Run with::

    HUFU_RUN_E2E=1 HUFU_SPACY_MODEL=en_core_web_sm \\
        .venv-e2e/Scripts/python.exe -m pytest tests/e2e/test_build_golden.py -v
"""

from __future__ import annotations

import importlib.util
import os
import unittest

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_SPACY_OK = importlib.util.find_spec("spacy") is not None

# Fixed inputs.  Keep them short and unambiguous; every case names the anchor
# it is responsible for so failures point at a single linguistic decision.
HEMINGWAY_QUERY = "Where was the author of 'The Old Man and the Sea' born?"
HEMINGWAY_DOC = (
    "Ernest Hemingway wrote The Old Man and the Sea. "
    "He was born in Oak Park, Illinois, in 1899."
)
NUMERIC_DOC = "The reservoir holds 3.2 million cubic meters of water."
TEMPORAL_DOC = "The treaty was signed on 11 November 1918 in Compiègne."


def _build(text: str, *, is_query: bool = False):
    from hufu.engine.build_hypergraph import text_to_hypergraph

    return text_to_hypergraph(text, is_query=is_query)


def _vertex_texts(hg) -> list[str]:
    return [v.text() for v in hg.vertices]


def _edge_roles(edge) -> list[tuple[str, str]]:
    """(dep-label, vertex text) for every non-root participant."""

    roles = []
    for vertex in edge.vertices:
        if vertex == edge.root:
            continue
        node = edge.current_node(vertex)
        roles.append((node.dep.name, vertex.text()))
    return roles


@unittest.skipUnless(_ENABLED and _SPACY_OK, "requires HUFU_RUN_E2E=1 and spacy")
class BuildInvariantTests(unittest.TestCase):
    """Invariants every constructed hypergraph must satisfy."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["HUFU_SPACY_MODEL"] = "en_core_web_sm"  # pinned (get_nlp reloads per suite)

    def _check_invariants(self, hg, text: str) -> None:
        vertex_ids = [v.id for v in hg.vertices]
        self.assertEqual(len(vertex_ids), len(set(vertex_ids)), f"duplicate vertex ids in {text!r}")
        for vertex in hg.vertices:
            self.assertTrue(vertex.text(), f"empty vertex text in {text!r}")
        for edge in hg.hyperedges:
            self.assertIn(edge.root, edge.vertices, f"root not among participants: {edge.desc!r}")
            # Contracts reject root-only pseudo-facts; the builder should only
            # emit relations with at least one semantic role.
            non_root = [v for v in edge.vertices if v != edge.root]
            self.assertTrue(non_root, f"root-only hyperedge: {edge.desc!r}")
            edge.assert_nodes_reach_root()

    def test_doc_invariants(self) -> None:
        for text in (HEMINGWAY_DOC, NUMERIC_DOC, TEMPORAL_DOC):
            with self.subTest(text=text):
                self._check_invariants(_build(text), text)

    def test_query_invariants(self) -> None:
        self._check_invariants(_build(HEMINGWAY_QUERY, is_query=True), HEMINGWAY_QUERY)


@unittest.skipUnless(_ENABLED and _SPACY_OK, "requires HUFU_RUN_E2E=1 and spacy")
class BuildAnchorTests(unittest.TestCase):
    """Semantic anchors: one linguistic decision per assertion."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["HUFU_SPACY_MODEL"] = "en_core_web_sm"  # pinned (get_nlp reloads per suite)

    def test_query_marks_placeholder(self) -> None:
        hg = _build(HEMINGWAY_QUERY, is_query=True)
        queries = [v for v in hg.vertices if v.is_query()]
        self.assertTrue(queries, "no query placeholder detected")
        for vertex in queries:
            self.assertTrue(vertex.text().startswith("?"), vertex.text())
            self.assertIsNotNone(vertex.query_type(), vertex.text())

    def test_query_root_is_verb(self) -> None:
        hg = _build(HEMINGWAY_QUERY, is_query=True)
        roots = [e.root for e in hg.hyperedges]
        self.assertTrue(roots, "query produced no hyperedges")
        self.assertTrue(
            any(r.is_verb() for r in roots),
            f"no verb-rooted relation; roots={[r.text() for r in roots]}",
        )

    def test_doc_detects_person(self) -> None:
        from hufu.hypergraph.entity import ENT

        hg = _build(HEMINGWAY_DOC)
        persons = [v for v in hg.vertices if v.type() == ENT.PERSON]
        self.assertTrue(
            any("Hemingway" in v.text() for v in persons),
            f"PERSON not detected; vertices={_vertex_texts(hg)}",
        )

    def test_doc_detects_number(self) -> None:
        from hufu.hypergraph.entity import ENT

        hg = _build(NUMERIC_DOC)
        numbers = [v for v in hg.vertices if v.type() == ENT.NUMBER]
        self.assertTrue(
            any("3.2" in v.text() for v in numbers),
            f"NUMBER not detected; vertices={_vertex_texts(hg)}",
        )

    def test_doc_detects_temporal(self) -> None:
        from hufu.hypergraph.entity import ENT

        hg = _build(TEMPORAL_DOC)
        temporals = [v for v in hg.vertices if v.type() == ENT.TEMPORAL]
        self.assertTrue(
            any("1918" in v.text() for v in temporals),
            f"TEMPORAL not detected; vertices={_vertex_texts(hg)}",
        )

    def test_roles_are_dependency_labels(self) -> None:
        from hufu.hypergraph.linguistic import Dep

        hg = _build(HEMINGWAY_DOC)
        seen = set()
        for edge in hg.hyperedges:
            for name, _text in _edge_roles(edge):
                seen.add(name)
        self.assertTrue(seen, "no roles extracted")
        for name in seen:
            self.assertIn(name, Dep.__members__, f"role {name!r} is not a Dep label")

    def test_doc_mentions_oak_park_as_place(self) -> None:
        from hufu.hypergraph.entity import ENT

        hg = _build(HEMINGWAY_DOC)
        places = {
            v.text()
            for v in hg.vertices
            if v.type() in {ENT.GPE, ENT.LOC, ENT.FAC, ENT.COUNTRY}
        }
        self.assertTrue(
            any("Oak Park" in text for text in places),
            f"Oak Park not typed as place; places={places}, all={_vertex_texts(hg)}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
