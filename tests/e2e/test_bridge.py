"""Bridge conversion tests: LocalHypergraph -> contracts.Hypergraph.

Runs on the same model stack as the L1 golden tests (en_core_web_sm, no Rust,
no embedding models).  Assertions pin the mapping table recorded in
hufu-e2e-测试驱动开发记录.md: kinds come from the Vertex.is_* family, roles
are dependency labels, query placeholders carry expected_type.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

_ENABLED = os.environ.get("HUFU_RUN_E2E") == "1"
_SPACY_OK = importlib.util.find_spec("spacy") is not None

QUERY = "Where was the author of 'The Old Man and the Sea' born?"
DOC = (
    "Ernest Hemingway wrote The Old Man and the Sea. "
    "He was born in Oak Park, Illinois, in 1899."
)


def _bridge(text: str, *, side: str, is_query: bool = False):
    from hufu.engine.bridge import bridge_hypergraph
    from hufu.engine.build_hypergraph import text_to_hypergraph

    local = text_to_hypergraph(text, is_query=is_query)
    return local, bridge_hypergraph(local, side=side)


@unittest.skipUnless(_ENABLED and _SPACY_OK, "requires HUFU_RUN_E2E=1 and spacy")
class BridgeStructuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("HUFU_SPACY_MODEL", "en_core_web_sm")

    def test_all_local_vertices_convert(self) -> None:
        local, result = _bridge(DOC, side="data")
        self.assertEqual(len(result.local_to_core_vertex), len(local.vertices))
        # Data-side nominal roots synthesize predicate vertices (H05/L07 fix),
        # so the contracts graph may hold MORE vertices than the local graph.
        self.assertGreaterEqual(len(result.hypergraph.vertices), len(local.vertices))
        local_ids = set(result.local_to_core_vertex.values())
        for vertex in result.hypergraph.vertices:
            if vertex.id not in local_ids:
                self.assertEqual(vertex.kind, "predicate")
                self.assertTrue(vertex.id.endswith("-pred"), vertex.id)

    def test_query_side_has_no_synthetic_predicates(self) -> None:
        # Query-side roots stay forced-predicate: synthesizing them would
        # bloat the critical coverage set (golden100 A/B regression).
        _local, result = _bridge(QUERY, side="query", is_query=True)
        for vertex in result.hypergraph.vertices:
            self.assertFalse(vertex.id.endswith("-pred"), vertex.id)

    def test_contracts_validation_passes(self) -> None:
        # Hypergraph construction itself validates: unique ids, known vertex
        # refs, predicate-kind predicates, non-root-only edges.
        _local, result = _bridge(DOC, side="data")
        self.assertTrue(result.hypergraph.hyperedges)
        self.assertEqual(result.dropped_edges, 0)

    def test_every_edge_has_predicate_kind_root(self) -> None:
        _local, result = _bridge(DOC, side="data")
        graph = result.hypergraph
        for edge in graph.hyperedges:
            self.assertEqual(graph.vertex(edge.predicate_id).kind, "predicate")

    def test_roles_are_nonempty_and_reference_known_vertices(self) -> None:
        _local, result = _bridge(DOC, side="data")
        graph = result.hypergraph
        for edge in graph.hyperedges:
            self.assertTrue(edge.roles, edge.id)
            for role in edge.roles:
                graph.vertex(role.vertex_id)  # raises KeyError if unknown
                self.assertTrue(role.name)

    def test_query_placeholder_maps_to_query_kind(self) -> None:
        local, result = _bridge(QUERY, side="query", is_query=True)
        graph = result.hypergraph
        query_kinds = [v for v in graph.vertices if v.kind == "query"]
        local_queries = [v for v in local.vertices if v.is_query()]
        self.assertEqual(len(query_kinds), len(local_queries))
        for vertex in query_kinds:
            self.assertTrue(
                vertex.expected_type or vertex.semantic_type,
                f"query vertex {vertex.id} lacks expected/semantic type",
            )

    def test_non_query_vertices_have_no_expected_type(self) -> None:
        _local, result = _bridge(DOC, side="data")
        for vertex in result.hypergraph.vertices:
            self.assertIsNone(vertex.expected_type, vertex.id)

    def test_context_ids_carry_side(self) -> None:
        _local, q = _bridge(QUERY, side="query", is_query=True)
        _local, d = _bridge(DOC, side="data")
        for ctx in q.hypergraph.context_ids:
            self.assertTrue(ctx.startswith("query-"), ctx)
        for ctx in d.hypergraph.context_ids:
            self.assertTrue(ctx.startswith("data-"), ctx)


@unittest.skipUnless(_ENABLED and _SPACY_OK, "requires HUFU_RUN_E2E=1 and spacy")
class BridgeAllowedPairsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("HUFU_SPACY_MODEL", "en_core_web_sm")

    def test_domain_strategy_pairs_reference_core_vertices(self) -> None:
        from hufu.engine.bridge import bridge_allowed_pairs

        _ql, q = _bridge(QUERY, side="query", is_query=True)
        _dl, d = _bridge(DOC, side="data")
        pairs = bridge_allowed_pairs(_ql, _dl, strategy="domain")
        for q_id, d_id in pairs:
            q.hypergraph.vertex(q_id)
            d.hypergraph.vertex(d_id)

    def test_typed_strategy_pairs_placeholder_with_places(self) -> None:
        from hufu.engine.bridge import bridge_allowed_pairs

        ql, q = _bridge(QUERY, side="query", is_query=True)
        dl, d = _bridge(DOC, side="data")
        pairs = bridge_allowed_pairs(ql, dl, strategy="typed")
        self.assertTrue(pairs, "no allowed pairs between related query and doc")
        # The ?Where placeholder must pair with the answer-bearing places;
        # that coverage is what the consistency check quantifies over.
        placeholder_targets = {
            d.hypergraph.vertex(did).text
            for qid, did in pairs
            if q.hypergraph.vertex(qid).kind == "query"
        }
        self.assertIn("Oak Park", placeholder_targets)
        self.assertIn("Illinois", placeholder_targets)

    def test_typed_strategy_excludes_predicates(self) -> None:
        from hufu.engine.bridge import bridge_allowed_pairs

        ql, q = _bridge(QUERY, side="query", is_query=True)
        dl, d = _bridge(DOC, side="data")
        pairs = bridge_allowed_pairs(ql, dl, strategy="typed")
        for qid, did in pairs:
            self.assertTrue(q.hypergraph.vertex(qid).matchable, qid)
            self.assertTrue(d.hypergraph.vertex(did).matchable, did)


if __name__ == "__main__":
    unittest.main(verbosity=2)
