"""Self-contained parity tests for the optional PyO3 worklist backend.

Build/install ``src/simulation`` with maturin before running this file.  The
test is skipped when the extension is unavailable; the pure-Python reference
core can be tested independently.
"""

from __future__ import annotations

import os
import tempfile
import unittest

try:
    from simulation import DMatch, Delta, Hyperedge, Hypergraph
except ImportError:  # the compiled extension is an optional runtime component
    DMatch = Delta = Hyperedge = Hypergraph = None

CURRENT_BINDING_AVAILABLE = bool(
    Delta is not None
    and hasattr(Delta(), "add_sematic_cluster_pair_for_pairs")
    and hasattr(Hypergraph, "get_hyper_simulation_naive")
)

@unittest.skipUnless(
    CURRENT_BINDING_AVAILABLE,
    "the current maturin extension is not installed",
)
class RustHyperSimulationTests(unittest.TestCase):
    """Check shared-HC registration and indexed/naive solver parity."""

    @staticmethod
    def _graphs(size: int):
        query = Hypergraph()
        data = Hypergraph()
        for index in range(size):
            query.add_node(f"q{index}")
            data.add_node(f"d{index}")
        query.set_type_same_fn(lambda left, right: left == right)
        data.set_type_same_fn(lambda left, right: left == right)
        return query, data

    def test_one_logical_hc_can_constrain_multiple_delta_anchors(self) -> None:
        query, data = self._graphs(2)
        query_edge = Hyperedge({0, 1}, "query fact", 0)
        data_edge = Hyperedge({0, 1}, "data fact", 0)
        query.add_hyperedge(query_edge)
        data.add_hyperedge(data_edge)

        delta = Delta()
        anchors = [(0, 0), (1, 1)]
        cluster_id = delta.add_sematic_cluster_pair_for_pairs(
            anchors, [query_edge], [data_edge]
        )
        dmatch = DMatch.from_dict(
            {(cluster_id, cluster_id): {(0, 0), (1, 1)}}
        )

        indexed, naive = self._run_both(query, data, delta, dmatch)
        self.assertEqual(indexed, naive)
        self.assertEqual(indexed, {0: {0}, 1: {1}})
        self.assertEqual(delta.cluster_count(), 1)
        self.assertEqual(delta.association_count(), 2)

    def test_indexed_worklist_matches_naive_on_a_three_step_cascade(self) -> None:
        query, data = self._graphs(3)
        query_edges = []
        data_edges = []
        for index in range(3):
            query_edge = Hyperedge({index}, f"query fact {index}", index)
            data_edge = Hyperedge({index}, f"data fact {index}", index)
            query.add_hyperedge(query_edge)
            data.add_hyperedge(data_edge)
            query_edges.append(query_edge)
            data_edges.append(data_edge)

        delta = Delta()
        cluster_ids = [
            delta.add_sematic_cluster_pair_for_pairs(
                [(index, index)], [query_edges[index]], [data_edges[index]]
            )
            for index in range(3)
        ]
        dmatch = DMatch.from_dict(
            {
                (cluster_ids[0], cluster_ids[0]): {(0, 0), (1, 1)},
                (cluster_ids[1], cluster_ids[1]): {(1, 1), (2, 2)},
                (cluster_ids[2], cluster_ids[2]): set(),
            }
        )

        indexed, naive = self._run_both(query, data, delta, dmatch)
        self.assertEqual(indexed, naive)
        self.assertEqual(indexed, {0: set(), 1: set(), 2: set()})

    @staticmethod
    def _run_both(query, data, delta, dmatch):
        # A fresh empty working directory catches accidental dependencies on a
        # pre-existing logs/ directory or repository-relative data.
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary:
            os.chdir(temporary)
            try:
                indexed = Hypergraph.get_hyper_simulation(
                    query, data, delta, dmatch
                )
                naive = Hypergraph.get_hyper_simulation_naive(
                    query, data, delta, dmatch
                )
            finally:
                os.chdir(previous)
        return indexed, naive


if __name__ == "__main__":
    unittest.main(verbosity=2)
