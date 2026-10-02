"""Minimal dependency-free example: run the HyperMatch reference core.

This script needs no model packages, no network, and no GPU — a toy
embedding backend and a role scorer are injected as plain callables.

Run from the repository root:  python examples/quickstart_core.py
"""

from hufu import (
    CallableRoleCellScorer,
    Cluster,
    Hyperedge,
    Hypergraph,
    Role,
    RoleScoringResult,
    Vertex,
    run_hypermatch,
)


def main() -> None:
    # Query: "Where was Alice born, and in which country is that place?"
    query = Hypergraph(
        (
            Vertex("qp1", "born", "PREDICATE", "predicate"),
            Vertex("qp2", "located in", "PREDICATE", "predicate"),
            Vertex("q_person", "Alice", "PERSON"),
            Vertex("q_place", "?birthplace", "LOC", "query", expected_type="LOC"),
            Vertex("q_country", "?country", "GPE", "query", expected_type="GPE"),
        ),
        (
            Hyperedge(
                "qe1", "qp1",
                (Role("person", "q_person"), Role("place", "q_place")),
                "query", canonical_frame="birth",
            ),
            Hyperedge(
                "qe2", "qp2",
                (Role("located", "q_place"), Role("region", "q_country")),
                "query", canonical_frame="location",
            ),
        ),
        "query",
    )

    # Retrieved documents (two contexts): one supporting, one conflicting.
    data = Hypergraph(
        (
            Vertex("dp1", "was born", "PREDICATE", "predicate"),
            Vertex("dp2", "is located in", "PREDICATE", "predicate"),
            Vertex("dp3", "is located in", "PREDICATE", "predicate"),
            Vertex("d_alice", "Alice", "PERSON", referent_id="ref:alice"),
            Vertex("d_paris", "Paris", "LOC", referent_id="ref:paris-france"),
            Vertex("d_france", "France", "GPE", referent_id="ref:france"),
            Vertex("d_paris_tx", "Paris", "LOC", referent_id="ref:paris-texas"),
            Vertex("d_texas", "Texas", "GPE", referent_id="ref:texas"),
        ),
        (
            Hyperedge("d1", "dp1", (Role("person", "d_alice"), Role("place", "d_paris")), "c1", canonical_frame="birth"),
            Hyperedge("d2", "dp2", (Role("located", "d_paris"), Role("region", "d_france")), "c1", canonical_frame="location"),
            Hyperedge("d3", "dp3", (Role("located", "d_paris_tx"), Role("region", "d_texas")), "c2", canonical_frame="location"),
        ),
        "data",
    )

    hv_allowed_pairs = {
        ("q_person", "d_alice"),
        ("q_place", "d_paris"),
        ("q_place", "d_paris_tx"),
        ("q_country", "d_france"),
        ("q_country", "d_texas"),
    }

    class ToyEmbedding:
        def encode(self, texts, *, prompt=None):
            del prompt
            return tuple((1.0, 0.0) for _ in texts)

    def score_roles(cells):
        return RoleScoringResult(
            {
                cell.pair: (1.0 if set(cell.query_roles) & set(cell.data_roles) else 0.0)
                for cell in cells
            }
        )

    output = run_hypermatch(
        query,
        data,
        hv_allowed_pairs=hv_allowed_pairs,
        hc_backend=ToyEmbedding(),
        role_scorer=CallableRoleCellScorer(score_roles),
        query_clusters=(Cluster.from_edges(query, ("qe1", "qe2"), context_id="query"),),
    )

    print("query consistent:", output.query_consistency.consistent)
    for mark in output.document_marks:
        print(f"  context {mark.context_id}: {mark.label}")


if __name__ == "__main__":
    main()
