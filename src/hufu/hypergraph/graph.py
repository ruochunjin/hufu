"""Lossy pairwise graph view of a linguistic hypergraph."""

from dataclasses import dataclass

from hufu.hypergraph.dependency import LocalDoc
from hufu.hypergraph.hypergraph import Hypergraph, Vertex


@dataclass
class Edge:
    """One labeled directed edge in the pairwise graph view."""

    src: Vertex
    dst: Vertex
    label: str
class Graph:
    """Pairwise projection used by consumers that cannot process hyperedges."""

    def __init__(self, vertices: list[Vertex], edges: list[Edge], doc: LocalDoc) -> None:
        self.vertices = vertices
        self.edges = edges
        self.doc = doc
    @classmethod
    def from_hypergraph(cls, hypergraph: Hypergraph) -> 'Graph':
        """Project each hyperedge into edges from its first incident vertex."""

        vertices = hypergraph.vertices
        edges: list[Edge] = []
        for he in hypergraph.hyperedges:
            if not he.vertices:
                continue
            src = he.vertices[0]
            for dst in he.vertices[1:]:
                label = dst.dep().name if dst.dep().name else "dep"
                edges.append(Edge(src=src, dst=dst, label=label))
        return cls(vertices=vertices, edges=edges, doc=hypergraph.doc)
