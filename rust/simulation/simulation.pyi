
"""Type declarations for the optional ``simulation`` PyO3 extension.

The misspelling ``SematicCluster`` in the underlying ABI is retained for
backward compatibility; the public Delta helpers associate one such logical
cluster with every relevant `(u, v)` anchor pair.
"""

from typing import Callable, Dict, Optional
import networkx


def get_simulation_inter(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, is_label_cached=False) -> Dict: 
    """
    Get the simulation between two graphs.
    """

def is_simulation_isomorphic(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, is_label_cached=False) -> bool:
    """
    Check if two graphs are isomorphic by graph simulation.
    """

def get_simulation_inter_fn(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, compare_fn: Callable, is_label_cached=False) -> Dict: 
    """
    Get the simulation between two graphs.
    """
    
def is_simulation_isomorphic_fn(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, compare_fn: Callable, is_label_cached=False) -> bool:
    """
    Check if two graphs are isomorphic by graph simulation.
    """

def is_simulation_isomorphic_of_node_edge_fn(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, node_compare_fn: Callable,  edge_compare_fn: Callable, is_label_cached=False) -> bool:
    """
    Check if two graphs are isomorphic by graph simulation.
    """

def is_simulation_isomorphic_of_edge_fn(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, node_edge_compare_fn: Callable, is_label_cached=False) -> bool:
    """
    Check if two graphs are isomorphic by graph simulation.
    """

def get_bounded_simulation(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, compare: Callable, bound: Callable, is_label_cached=False) -> Dict:
    """
    Get the bounded simulation between two graphs.
    
    The compare function should take two node attribute dictionaries and return True if they have the same label.
    The bound function should take a node attribute dictionary and return an integer representing the bound value for that node.
    
    Returns a dictionary mapping nodes from nx_graph1 to sets of nodes from nx_graph2 that satisfy the bounded simulation.
    """

class Node:
    """
    Node for hypergraph.
    """
    def __init__(self, id: int, desc: str): ...
    def id(self) -> int: ...
    def desc(self) -> str: ...

class Hyperedge:
    """
    Hyperedge for hypergraph.
    """
    def __init__(self, id_set: set[int], desc: str, id: int): ...
    def id_set(self) -> set[int]: ...
    def desc(self) -> str: ...
    
class Event:
    """
    Event class.
    """
    phrase: str
    sc_id: int
    binary_relation: set[tuple[int, int]]
    
    def __init__(self, phrase: str, sc_id: int, binary_relation: set[tuple[int, int]]): ...

# (cluster_u, cluster_v)
class DMatch:
    """
    Frozen D-match relations consumed by Hyper Simulation.
    """
    def __init__(self) -> None: ...
    @staticmethod
    def from_dict(d_match_by_sc_id: dict[tuple[int, int], set[tuple[int, int]]]) -> 'DMatch': ...
    """
    Register D-match by the logical HC id returned by Delta.
    """

class Delta: # Delta(u, v) 
    """Register logical HC dependencies for one or more Delta anchors."""

    def __init__(self) -> None: ...
    def add_sematic_cluster_pair(self, u: Node, v: Node, cluster_u: list[Hyperedge], cluster_v: list[Hyperedge]) -> int: ...
    """
    Compatibility helper that creates one HC for one Delta anchor.
    """
    def add_sematic_cluster_pair_for_pairs(
        self,
        node_pairs: list[tuple[int, int]],
        cluster_u: list[Hyperedge],
        cluster_v: list[Hyperedge],
    ) -> int: ...
    """Create one logical HC and associate the same id with every anchor."""
    def cluster_count(self) -> int: ...
    def association_count(self) -> int: ...

# (u, v) (cluster_u, cluster_v)
# (u', v') 

class Hypergraph:
    """
    Hypergraph class.
    """
    def __init__(self): ...
    
    def add_node(self, desc: str): ...
    
    def add_hyperedge(self, hyperedge: Hyperedge): ...
    
    def set_type_same_fn(self, type_same_fn: Callable[[int, int], bool]): ... # L(v) = L(u)
    """
    Set a function as the Denial Comment, where inputs the id of the nodes, and return if is conflict.
    """
    
    def set_l_predicate_fn(self, l_predicate_fn: Callable[[Hyperedge, Hyperedge], bool]): ... # L_P(e1, e2)
    
    def get_node_desc_by_id(self, node_id: int) -> Optional[str]: ...
    
    def get_hyper_simulation_trace(self) -> list[Event]: 
        """
        Get trace events of last time hyper simulation.
        """
    
    @staticmethod
    def hyper_simulation(query: 'Hypergraph', data: 'Hypergraph', l_match_fn: Callable[[Hyperedge, Hyperedge], dict[int, set[int]]]) -> dict[int, set[int]]:
        """
        Hyper simulation.
        """
    
    @staticmethod
    def soft_hyper_simulation(query: 'Hypergraph', data: 'Hypergraph', l_match_fn: Callable[[Hyperedge, Hyperedge], dict[int, set[int]]]) -> dict[int, set[int]]:
        """
        Soft hyper simulation.
        """
        
    @staticmethod
    def get_hyper_simulation(query: 'Hypergraph', data: 'Hypergraph', delta: Delta, d_match: DMatch) -> dict[int, set[int]]:
        """
        Hyper Simulation using the indexed worklist implementation.
        """

    @staticmethod
    def get_hyper_simulation_naive(query: 'Hypergraph', data: 'Hypergraph', delta: Delta, d_match: DMatch) -> dict[int, set[int]]:
        """Reference fixed-point implementation used for parity testing and traces."""
        
    @staticmethod
    def get_hyper_simulation_strict(query: 'Hypergraph', data: 'Hypergraph', delta: Delta, d_match: DMatch) -> dict[int, set[int]]:
        """
        Strict Hyper Simulation
        """
