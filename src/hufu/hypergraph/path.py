"""Shortest-path searches over the vertex-hyperedge incidence graph."""

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Union

from hufu.hypergraph.hypergraph import Hyperedge, Hypergraph, Vertex


def find_shortest_hyperpaths(
    hypergraph: Hypergraph,
    pairs: List[Tuple[Vertex, Vertex]]
) -> Dict[Tuple[Vertex, Vertex], List[Hyperedge]]:
    """Find an unbounded shortest hyperedge path for each requested pair."""

    queries_by_source = defaultdict(list)
    for u, v in pairs:
        queries_by_source[u].append(v)
    result = {}
    for u, targets in queries_by_source.items():
        distances = {u: 0}
        parents: Dict[Union[Vertex, Hyperedge], Union[Vertex, Hyperedge, None]] = {u: None}
        queue = deque([u])
        while queue:
            curr = queue.popleft()
            if isinstance(curr, Vertex):
                for edge in hypergraph.contained_edges.get(curr, []):
                    if edge not in distances:
                        distances[edge] = distances[curr] + 1
                        parents[edge] = curr
                        queue.append(edge)
            elif isinstance(curr, Hyperedge):
                for vertex in curr.vertices:
                    if vertex not in distances:
                        distances[vertex] = distances[curr] + 1
                        parents[vertex] = curr
                        queue.append(vertex)
        for v in targets:
            if v not in distances:
                result[(u, v)] = []
            elif u == v:
                result[(u, v)] = []
            else:
                path = []
                curr_node = v
                while curr_node is not None:
                    if isinstance(curr_node, Hyperedge):
                        path.append(curr_node)
                    curr_node = parents[curr_node]
                path.reverse()
                result[(u, v)] = path
    return result
def find_shortest_hyperpaths_local(
    hypergraph: Hypergraph,
    pairs: List[Tuple[Vertex, Vertex]]
) -> Dict[Tuple[Vertex, Vertex], List[Hyperedge]]:
    """Find shortest paths without crossing source-hypergraph boundaries."""

    queries_by_source = defaultdict(list)
    for u, v in pairs:
        queries_by_source[u].append(v)
    result: Dict[Tuple[Vertex, Vertex], List[Hyperedge]] = {}
    for u, targets in queries_by_source.items():
        # Component id is part of the BFS state: after the first edge, a path
        # remains inside that edge's source context even through fused vertices.
        unset_component = object()
        start_state = (u, unset_component)
        distances: dict[tuple[Union[Vertex, Hyperedge], object], int] = {start_state: 0}
        parents: dict[
            tuple[Union[Vertex, Hyperedge], object],
            tuple[Union[Vertex, Hyperedge], object] | None,
        ] = {start_state: None}
        queue = deque([start_state])
        wanted = {v for v in targets if v != u}
        best_target_state: dict[Vertex, tuple[Union[Vertex, Hyperedge], object]] = {}
        while queue:
            curr_obj, curr_component = queue.popleft()
            if isinstance(curr_obj, Vertex):
                for edge in hypergraph.contained_edges.get(curr_obj, []):
                    edge_component = edge.hypergraph_id
                    if curr_component is not unset_component and edge_component != curr_component:
                        continue
                    next_state = (edge, edge_component)
                    if next_state in distances:
                        continue
                    distances[next_state] = distances[(curr_obj, curr_component)] + 1
                    parents[next_state] = (curr_obj, curr_component)
                    queue.append(next_state)
            else:
                for vertex in curr_obj.vertices:
                    next_state = (vertex, curr_component)
                    if next_state in distances:
                        continue
                    distances[next_state] = distances[(curr_obj, curr_component)] + 1
                    parents[next_state] = (curr_obj, curr_component)
                    queue.append(next_state)
                    if vertex in wanted and vertex not in best_target_state:
                        best_target_state[vertex] = next_state
            if len(best_target_state) == len(wanted):
                break
        for v in targets:
            if u == v:
                result[(u, v)] = []
                continue
            terminal_state = best_target_state.get(v)
            if terminal_state is None:
                result[(u, v)] = []
                continue
            path: list[Hyperedge] = []
            state: tuple[Union[Vertex, Hyperedge], object] | None = terminal_state
            while state is not None:
                obj, _ = state
                if isinstance(obj, Hyperedge):
                    path.append(obj)
                state = parents[state]
            path.reverse()
            result[(u, v)] = path
    return result
def find_shortest_hyperpaths_bounded(
    hypergraph: Hypergraph,
    pairs: List[Tuple[Vertex, Vertex]],
    max_hops: int,
) -> Dict[Tuple[Vertex, Vertex], List[Hyperedge]]:
    """Find shortest paths using at most ``max_hops`` hyperedges."""

    if max_hops < 0:
        return {(u, v): [] for u, v in pairs}
    queries_by_source = defaultdict(list)
    for u, v in pairs:
        queries_by_source[u].append(v)
    result: Dict[Tuple[Vertex, Vertex], List[Hyperedge]] = {}
    for u, targets in queries_by_source.items():
        dist_hops: dict[Vertex, int] = {u: 0}
        parent: dict[Vertex, tuple[Vertex, Hyperedge] | None] = {u: None}
        queue = deque([u])
        wanted = {v for v in targets if v != u}
        found: set[Vertex] = set()
        while queue:
            curr = queue.popleft()
            curr_hops = dist_hops[curr]
            if curr_hops >= max_hops:
                continue
            for edge in hypergraph.contained_edges.get(curr, []):
                for nxt in edge.vertices:
                    if nxt == curr:
                        continue
                    nxt_hops = curr_hops + 1
                    if nxt_hops > max_hops:
                        continue
                    if nxt not in dist_hops:
                        dist_hops[nxt] = nxt_hops
                        parent[nxt] = (curr, edge)
                        queue.append(nxt)
                        if nxt in wanted:
                            found.add(nxt)
            if len(found) == len(wanted):
                break
        for v in targets:
            if u == v:
                result[(u, v)] = []
                continue
            if v not in dist_hops:
                result[(u, v)] = []
                continue
            path: list[Hyperedge] = []
            cur = v
            while cur != u:
                prev_info = parent.get(cur)
                if prev_info is None:
                    path = []
                    break
                prev, edge = prev_info
                path.append(edge)
                cur = prev
            path.reverse()
            result[(u, v)] = path
    return result
def find_shortest_hyperpaths_local_bounded(
    hypergraph: Hypergraph,
    pairs: List[Tuple[Vertex, Vertex]],
    max_hops: int,
) -> Dict[Tuple[Vertex, Vertex], List[Hyperedge]]:
    """Find bounded shortest paths constrained to one source hypergraph."""

    if max_hops < 0:
        return {(u, v): [] for u, v in pairs}
    queries_by_source = defaultdict(list)
    for u, v in pairs:
        queries_by_source[u].append(v)
    result: Dict[Tuple[Vertex, Vertex], List[Hyperedge]] = {}
    for u, targets in queries_by_source.items():
        unset_component = object()
        start_state = (u, unset_component)
        dist_hops: dict[tuple[Vertex, object], int] = {start_state: 0}
        parent: dict[tuple[Vertex, object], tuple[tuple[Vertex, object], Hyperedge] | None] = {start_state: None}
        queue = deque([start_state])
        wanted = {v for v in targets if v != u}
        found_state: dict[Vertex, tuple[Vertex, object]] = {}
        while queue:
            curr_v, curr_comp = queue.popleft()
            curr_hops = dist_hops[(curr_v, curr_comp)]
            if curr_hops >= max_hops:
                continue
            for edge in hypergraph.contained_edges.get(curr_v, []):
                edge_comp = edge.hypergraph_id
                if curr_comp is not unset_component and edge_comp != curr_comp:
                    continue
                next_comp = edge_comp
                for nxt in edge.vertices:
                    if nxt == curr_v:
                        continue
                    next_state = (nxt, next_comp)
                    nxt_hops = curr_hops + 1
                    if nxt_hops > max_hops:
                        continue
                    if next_state in dist_hops:
                        continue
                    dist_hops[next_state] = nxt_hops
                    parent[next_state] = ((curr_v, curr_comp), edge)
                    queue.append(next_state)
                    if nxt in wanted and nxt not in found_state:
                        found_state[nxt] = next_state
            if len(found_state) == len(wanted):
                break
        for v in targets:
            if u == v:
                result[(u, v)] = []
                continue
            terminal = found_state.get(v)
            if terminal is None:
                result[(u, v)] = []
                continue
            path: list[Hyperedge] = []
            state = terminal
            while state != start_state:
                prev_info = parent.get(state)
                if prev_info is None:
                    path = []
                    break
                prev_state, edge = prev_info
                path.append(edge)
                state = prev_state
            path.reverse()
            result[(u, v)] = path
    return result
