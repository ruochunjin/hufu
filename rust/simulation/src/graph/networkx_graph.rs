//! NetworkX-to-Rust adapters for classical and bounded graph simulation.
//!
//! Attribute comparison callbacks remain Python-owned while the fixed-point
//! traversal executes in Rust.  Conversion functions preserve NetworkX node
//! identities when translating results back to dictionaries and sets.

use graph_base::interfaces::vertex::Vertex;
use graph_simulation::algorithm::simulation::Simulation;
use graph_simulation::algorithm::bounded::{BoundedSimulation, Bounded};
use pyo3::types::PySet;
use pyo3::{prelude::*, types::PyDict};
use graph_base::interfaces::labeled::{Label, Labeled, LabeledAdjacency};
use graph_base::interfaces::graph::{Graph, Directed, Adjacency, AdjacencyInv, SingleId, IdPair, Degree, DegreeList};

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
// use std::path::Display;
// use std::sync::Arc;

// type SharedRustFn = Arc<dyn Fn(&Attributes, &Attributes) -> bool + Send + Sync>;


// Rust-owned graph representation of NetworkX nodes, edges, and attributes.

#[derive(Debug)]
struct Attributes(HashMap<String, Py<PyAny>>);

impl Clone for Attributes {
    fn clone(&self) -> Self {
        Python::attach(|py| {
            let cloned_map = self.0.iter()
                .map(|(k, v)| (k.clone(), v.clone_ref(py)))
                .collect();
            Attributes(cloned_map)
        })
    }
}

impl std::fmt::Display for Attributes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));

        write!(f, "{{")?;
        for (key, value) in entries {
            write!(f, "{}: {}, ", key, value)?;
        }
        write!(f, "}}")
    }
}

impl PartialEq for Attributes {
    fn eq(&self, other: &Self) -> bool {
        // Different key counts cannot represent equal attribute dictionaries.
        if self.0.len() != other.0.len() {
            return false;
        }

        Python::attach(|py| {
            self.0.iter().all(|(key, value)| {
                match other.0.get(key) {
                    Some(other_value) => {
                        match value.call_method1(py, "__eq__", (other_value,)) {
                            Ok(result) => result.extract::<bool>(py).unwrap_or(false),
                            Err(_) => false,
                        }
                    },
                    None => false,
                }
            })
        })
    }
}

impl Eq for Attributes {}

impl Hash for Attributes {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Sort keys so equivalent Python dictionaries have identical hashes.
        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));

        Python::attach(|py| {
            for (key, value) in entries {
                key.hash(state);
                // Reuse each Python object's own hash when it is available.
                match value.call_method0(py, "__hash__") {
                    Ok(hash_result) => {
                        if let Ok(hash_value) = hash_result.extract::<isize>(py) {
                            hash_value.hash(state);
                        }
                    },
                    Err(_) => {
                        // Use a deterministic sentinel for unhashable values.
                        0isize.hash(state);
                    }
                }
            }
        });
    }
}

impl Label for Attributes {
    fn label(&self) -> &str {
        ""
    }
}

/// Rust-owned node whose attributes retain their Python values.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Node {
    id: usize,
    attributes: Attributes,
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.id)
    }
}

impl SingleId for Node {
    fn id(&self) -> usize {
        self.id
    }
}

/// Directed edge with Python-originated attribute values.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Edge {
    source: usize,
    target: usize,
    attributes: Attributes,
}

impl IdPair for Edge {
    fn pair(&self) -> (usize, usize) {
        (self.source, self.target)
    }
}

/// PyO3 wrapper that caches indices and caller-provided label predicates.
#[pyclass]
pub struct NetworkXGraph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    node_indices: HashMap<String, usize>,
    same_label_fn: Option<Py<PyAny>>,
    same_edge_fn: Option<Py<PyAny>>,
    same_node_edge_fn: Option<Py<PyAny>>,
    same_label_cache: Option<HashSet<(usize, usize)>>,
    bound_values: HashMap<usize, usize>,  // Node id to its bounded-simulation radius.
}

impl Clone for NetworkXGraph {
    fn clone(&self) -> Self {
        Python::attach(|py| {
            NetworkXGraph {
                nodes: self.nodes.clone(),
                edges: self.edges.clone(),
                node_indices: self.node_indices.clone(),
                same_label_fn: self.same_label_fn.as_ref().map(|f| f.clone_ref(py)),
                same_edge_fn: self.same_edge_fn.as_ref().map(|f| f.clone_ref(py)),
                same_node_edge_fn: self.same_node_edge_fn.as_ref().map(|f| f.clone_ref(py)),
                same_label_cache: self.same_label_cache.clone(),
                bound_values: self.bound_values.clone(),
            }
        })
    }
}

fn convert_to_string(obj: &Py<PyAny>) -> PyResult<String> {
    Python::attach(|py| {
        // Try direct conversion first
        obj.call_method0(py, "__str__")?.extract::<String>(py)
            .or_else(|_| {
                // If that fails, try to convert to a string using repr
                obj.call_method0(py, "__repr__")?.extract::<String>(py)
            })
    })
}

#[pymethods]
impl NetworkXGraph {
    #[new]
    fn new() -> Self {
        NetworkXGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_indices: HashMap::new(),
            same_label_fn: None,
            same_edge_fn: None,
            same_node_edge_fn: None, 
            same_label_cache: None,
            bound_values: HashMap::new(),
        }
    }

    // Convert a NetworkX graph while retaining node and edge attributes.
    #[staticmethod]
    fn from_networkx(nx_graph: &Bound<'_, PyAny>) -> PyResult<Self> {
        let nodes = nx_graph.getattr("nodes")?.call_method1("items", ())?;
        let edges = nx_graph.getattr("edges")?.call_method1("data", ())?;

        let mut graph = NetworkXGraph::new();

        for node in nodes.try_iter()? {
            let node = node?;
            let id = node.call_method1("__getitem__", (0, ))?.extract::<PyObject>()?;
            let id = convert_to_string(&id)?;
            let attrs = node.call_method1("__getitem__", (1, ))?.extract::<HashMap<String, PyObject>>()?;
            graph.add_node(id, attrs);
        }
        for edge in edges.try_iter()? {
            let edge = edge?;
            let source = edge.call_method1("__getitem__", (0, ))?.extract::<PyObject>()?;
            let source = convert_to_string(&source)?;
            let target = edge.call_method1("__getitem__", (1, ))?.extract::<PyObject>()?;
            let target = convert_to_string(&target)?;
            let attrs = edge.call_method1("__getitem__", (2, ))?.extract::<HashMap<String, PyObject>>()?;
            graph.add_edge(source, target, attrs);
        }
        
        Ok(graph)
    }

    // Reconstruct a NetworkX graph from the Rust-owned representation.
    fn to_networkx(&self, py: Python) -> PyResult<PyObject> {
        let nx = py.import("networkx")?;
        let graph = nx.getattr("Graph")?.call0()?;

        // Restore nodes before edges so all edge endpoints exist.
        for node in &self.nodes {
            let attrs_dict = PyDict::new(py);
            for (k, v) in &node.attributes.0 {
                attrs_dict.set_item(k, v.clone_ref(py))?;
            }
            graph.call_method1(
                "add_node",
                (node.id.clone(), attrs_dict),
            )?;
        }

        // Clone Python attribute values back into the destination graph.
        for edge in &self.edges {
            let attrs_dict = PyDict::new(py);
            for (k, v) in &edge.attributes.0 {
                attrs_dict.set_item(k, v.clone_ref(py))?;
            }
            graph.call_method1(
                "add_edge",
                (
                    edge.source.clone(),
                    edge.target.clone(),
                    attrs_dict,
                ),
            )?;
        }

        Ok(graph.into())
    }

    // Mutation and inspection helpers exposed through PyO3.
    fn add_node(&mut self, id: String, attributes: HashMap<String, PyObject>) {
        let index = self.nodes.len();
        self.node_indices.insert(id.clone(), index);
        let attributes = Attributes(attributes);
        self.nodes.push(Node { id: index, attributes });
    }

    fn add_edge(
        &mut self,
        source: String,
        target: String,
        attributes: HashMap<String, PyObject>,
    ) {
        let attributes = Attributes(attributes);
        let source = *self.node_indices.get(&source).unwrap();
        let target = *self.node_indices.get(&target).unwrap();
        self.edges.push(Edge {
            source,
            target,
            attributes,
        });
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    // Return a Python-owned copy of one node's attributes.
    fn get_node_attributes(&self, node_id: &str) -> Option<HashMap<String, PyObject>> {
        self.node_indices.get(node_id).map(|&index| {
            Python::with_gil(|py| {
                self.nodes[index].attributes.0.iter()
                    .map(|(k, v)| (k.clone(), v.clone_ref(py)))
                    .collect()
            })
        })
    }

    // Return a Python-owned copy of one edge's attributes.
    fn get_edge_attributes(
        &self,
        source: &str,
        target: &str,
    ) -> Option<HashMap<String, PyObject>> {
        self.edges
            .iter()
            .find(|e| e.source == *self.node_indices.get(source).unwrap() 
                            && e.target == *self.node_indices.get(target).unwrap())
            .map(|e| Python::with_gil(|py| {
                e.attributes.0.iter()
                    .map(|(k, v)| (k.clone(), v.clone_ref(py)))
                    .collect()
            }))
    }

    fn register_compare_fn(&mut self, compare: Py<PyAny>) {
        self.same_label_fn = Some(compare);
    }

    fn register_edge_compare_fn(&mut self, compare: Py<PyAny>) {
        self.same_edge_fn = Some(compare);
    }

    fn register_node_edge_compare_fn(&mut self, compare: Py<PyAny>) {
        self.same_node_edge_fn = Some(compare);
    }

    fn build_compare_cache(&mut self, other: &NetworkXGraph) {

        let cache: HashSet<_> = self.nodes.iter().flat_map(|node1| {
            other.nodes.iter().filter_map(|node2| {
                if self.label_same(node1, node2) {
                    Some((node1.id, node2.id))
                } else {
                    None
                }
            }).collect::<HashSet<_>>()
        }).collect();

        self.same_label_cache = Some(cache);
    }

    fn set_bound_values(&mut self, bound_fn: Py<PyAny>) {
        // Evaluate the caller's radius function once per node and cache it.
        for node in &self.nodes {
            let bound_value = Python::attach(|py| {
                let node_attrs = node.attributes.0.iter()
                    .map(|(k, v)| (k.clone(), v.clone_ref(py)))
                    .collect::<HashMap<_, _>>();
                bound_fn.call1(py, (node_attrs,))
                    .ok()
                    .and_then(|result| result.extract::<usize>(py).ok())
                    .unwrap_or(0)
            });
            self.bound_values.insert(node.id, bound_value);
        }
    }
}

impl Vertex for Node {}

impl<'a> Graph<'a> for NetworkXGraph {
    type Node = Node;

    type Edge = Edge;

    fn new() -> Self {
        NetworkXGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_indices: HashMap::new(),
            same_label_fn: None,
            same_edge_fn: None,
            same_node_edge_fn: None,
            same_label_cache: None,
            bound_values: HashMap::new(),
        }
    }

    fn nodes(&'a self) -> impl Iterator<Item = &'a Self::Node> {
        self.nodes.iter()
    }

    fn edges(&'a self) -> impl Iterator<Item = &'a Self::Edge> {
        self.edges.iter()
    }

    fn get_edges_pair(&'a self) -> impl Iterator<Item = (&'a Self::Node, &'a Self::Node)> {
        let id_map: HashMap<_, _, std::collections::hash_map::RandomState> = HashMap::from_iter(self.nodes.iter().map(|node| (node.id, node)));
        self.edges.iter().map(|edge| (id_map.get(&edge.source).unwrap().clone(), id_map.get(&edge.target).unwrap().clone()) ).collect::<Vec<_>>().into_iter()
    }

    fn add_node(&mut self, node: Self::Node) {
        let index = self.nodes.len();
        self.node_indices.insert(format!("Node{}.", index), index);
        self.nodes.push(node);
    }

    fn add_edge(&mut self, edge: Self::Edge) {
        self.edges.push(edge);
    }
}

fn test_eq(a: &Py<PyAny>, b: &Py<PyAny>) -> bool {
    Python::attach(|py| {
        match a.call_method1(py, "__eq__", (b,)) {
            Ok(result) => result.extract::<bool>(py).unwrap_or(false),
            Err(_) => false
        }
    })
}

fn native_same_label_fn(a: &Attributes, b: &Attributes) -> bool {
    for (k, v) in &a.0 {
        if let Some(other_v) = b.0.get(k) {
            if test_eq(v, other_v) {
                continue;
            }
        } else {
            return false;
        }
    }
    true
}

impl<'a> Labeled<'a> for NetworkXGraph {
    fn label_same(&self, node: &Self::Node, label: &Self::Node) -> bool {

        if let Some(cache) = self.same_label_cache.as_ref() {
            return cache.contains(&(node.id, label.id));
        }

        if let Some(compare_fn) = self.same_label_fn.as_ref() {
            Python::attach(|py| {
                let attr1 = node.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let attr2 = label.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                compare_fn.call1(py,(attr1, attr2)).unwrap().extract::<bool>(py).unwrap()
            })
        } else {
            return native_same_label_fn(&node.attributes, &label.attributes);
        }
    }

    fn get_label(&'a self, node: &'a Self::Node) -> &'a impl Label {
        &node.attributes
    }

    fn get_edges_pair_label(&'a self) -> impl Iterator<Item = (&'a Self::Node, &'a Self::Node, &'a impl Label)> {
        let id_map: HashMap<_, _, std::collections::hash_map::RandomState> = HashMap::from_iter(self.nodes.iter().map(|node| (node.id, node)));
        self.edges.iter().map(move |edge| (id_map.get(&edge.source).unwrap().clone(), id_map.get(&edge.target).unwrap().clone(), &edge.attributes)).collect::<Vec<_>>().into_iter()
    }

    fn edge_label_same(&self, edge1: &Self::Edge, edge2: &Self::Edge) -> bool {
        if let Some(compare_fn) = self.same_edge_fn.as_ref() {
            Python::attach(|py| {
                let attr1 = edge1.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let attr2 = edge2.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                compare_fn.call1(py,(attr1, attr2)).unwrap().extract::<bool>(py).unwrap()
            })
        } else {
            native_same_label_fn(&edge1.attributes, &edge2.attributes)
        }
    }

    fn edge_node_label_same(&self, src1: &Self::Node, edge1: &Self::Edge, dst1: &Self::Node, src2: &Self::Node, edge2: &Self::Edge, dst2: &Self::Node) -> bool {
        if let Some(compare_fn) = self.same_node_edge_fn.as_ref() {
            Python::attach(|py| {
                let src1_attr = src1.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let dst1_attr = dst1.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let edge1_attr = edge1.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let src2_attr = src2.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let dst2_attr = dst2.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let edge2_attr = edge2.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                compare_fn.call1(py,(src1_attr, edge1_attr, dst1_attr, src2_attr, edge2_attr, dst2_attr)).unwrap().extract::<bool>(py).unwrap()
            })
        } else {
            native_same_label_fn(&src1.attributes, &src2.attributes) 
                && native_same_label_fn(&dst1.attributes, &dst2.attributes) 
                && native_same_label_fn(&edge1.attributes, &edge2.attributes)
        }
    }
}

impl LabeledAdjacency<'_> for NetworkXGraph {}

impl std::fmt::Display for NetworkXGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NetworkXGraph(\n")?;
        write!(f, "Nodes: [\n")?;
        for node in &self.nodes {
            write!(f, "  {},\n", node)?;
        }
        write!(f, "],\n")?;
        write!(f, "Edges: [\n")?;
        for edge in &self.edges {
            write!(f, "{} -> {}, ", edge.source, edge.target)?;
        }
        write!(f, "]\n")?;
        write!(f, ")")
    }
}

fn to_nx_node(_py: Python, node: &Node) -> PyResult<Py<PyAny>> {
    // Return the node id as a Python object
    Python::attach(|py| {
        let builtins = py.import("builtins")?;
        let int_fn = builtins.getattr("int")?;
        let val = int_fn.call1((node.id,))?;
        Ok(val.into())
    })
}

/// Return the greatest simulation relation between two NetworkX graphs.
#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, is_label_cached = false))]
pub fn get_simulation_inter(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, is_label_cached: bool) -> PyResult<Py<PyAny>> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;

    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    let sim = graph1.get_simulation_inter(&graph2);
    

    // Convert simulation to a list of pairs (i, j) where i is a node in graph1, j is a node in graph2
    Python::attach(|py| {
        let map = PyDict::new(py);
        
        for (node, set) in sim.iter() {
            let py_set = PySet::new(py, set.iter().map(|node| to_nx_node(py, node)).collect::<PyResult<Vec<_>>>()?)?;
            map.set_item(to_nx_node(py, node)?, py_set)?;
        }
    
        Ok(map.into())
    })
}

/// Test mutual simulation using default node-label equality.
#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, is_label_cached = false))]
pub fn is_simulation_isomorphic(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, is_label_cached: bool) -> PyResult<bool> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;

    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    Ok(NetworkXGraph::has_simulation(graph1.get_simulation_inter(&graph2)))
}

/// Compute inter-graph simulation with a Python node comparator.
#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, compare, is_label_cached = false))]
pub fn get_simulation_inter_fn(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, compare: Py<PyAny>, is_label_cached: bool) -> PyResult<Py<PyAny>> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;

    graph1.register_compare_fn(compare);
    
    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    let sim = graph1.get_simulation_inter(&graph2);

    Python::attach(|py| {
        let map = PyDict::new(py);
        
        for (node, set) in sim.iter() {
            let py_set = PySet::new(py, set.iter().map(|node| to_nx_node(py, node)).collect::<PyResult<Vec<_>>>()?)?;
            map.set_item(to_nx_node(py, node)?, py_set)?;
        }
    
        Ok(map.into())
    })
}

/// Test mutual simulation with a Python node comparator.
#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, compare, is_label_cached = false))]
pub fn is_simulation_isomorphic_fn(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, compare: Py<PyAny>, is_label_cached: bool) -> PyResult<bool> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;
    
    graph1.register_compare_fn(compare);

    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    Ok(NetworkXGraph::has_simulation(graph1.get_simulation_inter(&graph2)))
}

/// Test simulation isomorphism with independent node and edge comparators.
#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, node_compare, edge_compare, is_label_cached = false))]
pub fn is_simulation_isomorphic_of_node_edge_fn(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, node_compare: Py<PyAny>, edge_compare: Py<PyAny>, is_label_cached: bool) -> PyResult<bool> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;
    
    graph1.register_compare_fn(node_compare);
    graph1.register_edge_compare_fn(edge_compare);

    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    Ok(NetworkXGraph::has_simulation(graph1.get_simulation_of_node_edge(&graph2)))
}

/// Test simulation isomorphism with a joint node-edge comparator.
#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, node_edge_compare, is_label_cached = false))]
pub fn is_simulation_isomorphic_of_edge_fn(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, node_edge_compare: Py<PyAny>, is_label_cached: bool) -> PyResult<bool> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;
    
    graph1.register_node_edge_compare_fn(node_edge_compare);

    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    Ok(NetworkXGraph::has_simulation(graph1.get_simulation_of_edge(&graph2)))
}

impl Directed for NetworkXGraph {}

impl Adjacency<'_> for NetworkXGraph {}

impl AdjacencyInv<'_> for NetworkXGraph {}

impl<'a> Degree<'a> for NetworkXGraph {
    fn get_out_degree(&'a self) -> DegreeList<'a, Self> {
        let mut degree_map = HashMap::new();
        
        for node in &self.nodes {
            let out_degree = self.edges.iter()
                .filter(|edge| edge.source == node.id)
                .count();
            degree_map.insert(node, out_degree);
        }
        
        // DegreeList has private fields upstream; transmute the equivalent map.
        unsafe { std::mem::transmute::<HashMap<&'a Self::Node, usize>, DegreeList<'a, Self>>(degree_map) }
    }

    fn get_in_degree(&'a self) -> DegreeList<'a, Self> {
        let mut degree_map = HashMap::new();
        
        for node in &self.nodes {
            let in_degree = self.edges.iter()
                .filter(|edge| edge.target == node.id)
                .count();
            degree_map.insert(node, in_degree);
        }
        
        // DegreeList has private fields upstream; transmute the equivalent map.
        unsafe { std::mem::transmute::<HashMap<&'a Self::Node, usize>, DegreeList<'a, Self>>(degree_map) }
    }

    fn out_degree(&'a self, degree_list: &DegreeList<'a, Self>, node: &Self::Node) -> usize {
        // Read the upstream private map through its known representation.
        unsafe { 
            let degree_map = &*(degree_list as *const _ as *const HashMap<&'a Self::Node, usize>);
            *degree_map.get(node).unwrap_or(&0)
        }
    }

    fn in_degree(&'a self, degree_list: &DegreeList<'a, Self>, node: &Self::Node) -> usize {
        // Read the upstream private map through its known representation.
        unsafe { 
            let degree_map = &*(degree_list as *const _ as *const HashMap<&'a Self::Node, usize>);
            *degree_map.get(node).unwrap_or(&0)
        }
    }
}

impl<'a> Bounded<'a> for NetworkXGraph {
    fn get_bound(&'a self, u: &'a Self::Node, v: &'a Self::Node) -> usize {
        // This adapter defines the pair bound from the source node's cached
        // radius; callers can encode pair-specific behavior in their callback.
        *self.bound_values.get(&u.id).unwrap_or(&0)
    }
}

/// Compute bounded simulation using caller-provided label and radius functions.
#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, compare, bound, is_label_cached = false))]
pub fn get_bounded_simulation(
    nx_graph1: &Bound<'_, PyAny>, 
    nx_graph2: &Bound<'_, PyAny>, 
    compare: Py<PyAny>,
    bound: Py<PyAny>,
    is_label_cached: bool
) -> PyResult<Py<PyAny>> {
    // 1. Convert both NetworkX inputs into Rust-owned graphs.
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;
    
    // 2. Register the caller-provided label comparator.
    graph1.register_compare_fn(compare);
    
    // 3. Materialize the bound for every query-side node.
    graph1.set_bound_values(bound);
    
    // 4. Optionally precompute cross-graph label compatibility.
    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }
    
    // 5. Run the bounded-simulation fixed point.
    let sim = graph1.get_bounded_simulation(&graph2);
    
    // 6. Convert borrowed Rust nodes back to Python sets.
    Python::attach(|py| {
        let map = PyDict::new(py);
        
        for (node, set) in sim.iter() {
            let py_set = PySet::new(py, set.iter().map(|node| to_nx_node(py, node)).collect::<PyResult<Vec<_>>>()?)?;
            map.set_item(to_nx_node(py, node)?, py_set)?;
        }
    
        Ok(map.into())
    })
}

// Historical standalone module definition retained as a reference.
// #[pymodule]
// pub fn networkx_graph(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_class::<NetworkXGraph>()?;
//     Ok(())
// }
