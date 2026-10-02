//! Python-compatible hypergraph objects and Hyper Simulation integration.
//!
//! This adapter converts Python-owned nodes, hyperedges, Delta associations,
//! and D-match tables into the generic Rust traits.  It preserves one stable
//! logical HC identifier across every anchor pair to avoid duplicated work.

use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
};

use graph_base::interfaces::{
    edge, graph,
    hypergraph::{self, ContainedHyperedge},
    typed, vertex,
};
use graph_simulation::{
    algorithm::hyper_simulation::{
        DMatch, Delta, HSEvent, HyperSimulation, HyperSimulationTrace, LMatch, LPredicate,
        SematicCluster,
    },
    utils::logger::TraceLog,
};
use pyo3::{exceptions::PyValueError, prelude::*};

// use graph_base::interfaces::hypergraph;

/// Python-visible vertex with a stable integer id and human-readable text.
#[derive(Clone, Debug, Eq)]
#[pyclass(name = "Node")]
pub struct Node {
    id: usize,
    desc: String,
}

impl Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

#[pymethods]
impl Node {
    #[new]
    pub fn new(id: usize, desc: String) -> Self {
        Node { id, desc }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn desc(&self) -> &String {
        &self.desc
    }
}

/// Python-visible n-ary hyperedge represented by its incident node ids.
#[derive(Clone, Debug, Eq)]
#[pyclass(name = "Hyperedge")]
pub struct Hyperedge {
    id_set: HashSet<usize>,
    desc: String,
    id: usize,
}

impl PartialEq for Hyperedge {
    fn eq(&self, other: &Self) -> bool {
        self.id_set == other.id_set
    }
}

impl Hash for Hyperedge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for id in &self.id_set {
            id.hash(state);
        }
    }
}

#[pymethods]
impl Hyperedge {
    #[new]
    pub fn new(id_set: HashSet<usize>, desc: String, id: usize) -> Self {
        Hyperedge { id_set, desc, id }
    }

    pub fn id_set(&self) -> Vec<usize> {
        self.id_set.iter().cloned().collect()
    }

    pub fn desc(&self) -> &String {
        &self.desc
    }
}

/// Python-owned hypergraph that implements the generic Rust graph traits.
#[pyclass(name = "Hypergraph")]
pub struct Hypergraph {
    nodes: Vec<Node>,
    hyperedges: Vec<Hyperedge>,
    type_same_fn: Option<Py<PyAny>>,   // (str, str) -> bool
    l_predicate_fn: Option<Py<PyAny>>, // (Hyperedge, Hyperedge) -> bool
}

/// Compatibility record for events read from the historical trace format.
#[pyclass(name = "Event")]
pub struct Event {
    pub phrase: String,
    pub sc_id: usize,
    pub binary_relation: HashSet<(usize, usize)>,
}

#[pymethods]
impl Event {
    #[new]
    pub fn new(phrase: String, sc_id: usize, binary_relation: HashSet<(usize, usize)>) -> Self {
        Event {
            phrase,
            sc_id,
            binary_relation,
        }
    }
}

#[pymethods]
impl Hypergraph {
    #[new]
    pub fn new() -> Self {
        Hypergraph {
            nodes: Vec::new(),
            hyperedges: Vec::new(),
            type_same_fn: None,
            l_predicate_fn: None,
        }
    }

    pub fn add_node(&mut self, desc: String) {
        self.nodes.push(Node {
            id: self.nodes.len(),
            desc: desc,
        });
    }

    pub fn add_hyperedge(&mut self, hyperedge: PyRef<Hyperedge>) {
        self.hyperedges.push(Hyperedge {
            id_set: hyperedge.id_set.clone(),
            desc: hyperedge.desc.clone(),
            id: self.hyperedges.len(),
        });
    }

    pub fn set_type_same_fn(&mut self, type_same_fn: Py<PyAny>) {
        self.type_same_fn = Some(type_same_fn);
    }

    pub fn set_l_predicate_fn(&mut self, l_predicate_fn: Py<PyAny>) {
        self.l_predicate_fn = Some(l_predicate_fn);
    }

    pub fn get_node_desc_by_id(&self, id: usize) -> Option<String> {
        self.nodes.get(id).map(|node| node.desc.clone())
    }

    #[staticmethod]
    pub fn hyper_simulation(
        query: PyRef<Hypergraph>,
        data: PyRef<Hypergraph>,
        l_match_fn: Py<PyAny>,
    ) -> HashMap<usize, HashSet<usize>> {
        let mut l_match = LMatchImpl::from(l_match_fn);
        let sim = HyperSimulation::get_simulation_naive(&*query, &*data, &mut l_match);
        // Convert HashMap<&Node, HashSet<&Node>> to HashMap<usize, HashSet<usize>>
        sim.into_iter()
            .map(|(k, v)| (k.id(), v.into_iter().map(|n| n.id()).collect()))
            .collect()
    }

    #[staticmethod]
    pub fn soft_hyper_simulation(
        query: PyRef<Hypergraph>,
        data: PyRef<Hypergraph>,
        l_match_fn: Py<PyAny>,
    ) -> HashMap<usize, HashSet<usize>> {
        let mut l_match = LMatchImpl::from(l_match_fn);
        let sim = HyperSimulation::get_soft_simulation_naive(&*query, &*data, &mut l_match);
        // Convert HashMap<&Node, HashSet<&Node>> to HashMap<usize, HashSet<usize>>
        sim.into_iter()
            .map(|(k, v)| (k.id(), v.into_iter().map(|n| n.id()).collect()))
            .collect()
    }

    #[staticmethod]
    pub fn get_hyper_simulation(
        query: PyRef<Hypergraph>,
        data: PyRef<Hypergraph>,
        delta: PyRef<DeltaPy>,
        d_match: PyRef<DMatchImpl>,
    ) -> HashMap<usize, HashSet<usize>> {
        let delta_inner = DeltaImpl::from(delta.clone(), &*query, &*data);
        let sim =
            HyperSimulation::get_hyper_simulation_effect(&*query, &*data, &delta_inner, &*d_match);
        sim.into_iter()
            .map(|(k, v)| (k.id(), v.into_iter().map(|n| n.id()).collect()))
            .collect()
    }

    #[staticmethod]
    pub fn get_hyper_simulation_naive(
        query: PyRef<Hypergraph>,
        data: PyRef<Hypergraph>,
        delta: PyRef<DeltaPy>,
        d_match: PyRef<DMatchImpl>,
    ) -> HashMap<usize, HashSet<usize>> {
        let delta_inner = DeltaImpl::from(delta.clone(), &*query, &*data);
        let sim =
            HyperSimulation::get_hyper_simulation_naive(&*query, &*data, &delta_inner, &*d_match);
        sim.into_iter()
            .map(|(k, v)| (k.id(), v.into_iter().map(|n| n.id()).collect()))
            .collect()
    }

    #[staticmethod]
    pub fn get_hyper_simulation_strict(
        query: PyRef<Hypergraph>,
        data: PyRef<Hypergraph>,
        delta: PyRef<DeltaPy>,
        d_match: PyRef<DMatchImpl>,
    ) -> HashMap<usize, HashSet<usize>> {
        let delta_inner = DeltaImpl::from(delta.clone(), &*query, &*data);
        let sim =
            HyperSimulation::get_hyper_simulation_strict(&*query, &*data, &delta_inner, &*d_match);
        sim.into_iter()
            .map(|(k, v)| (k.id(), v.into_iter().map(|n| n.id()).collect()))
            .collect()
    }

    pub fn get_hyper_simulation_trace(&self) -> Vec<Event> {
        let trace = HyperSimulationTrace::get_trace("hyper_simulation.trace").unwrap();
        let events = trace
            .into_iter()
            .map(|event| match event {
                HSEvent::Base(sc_id, relation) => Event {
                    phrase: "base".to_string(),
                    sc_id,
                    binary_relation: relation,
                },
                HSEvent::Derivation(sc_id, relation) => Event {
                    phrase: "derivation".to_string(),
                    sc_id,
                    binary_relation: relation,
                },
            })
            .collect();
        events
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node(id: {}, desc: {})", self.id, self.desc)
    }
}

impl graph::SingleId for Node {
    fn id(&self) -> usize {
        self.id
    }
}

impl vertex::Vertex for Node {}

impl edge::Hyperedge for Hyperedge {
    fn id_set(&self) -> HashSet<usize> {
        self.id_set.clone()
    }
}

impl hypergraph::IdVector for Hyperedge {
    fn id(&self) -> Vec<usize> {
        self.id_set.iter().cloned().collect()
    }
}

impl<'a> hypergraph::Hypergraph<'a> for Hypergraph {
    type Node = Node;
    type Edge = Hyperedge;

    fn new() -> Self {
        Hypergraph {
            nodes: Vec::new(),
            hyperedges: Vec::new(),
            type_same_fn: None,
            l_predicate_fn: None,
        }
    }

    fn nodes(&'a self) -> impl Iterator<Item = &'a Self::Node> {
        self.nodes.iter()
    }

    fn hyperedges(&'a self) -> impl Iterator<Item = &'a Self::Edge> {
        self.hyperedges.iter()
    }

    fn add_node(&mut self, node: Self::Node) {
        self.nodes.push(node);
    }

    fn add_hyperedge(&mut self, edge: Self::Edge) {
        self.hyperedges.push(edge);
    }

    fn get_node_by_id(&'a self, id: usize) -> Option<&'a Self::Node> {
        if id < self.nodes.len() {
            Some(&self.nodes[id])
        } else {
            None
        }
    }
}

impl<'a> typed::Typed<'a> for Hypergraph {
    fn type_same(&self, x: &Node, y: &Node) -> bool {
        if let Some(type_same_fn) = self.type_same_fn.as_ref() {
            // You cannot pass Rust trait objects directly to Python.
            // Instead, you need to define what data you want to compare and pass only that.
            // For example, if your Type trait has an id() method, you could do:
            Python::attach(|py| {
                type_same_fn
                    .call1(py, (&x.id, &y.id))
                    .map_or(false, |result| result.extract::<bool>(py).unwrap_or(false))
            })
        } else {
            false
        }
    }
}

impl<'a> ContainedHyperedge<'a> for Hypergraph {}

#[pyclass(name = "LMatch")]
pub struct LMatchImpl {
    l_match_fn: Option<Py<PyAny>>, // (Hyperedge, Hyperedge) -> dict[int, set[int]]
    l_match_cache: HashMap<(usize, usize), HashMap<usize, HashSet<usize>>>,
    empty_match: HashMap<usize, HashSet<usize>>,
    empty_set: HashSet<usize>,
}

#[pymethods]
impl LMatchImpl {
    #[new]
    pub fn new() -> Self {
        LMatchImpl {
            l_match_fn: None,
            l_match_cache: HashMap::new(),
            empty_match: HashMap::new(),
            empty_set: HashSet::new(),
        }
    }

    #[staticmethod]
    pub fn from(l_match_fn: Py<PyAny>) -> Self {
        LMatchImpl {
            l_match_fn: Some(l_match_fn),
            l_match_cache: HashMap::new(),
            empty_match: HashMap::new(),
            empty_set: HashSet::new(),
        }
    }
}

impl LMatch for LMatchImpl {
    type Edge = Hyperedge;

    fn new() -> Self {
        LMatchImpl {
            l_match_fn: None,
            l_match_cache: HashMap::new(),
            empty_match: HashMap::new(),
            empty_set: HashSet::new(),
        }
    }

    fn l_match_with_node_mut(
        &mut self,
        e: &Self::Edge,
        e_prime: &Self::Edge,
        u: usize,
    ) -> &HashSet<usize> {
        let l_match = self
            .l_match_cache
            .entry((e.id, e_prime.id))
            .or_insert_with(|| {
                if let Some(l_match_fn) = self.l_match_fn.as_ref() {
                    Python::attach(|py| {
                        l_match_fn
                            .call1(py, (e.clone(), e_prime.clone()))
                            .map_or_else(
                                |_| HashMap::new(),
                                |result| {
                                    result
                                        .extract::<HashMap<usize, HashSet<usize>>>(py)
                                        .unwrap_or_default()
                                },
                            )
                    })
                } else {
                    HashMap::new()
                }
            });

        l_match.entry(u).or_insert_with(HashSet::new)
    }

    fn l_match_with_node(&self, e: &Self::Edge, e_prime: &Self::Edge, u: usize) -> &HashSet<usize> {
        if let Some(let_match) = self.l_match_cache.get(&(e.id, e_prime.id)) {
            if let Some(match_set) = let_match.get(&u) {
                return match_set;
            }
        }
        return &self.empty_set;
    }

    fn dom(&self, e: &Self::Edge, e_prime: &Self::Edge) -> impl Iterator<Item = &usize> {
        if let Some(let_match) = self.l_match_cache.get(&(e.id, e_prime.id)) {
            return let_match.keys();
        } else {
            // If the cache does not contain the entry, return an empty iterator
            return self.empty_match.keys();
        }
    }
}

impl<'a> LPredicate<'a> for Hypergraph {
    fn l_predicate_node(&'a self, _u: &'a Self::Node, _v: &'a Self::Node) -> bool {
        // Node compatibility is supplied separately by `Typed::type_same`.
        true
    }

    fn l_predicate_edge(&'a self, e: &'a Self::Edge, e_prime: &'a Self::Edge) -> bool {
        if let Some(l_predicate_fn) = self.l_predicate_fn.as_ref() {
            Python::attach(|py| {
                l_predicate_fn
                    .call1(py, (e.clone(), e_prime.clone()))
                    .map_or(false, |result| result.extract::<bool>(py).unwrap_or(false))
            })
        } else {
            false
        }
    }

    fn l_predicate_set(
        &'a self,
        _x: &HashSet<&'a Self::Node>,
        _y: &HashSet<&'a Self::Node>,
    ) -> bool {
        // Set-level structure is handled by HC/D-match rather than this hook.
        true
    }
}

/// Frozen mapping from logical HC ids to semantic-role node-pair relations.
#[pyclass(name = "DMatch")]
pub struct DMatchImpl {
    d_match_cache: HashMap<(usize, usize), HashSet<(usize, usize)>>,
}

impl<'a> DMatch<'a> for DMatchImpl {
    type Edge = Hyperedge;

    fn d_match(
        &self,
        e: &SematicCluster<'a, Self::Edge>,
        e_prime: &SematicCluster<'a, Self::Edge>,
    ) -> &HashSet<(usize, usize)> {
        if let Some(set) = self.d_match_cache.get(&(e.id(), e_prime.id())) {
            return set;
        }
        unreachable!("D-Match not exist at ({}, {})", e.id(), e_prime.id())
    }

    // fn d_match_mut(&mut self, e: &SematicCluster<'a, Self::Edge>, e_prime: &SematicCluster<'a, Self::Edge>) -> &HashSet<(usize, usize)> {
    //     todo!()
    // }
}

#[pymethods]
impl DMatchImpl {
    #[new]
    fn new() -> Self {
        DMatchImpl {
            d_match_cache: HashMap::new(),
        }
    }

    #[staticmethod]
    fn from_dict(d_match_by_sc_id: HashMap<(usize, usize), HashSet<(usize, usize)>>) -> Self {
        DMatchImpl {
            d_match_cache: d_match_by_sc_id,
        }
    }
}

struct DeltaImpl<'a> {
    // ``SematicCluster`` keeps the historical upstream spelling for ABI
    // compatibility.  Python-facing code refers to the same object as an HC.
    sematic_cluster: HashMap<
        (&'a Node, &'a Node),
        Vec<(SematicCluster<'a, Hyperedge>, SematicCluster<'a, Hyperedge>)>,
    >,
}

impl<'a> Delta<'a> for DeltaImpl<'a> {
    type Edge = Hyperedge;
    type Node = Node;

    fn get_sematic_clusters(
        &'a self,
        u: &'a Self::Node,
        v: &'a Self::Node,
    ) -> &'a Vec<(
        SematicCluster<'a, Self::Edge>,
        SematicCluster<'a, Self::Edge>,
    )> {
        if let Some(pairs) = self.sematic_cluster.get(&(u, v)) {
            return pairs;
        }
        unreachable!("Have no sematic_clusters between {} to {}", u, v)
    }
}

impl<'a> DeltaImpl<'a> {
    fn from(delta: DeltaPy, query: &'a Hypergraph, data: &'a Hypergraph) -> Self {
        let mut pair_map: HashMap<
            (&Node, &Node),
            Vec<(SematicCluster<'a, Hyperedge>, SematicCluster<'a, Hyperedge>)>,
        > = HashMap::new();

        for u in &query.nodes {
            for v in &data.nodes {
                pair_map.entry((u, v)).or_default();
            }
        }

        let DeltaPy {
            sematic_cluster_cache,
            sematic_clusters,
            ..
        } = delta;

        for ((u_id, v_id), cluster_ids) in sematic_cluster_cache {
            let u = query.nodes.get(u_id).unwrap();
            let v = data.nodes.get(v_id).unwrap();
            for cluster_id in cluster_ids {
                let (q_edges_ids, d_edges_ids) =
                    sematic_clusters.get(&cluster_id).unwrap_or_else(|| {
                        unreachable!("Sematic cluster {} is not defined", cluster_id)
                    });
                let q_edges: Vec<&Hyperedge> = q_edges_ids
                    .iter()
                    .map(|id| query.hyperedges.get(*id).unwrap())
                    .collect();
                let d_edges: Vec<&Hyperedge> = d_edges_ids
                    .iter()
                    .map(|id| data.hyperedges.get(*id).unwrap())
                    .collect();

                let sc_q = SematicCluster::new(cluster_id, q_edges);
                let sc_d = SematicCluster::new(cluster_id, d_edges);
                pair_map.entry((u, v)).or_default().push((sc_q, sc_d));
            }
        }

        DeltaImpl {
            sematic_cluster: pair_map,
        }
    }
}

/// Python-side Delta registry separating HC definitions from anchor bindings.
#[derive(Clone)]
#[pyclass(name = "Delta")]
pub struct DeltaPy {
    // A logical HC pair is defined exactly once and may constrain many node
    // pairs.  Keeping definitions separate from associations avoids cloning
    // the cluster and D-match under a fresh id for every Delta(u, v) entry.
    sematic_clusters: HashMap<usize, (Vec<usize>, Vec<usize>)>,
    sematic_cluster_cache: HashMap<(usize, usize), Vec<usize>>, // (u_id, v_id) -> cluster ids
    global_cnt: usize,
}

impl DeltaPy {
    fn create_sematic_cluster(&mut self, q_edge_ids: Vec<usize>, d_edge_ids: Vec<usize>) -> usize {
        let id = self.global_cnt;
        self.global_cnt += 1;
        self.sematic_clusters.insert(id, (q_edge_ids, d_edge_ids));
        id
    }

    fn associate_sematic_cluster(&mut self, node_pair: (usize, usize), cluster_id: usize) {
        let associations = self.sematic_cluster_cache.entry(node_pair).or_default();
        if !associations.contains(&cluster_id) {
            associations.push(cluster_id);
        }
    }
}

#[pymethods]
impl DeltaPy {
    #[new]
    fn new() -> Self {
        DeltaPy {
            sematic_clusters: HashMap::new(),
            sematic_cluster_cache: HashMap::new(),
            global_cnt: 0,
        }
    }

    fn add_sematic_cluster_pair(
        &mut self,
        u: PyRef<Node>,
        v: PyRef<Node>,
        cluster_u: Vec<PyRef<Hyperedge>>,
        cluster_v: Vec<PyRef<Hyperedge>>,
    ) -> usize {
        let u_ids: Vec<_> = cluster_u.iter().map(|e_ref| e_ref.id).collect();
        let v_ids: Vec<_> = cluster_v.iter().map(|e_ref| e_ref.id).collect();
        let id = self.create_sematic_cluster(u_ids, v_ids);
        self.associate_sematic_cluster((u.id, v.id), id);
        id
    }

    fn add_sematic_cluster_pair_for_pairs(
        &mut self,
        node_pairs: Vec<(usize, usize)>,
        cluster_u: Vec<PyRef<Hyperedge>>,
        cluster_v: Vec<PyRef<Hyperedge>>,
    ) -> PyResult<usize> {
        // Define the logical HC once, then attach its stable id to every Delta
        // anchor.  This avoids repeating the same fixed-point dependency.
        if node_pairs.is_empty() {
            return Err(PyValueError::new_err(
                "a semantic cluster must be associated with at least one node pair",
            ));
        }

        let u_ids: Vec<_> = cluster_u.iter().map(|edge| edge.id).collect();
        let v_ids: Vec<_> = cluster_v.iter().map(|edge| edge.id).collect();
        let id = self.create_sematic_cluster(u_ids, v_ids);
        let mut seen = HashSet::new();
        for node_pair in node_pairs {
            if seen.insert(node_pair) {
                self.associate_sematic_cluster(node_pair, id);
            }
        }
        Ok(id)
    }

    fn cluster_count(&self) -> usize {
        self.sematic_clusters.len()
    }

    fn association_count(&self) -> usize {
        self.sematic_cluster_cache.values().map(Vec::len).sum()
    }
}

// fn test_sim(g: &Hypergraph, h: &Hypergraph)  {
//     let mut l_match = LMatchImpl::new();
//     let sim = HyperSimulation::get_simulation_naive(g, h, &mut l_match);
// }
