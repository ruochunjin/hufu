//! Hyper Simulation fixed-point algorithms.
//!
//! The primary solver starts from the non-conflict relation `h_v` and applies
//! two invariants until no pair can be removed: every Delta anchor occurs in
//! each associated D-match, and every D-match is contained in the current
//! relation.  The indexed implementation maintains reverse dependency maps
//! and a worklist; the naive implementation remains available for parity
//! testing.

use log::info;
use std::collections::{HashMap, HashSet, VecDeque};
// use std::fs::File;
// use std::io::{self, Write};

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use graph_base::interfaces::{
    edge::Hyperedge,
    graph::SingleId,
    hypergraph::{ContainedHyperedge, Hypergraph},
    typed::Typed,
};

use crate::utils::logger::init_global_logger_once;
use crate::utils::logger::TraceLog;

/// Supplies the initial label-compatible relation used by legacy solvers.
pub trait LMatch {
    type Edge;
    // fn l_match(&'a self, e: &'a Self::Edge, e_prime: &'a Self::Edge) -> HashMap<&'a Self::Node, &'a HashSet<&'a Self::Node>>;
    fn new() -> Self;
    fn l_match_with_node_mut(
        &mut self,
        e: &Self::Edge,
        e_prime: &Self::Edge,
        u: usize,
    ) -> &HashSet<usize>;
    fn l_match_with_node(&self, e: &Self::Edge, e_prime: &Self::Edge, u: usize) -> &HashSet<usize>;
    fn dom(&self, e: &Self::Edge, e_prime: &Self::Edge) -> impl Iterator<Item = &usize>;
}

/// One logical hyperedge cluster identified independently of its Delta anchors.
///
/// `SematicCluster` retains the historical misspelling for ABI compatibility.
#[derive(Hash)]
pub struct SematicCluster<'a, E: Hyperedge> {
    id: usize,
    hyperedges: Vec<&'a E>,
}

impl<'a, E: Hyperedge> SematicCluster<'a, E> {
    pub fn new(id: usize, hyperedges: Vec<&'a E>) -> Self {
        Self { id, hyperedges }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn hyperedges(&self) -> &Vec<&'a E> {
        &self.hyperedges
    }
}

/// Maps a candidate node pair `(u, v)` to all HC pairs constraining it.
pub trait Delta<'a> {
    type Node;
    type Edge: Hyperedge;
    fn get_sematic_clusters(
        &'a self,
        u: &'a Self::Node,
        v: &'a Self::Node,
    ) -> &'a Vec<(
        SematicCluster<'a, Self::Edge>,
        SematicCluster<'a, Self::Edge>,
    )>;
}

/// Returns the frozen semantic-role relation for one accepted HC pair.
pub trait DMatch<'a> {
    type Edge: Hyperedge;
    // fn d_match_mut(&mut self, e: &SematicCluster<'a, Self::Edge>, e_prime: &SematicCluster<'a, Self::Edge>) -> &HashSet<(usize, usize)>;
    fn d_match(
        &self,
        e: &SematicCluster<'a, Self::Edge>,
        e_prime: &SematicCluster<'a, Self::Edge>,
    ) -> &HashSet<(usize, usize)>;
}

/// Structural and label predicates required by the simulation algorithms.
pub trait LPredicate<'a>: Hypergraph<'a> {
    fn l_predicate_node(&'a self, u: &'a Self::Node, v: &'a Self::Node) -> bool;
    fn l_predicate_edge(&'a self, e: &'a Self::Edge, e_prime: &'a Self::Edge) -> bool;
    fn l_predicate_set(&'a self, x: &HashSet<&'a Self::Node>, y: &HashSet<&'a Self::Node>) -> bool;
}

/// Greatest-fixed-point algorithms for graph and hypergraph simulation.
pub trait HyperSimulation<'a>: Hypergraph<'a> {
    fn get_simulation_fixpoint(
        &'a self,
        _other: &'a Self,
        _l_match: &mut impl LMatch<Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;
    fn get_simulation_recursive(
        &'a self,
        _other: &'a Self,
        _l_match: &mut impl LMatch<Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;
    fn get_simulation_naive(
        &'a self,
        other: &'a Self,
        l_match: &mut impl LMatch<Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;
    fn get_soft_simulation_naive(
        &'a self,
        other: &'a Self,
        l_match: &mut impl LMatch<Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;
    fn get_hyper_simulation_naive(
        &'a self,
        other: &'a Self,
        delta: &'a impl Delta<'a, Node = Self::Node, Edge = Self::Edge>,
        d_match: &impl DMatch<'a, Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;
    /// Compute Hyper Simulation with reverse dependency indices and a worklist.
    fn get_hyper_simulation_effect(
        &'a self,
        other: &'a Self,
        delta: &'a impl Delta<'a, Node = Self::Node, Edge = Self::Edge>,
        d_match: &impl DMatch<'a, Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;
    fn get_hyper_simulation_effect_pass_by(
        &'a self,
        other: &'a Self,
        delta: &'a impl Delta<'a, Node = Self::Node, Edge = Self::Edge>,
        d_match: &impl DMatch<'a, Edge = Self::Edge>,
        type_same_lookup: &HashMap<&'a Self::Node, HashSet<&'a Self::Node>>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;
    fn get_hyper_simulation_effect_by_id(
        &'a self,
        hc_map: &HashMap<(usize, usize), Vec<((usize, usize), HashSet<(usize, usize)>)>>,
    ) -> HashSet<(usize, usize)>;
    fn get_hyper_simulation_strict(
        &'a self,
        other: &'a Self,
        delta: &'a impl Delta<'a, Node = Self::Node, Edge = Self::Edge>,
        d_match: &impl DMatch<'a, Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;
}
// struct MultiWriter<W1: Write, W2: Write> {
//     w1: W1,
//     w2: W2,
// }

// impl<W1: Write, W2: Write> Write for MultiWriter<W1, W2> {
//     fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
//         self.w1.write_all(buf)?;
//         self.w2.write_all(buf)?;
//         Ok(buf.len())
//     }
//     fn flush(&mut self) -> io::Result<()> {
//         self.w1.flush()?;
//         self.w2.flush()
//     }
// }

impl<'a, H> HyperSimulation<'a> for H
where
    H: Hypergraph<'a> + Typed<'a> + LPredicate<'a> + ContainedHyperedge<'a>,
{
    fn get_simulation_fixpoint(
        &'a self,
        _other: &'a Self,
        _l_match: &mut impl LMatch<Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        todo!()
    }

    fn get_simulation_recursive(
        &'a self,
        _other: &'a Self,
        _l_match: &mut impl LMatch<Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        todo!()
    }

    fn get_simulation_naive(
        &'a self,
        other: &'a Self,
        l_match: &mut impl LMatch<Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        // let log_file = File::create("hyper-simulation.log")
        //     .expect("Failed to create log file");
        // let multi_writer = MultiWriter {
        //     w1: log_file,
        //     w2: io::stdout(),
        // };

        // env_logger::Builder::new()
        //     .target(env_logger::Target::Pipe(Box::new(multi_writer)))
        //     .init();

        init_global_logger_once("hyper-simulation.log");

        info!("Start Naive Hyper Simulation");

        let self_contained_hyperedge = self.get_hyperedges_list();
        let other_contained_hyperedge = other.get_hyperedges_list();

        let mut simulation: HashMap<&Self::Node, HashSet<&Self::Node>> = self
            .nodes()
            .map(|u| {
                let res = other
                    .nodes()
                    .filter(|v| {
                        if self.type_same(u, *v) {
                            // For each e, compute the union of l_match(u) over all matching e_prime,
                            // then take the intersection across all e.
                            let mut l_match_intersection: Option<HashSet<usize>> = None;
                            for e in self.contained_hyperedges(&self_contained_hyperedge, u) {
                                let mut l_match_union: HashSet<usize> = HashSet::new();
                                for e_prime in
                                    other.contained_hyperedges(&other_contained_hyperedge, v)
                                {
                                    if self.l_predicate_edge(e, e_prime) {
                                        // let l_match = self.l_match(e, e_prime);
                                        let id_set = l_match.l_match_with_node(e, e_prime, u.id());
                                        l_match_union =
                                            l_match_union.union(&id_set).copied().collect();
                                    }
                                }
                                l_match_intersection = match l_match_intersection {
                                    Some(ref acc) => {
                                        Some(acc.intersection(&l_match_union).copied().collect())
                                    }
                                    None => Some(l_match_union),
                                };
                            }
                            if let Some(l_match_intersection) = l_match_intersection {
                                if l_match_intersection.contains(&v.id()) {
                                    return true;
                                }
                            }
                        }
                        false
                    })
                    .collect();
                (u, res)
            })
            .collect();

        info!("END Initial, sim: is ");
        for (u, v_set) in &simulation {
            info!(
                "\tsim({}) = {:?}",
                u.id(),
                v_set.iter().map(|v| v.id()).collect::<Vec<_>>()
            );
        }

        let mut changed = true;
        while changed {
            changed = false;
            for u in self.nodes() {
                let mut need_delete = Vec::new();
                for v in simulation.get(u).unwrap() {
                    info!("Checking {} -> {}", u.id(), v.id());
                    let mut _delete = true;
                    for e in self.contained_hyperedges(&self_contained_hyperedge, u) {
                        if !_delete {
                            break;
                        }
                        for e_prime in other.contained_hyperedges(&other_contained_hyperedge, v) {
                            if self.l_predicate_edge(e, e_prime) {
                                if l_match.dom(e, e_prime).all(|u_prime| {
                                    l_match
                                        .l_match_with_node(e, e_prime, u_prime.clone())
                                        .iter()
                                        .map(|id| other.get_node_by_id(*id))
                                        .any(|v_prime| {
                                            if let Some(v_prime) = v_prime {
                                                return simulation
                                                    .get(u)
                                                    .unwrap()
                                                    .contains(v_prime);
                                            } else {
                                                return false;
                                            }
                                        })
                                }) {
                                    info!("Keeping {} -> {}", u.id(), v.id());
                                    _delete = false;
                                    break;
                                }
                            }
                        }
                    }
                    if _delete {
                        info!("Deleting {} -> {}", u.id(), v.id());
                        need_delete.push(*v);
                    }
                }
                for v in need_delete {
                    simulation.get_mut(u).unwrap().remove(v);
                    changed = true;
                }
            }
        }

        simulation
    }

    fn get_soft_simulation_naive(
        &'a self,
        other: &'a Self,
        l_match: &mut impl LMatch<Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        init_global_logger_once("hyper-simulation.log");

        info!("Start Naive Hyper Simulation");

        // let self_contained_hyperedge = self.get_hyperedges_list();
        // let other_contained_hyperedge = other.get_hyperedges_list();

        let mut l_predicate_edges: HashMap<(usize, usize), Vec<(&Self::Edge, &Self::Edge)>> =
            HashMap::new();
        for e in self.hyperedges() {
            for e_prime in other.hyperedges() {
                if self.l_predicate_edge(e, e_prime) {
                    for u in e.id_set() {
                        for v in e_prime.id_set() {
                            l_predicate_edges
                                .entry((u, v))
                                .or_default()
                                .push((e, e_prime));
                        }
                    }
                }
            }
        }

        let mut simulation: HashMap<&'a Self::Node, HashSet<&'a Self::Node>> = self
            .nodes()
            .map(|u| {
                let res = other
                    .nodes()
                    .filter(|v| {
                        if self.type_same(u, *v) {
                            if let Some(edge_pairs) = l_predicate_edges.get(&(u.id(), v.id())) {
                                for (e, e_prime) in edge_pairs {
                                    let id_set = l_match.l_match_with_node(e, e_prime, u.id());
                                    if !id_set.contains(&v.id()) {
                                        return false;
                                    }
                                }
                                return true;
                            } else {
                                return true;
                            }
                        }
                        false
                    })
                    .collect();
                (u, res)
            })
            .collect();

        info!("END Initial, sim: is ");
        for (u, v_set) in &simulation {
            info!(
                "\tsim({}) = {:?}",
                u.id(),
                v_set.iter().map(|v| v.id()).collect::<Vec<_>>()
            );
        }

        let mut changed = true;
        while changed {
            changed = false;
            for u in self.nodes() {
                let mut need_delete = Vec::new();
                for v in simulation.get(u).unwrap() {
                    info!("Checking {} -> {}", u.id(), v.id());
                    let mut _delete = false;

                    if let Some(edge_pairs) = l_predicate_edges.get(&(u.id(), v.id())) {
                        for (e, e_prime) in edge_pairs {
                            if l_match.dom(e, e_prime).all(|u_prime| {
                                l_match
                                    .l_match_with_node(e, e_prime, u_prime.clone())
                                    .iter()
                                    .map(|id| other.get_node_by_id(*id))
                                    .any(|v_prime| {
                                        if let Some(v_prime) = v_prime {
                                            return simulation.get(u).unwrap().contains(v_prime);
                                        } else {
                                            return false;
                                        }
                                    })
                            }) {
                                info!("Keeping {} -> {}", u.id(), v.id());
                                _delete = true;
                                break;
                            }
                        }
                    }

                    if _delete {
                        info!("Deleting {} -> {}", u.id(), v.id());
                        need_delete.push(*v);
                    }
                }

                for v in need_delete {
                    simulation.get_mut(u).unwrap().remove(v);
                    changed = true;
                }
            }
        }

        simulation
    }

    fn get_hyper_simulation_naive(
        &'a self,
        other: &'a Self,
        delta: &'a impl Delta<'a, Node = Self::Node, Edge = Self::Edge>,
        d_match: &impl DMatch<'a, Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        init_global_logger_once("logs/hyper-simulation.log");
        let mut hs_trace = HyperSimulationTrace::new();
        let mut simulation: HashMap<&'a Self::Node, HashSet<&'a Self::Node>> = self
            .nodes()
            .map(|u| {
                let res = other
                    .nodes()
                    .filter(|v| {
                        if self.type_same(u, *v) {
                            let sematic_clusters = delta.get_sematic_clusters(u, v);
                            for (cluster_u, cluster_v) in sematic_clusters {
                                let d_match_set = d_match.d_match(cluster_u, cluster_v);
                                if !d_match_set.contains(&(u.id(), v.id())) {
                                    // Add the trace that nodes (u, v) are deleted by the `sematic_clusters`
                                    hs_trace.add_base_event(cluster_u.id, d_match_set.clone());
                                    return false;
                                }
                            }
                            return true;
                        }
                        false
                    })
                    .collect();
                (u, res)
            })
            .collect();

        info!("END Initial, raw-sim: is ");
        for (u, v_set) in &simulation {
            info!(
                "\tsim({}) = {:?}",
                u.id(),
                v_set.iter().map(|v| v.id()).collect::<Vec<_>>()
            );
        }

        let mut simulation_by_id: HashSet<(usize, usize)> = simulation
            .iter()
            .flat_map(|(u, v_set)| v_set.iter().map(move |v| (u.id(), v.id())))
            .collect();

        let mut changed = true;
        while changed {
            changed = false;
            for u in self.nodes() {
                let mut need_delete = Vec::new();
                for v in simulation.get(u).unwrap() {
                    info!("Checking {} -> {}", u.id(), v.id());
                    let mut _delete = false;

                    let sematic_clusters = delta.get_sematic_clusters(u, v);
                    for (cluster_u, cluster_v) in sematic_clusters {
                        let d_relation = d_match.d_match(cluster_u, cluster_v);
                        // Check if for all (u_id, v_id) in d_relation, (u_id, v_id) is in simulation, i.e., d_relation is a subset of simulation_by_id
                        if !d_relation.is_subset(&simulation_by_id) {
                            info!("Deleting {} -> {}", u.id(), v.id());
                            // Add the trace that nodes (u, v) are deleted by the `sematic_clusters`
                            let uncoverd: HashSet<(usize, usize)> =
                                d_relation.difference(&simulation_by_id).copied().collect();
                            hs_trace.add_derivation_event(cluster_u.id, uncoverd);
                            _delete = true;
                            break;
                        }
                    }

                    if _delete {
                        need_delete.push(*v);
                    }
                }

                for v in need_delete {
                    simulation.get_mut(u).unwrap().remove(v);
                    simulation_by_id.remove(&(u.id(), v.id()));
                    changed = true;
                }
            }
        }

        hs_trace
            .store_trace_file("logs/hyper_simulation.trace")
            .unwrap();

        return simulation;
    }

    fn get_hyper_simulation_strict(
        &'a self,
        other: &'a Self,
        delta: &'a impl Delta<'a, Node = Self::Node, Edge = Self::Edge>,
        d_match: &impl DMatch<'a, Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        init_global_logger_once("logs/hyper-simulation.log");
        let mut hs_trace = HyperSimulationTrace::new();
        let mut simulation: HashMap<&'a Self::Node, HashSet<&'a Self::Node>> = self
            .nodes()
            .map(|u| {
                let res = other
                    .nodes()
                    .filter(|v| {
                        if self.type_same(u, *v) {
                            let sematic_clusters = delta.get_sematic_clusters(u, v);
                            // Highlight!
                            if sematic_clusters.len() == 0 {
                                info!(
                                    "Deleting {} -> {} because no sematic cluster",
                                    u.id(),
                                    v.id()
                                );
                                return false;
                            }
                            info!(
                                "Checking {} -> {}, sematic clusters size: {}",
                                u.id(),
                                v.id(),
                                sematic_clusters.len()
                            );
                            for (cluster_u, cluster_v) in sematic_clusters {
                                let d_match_set = d_match.d_match(cluster_u, cluster_v);
                                if !d_match_set.contains(&(u.id(), v.id())) {
                                    // Add the trace that nodes (u, v) are deleted by the `sematic_clusters`
                                    hs_trace.add_base_event(cluster_u.id, d_match_set.clone());
                                    return false;
                                }
                            }
                            return true;
                        }
                        false
                    })
                    .collect();
                (u, res)
            })
            .collect();

        info!("END Initial, raw-sim: is ");
        for (u, v_set) in &simulation {
            info!(
                "\tsim({}) = {:?}",
                u.id(),
                v_set.iter().map(|v| v.id()).collect::<Vec<_>>()
            );
        }

        let mut simulation_by_id: HashSet<(usize, usize)> = simulation
            .iter()
            .flat_map(|(u, v_set)| v_set.iter().map(move |v| (u.id(), v.id())))
            .collect();

        let mut changed = true;
        while changed {
            changed = false;
            for u in self.nodes() {
                let mut need_delete = Vec::new();
                for v in simulation.get(u).unwrap() {
                    info!("Checking {} -> {}", u.id(), v.id());
                    let mut _delete = false;

                    let sematic_clusters = delta.get_sematic_clusters(u, v);
                    for (cluster_u, cluster_v) in sematic_clusters {
                        let d_relation = d_match.d_match(cluster_u, cluster_v);
                        // Check if for all (u_id, v_id) in d_relation, (u_id, v_id) is in simulation, i.e., d_relation is a subset of simulation_by_id
                        if !d_relation.is_subset(&simulation_by_id) {
                            info!("Deleting {} -> {}", u.id(), v.id());
                            // Add the trace that nodes (u, v) are deleted by the `sematic_clusters`
                            let uncoverd: HashSet<(usize, usize)> =
                                d_relation.difference(&simulation_by_id).copied().collect();
                            hs_trace.add_derivation_event(cluster_u.id, uncoverd);
                            _delete = true;
                            break;
                        }
                    }

                    if _delete {
                        need_delete.push(*v);
                    }
                }

                for v in need_delete {
                    simulation.get_mut(u).unwrap().remove(v);
                    simulation_by_id.remove(&(u.id(), v.id()));
                    changed = true;
                }
            }
        }

        hs_trace
            .store_trace_file("logs/hyper_simulation.trace")
            .unwrap();

        return simulation;
    }

    fn get_hyper_simulation_effect(
        &'a self,
        other: &'a Self,
        delta: &'a impl Delta<'a, Node = Self::Node, Edge = Self::Edge>,
        d_match: &impl DMatch<'a, Edge = Self::Edge>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        init_global_logger_once("logs/hyper-simulation.log");

        // Keep id-to-node maps so the id-based worklist can be converted back
        // to the graph's borrowed-node representation at the end.
        let mut id_to_u: HashMap<usize, &'a Self::Node> = HashMap::new();
        let mut id_to_v: HashMap<usize, &'a Self::Node> = HashMap::new();

        // HC dependencies indexed by their Delta anchor pair.  One logical HC
        // may be associated with several anchors by the Python binding.
        let mut hc_map: HashMap<(usize, usize), Vec<((usize, usize), HashSet<(usize, usize)>)>> =
            HashMap::new();

        // Pi is the current candidate Hyper Simulation relation.
        let mut pi: HashSet<(usize, usize)> = HashSet::new();

        // ==========================================
        // Phase 1: Declarative Initialization
        // ==========================================

        // Initialize Pi from h_v/type compatibility and freeze all HC/D-match
        // dependencies before any deletion is performed.
        for u in self.nodes() {
            id_to_u.insert(u.id(), u);
            for v in other.nodes() {
                id_to_v.insert(v.id(), v);

                if self.type_same(u, v) {
                    let sematic_clusters = delta.get_sematic_clusters(u, v);
                    let mut valid = true;
                    let mut local_clusters = Vec::new();

                    for (cluster_u, cluster_v) in sematic_clusters {
                        let cu_id = cluster_u.id;
                        let cv_id = cluster_v.id;
                        let d_match_set = d_match.d_match(cluster_u, cluster_v);

                        // Anchor-membership invariant: every HC registered for
                        // this pair must explicitly contain it in D-match.
                        if !d_match_set.contains(&(u.id(), v.id())) {
                            valid = false;
                            break; // All registered HCs are conjunctive dependencies.
                        }
                        local_clusters.push(((cu_id, cv_id), d_match_set.clone()));
                    }

                    if valid {
                        pi.insert((u.id(), v.id()));
                        hc_map.insert((u.id(), v.id()), local_clusters);
                    }
                }
            }
        }

        info!(
            "Initialized Pi and frozen HC/D-match; Pi size: {}",
            pi.len()
        );

        // Cache one D-match per logical HC pair.
        let mut a_cluster_d_match: HashMap<(usize, usize), HashSet<(usize, usize)>> =
            HashMap::new();

        // Reverse dependency indices:
        // D_cluster[(Cu, Cv)] -> Delta anchors that require this HC.
        let mut d_cluster: HashMap<(usize, usize), HashSet<(usize, usize)>> = HashMap::new();
        // D_pair[(u', v')] -> HCs whose D-match requires this node pair.
        let mut d_pair: HashMap<(usize, usize), HashSet<(usize, usize)>> = HashMap::new();

        for (&(u_id, v_id), clusters) in &hc_map {
            for ((cu_id, cv_id), d_match_set) in clusters {
                let c_pair = (*cu_id, *cv_id);

                // Record every anchor associated with the logical HC.
                d_cluster.entry(c_pair).or_default().insert((u_id, v_id));

                // The first encounter freezes the HC's D-match and reverse
                // pair dependencies. Later anchors reuse the same HC id.
                if !a_cluster_d_match.contains_key(&c_pair) {
                    a_cluster_d_match.insert(c_pair, d_match_set.clone());

                    for &(up_id, vp_id) in d_match_set {
                        d_pair.entry((up_id, vp_id)).or_default().insert(c_pair);
                    }
                }
            }
        }

        info!("Built HC and D-match dependency indices");

        // V_C contains HCs whose complete D-match is present in the initial Pi.
        let mut v_c: HashSet<(usize, usize)> = HashSet::new();
        for (c_pair, d_match_set) in &a_cluster_d_match {
            // D-match closure invariant: every required pair must survive in Pi.
            if d_match_set.is_subset(&pi) {
                v_c.insert(*c_pair);
            }
        }

        info!("Initialized valid HC set V_C");

        // Seed the worklist with anchors that depend on an invalid HC.
        let mut q: VecDeque<(usize, usize)> = VecDeque::new();
        let mut pi_retained = pi.clone();

        for &(u_id, v_id) in &pi {
            let mut all_in_vc = true;
            if let Some(clusters) = hc_map.get(&(u_id, v_id)) {
                for (c_pair, _) in clusters {
                    if !v_c.contains(c_pair) {
                        all_in_vc = false;
                        break;
                    }
                }
            }

            if !all_in_vc {
                q.push_back((u_id, v_id)); // enqueue once
                pi_retained.remove(&(u_id, v_id)); // Pi = Pi \ Q
            }
        }
        pi = pi_retained;

        info!("Seeded invalid anchor worklist");

        // ==========================================
        // Phase 2: Cascade deletions via the queue
        // ==========================================
        while let Some((up_id, vp_id)) = q.pop_front() {
            // A removed pair invalidates every HC whose D-match needs it.
            if let Some(dependent_clusters) = d_pair.get(&(up_id, vp_id)) {
                for c_pair in dependent_clusters {
                    // Each HC transitions from valid to invalid at most once.
                    if v_c.contains(c_pair) {
                        v_c.remove(c_pair);

                        // Cascade to all Delta anchors associated with the HC.
                        if let Some(dependent_node_pairs) = d_cluster.get(c_pair) {
                            for node_pair in dependent_node_pairs {
                                if pi.contains(node_pair) {
                                    pi.remove(node_pair);
                                    q.push_back(*node_pair);
                                }
                            }
                        }
                    }
                }
            }
        }

        info!("Completed Hyper Simulation worklist");

        // ==========================================
        // Phase 3: Construct Output
        // ==========================================
        // Convert the final id relation back to borrowed graph nodes.
        let mut result: HashMap<&'a Self::Node, HashSet<&'a Self::Node>> =
            self.nodes().map(|u| (u, HashSet::new())).collect();

        for (u_id, v_id) in pi {
            // Every surviving id was inserted while scanning the two graphs.
            let u_node = id_to_u[&u_id];
            let v_node = id_to_v[&v_id];
            if let Some(set) = result.get_mut(u_node) {
                set.insert(v_node);
            }
        }

        result
    }

    fn get_hyper_simulation_effect_pass_by(
        &'a self,
        _other: &'a Self,
        delta: &'a impl Delta<'a, Node = Self::Node, Edge = Self::Edge>,
        d_match: &impl DMatch<'a, Edge = Self::Edge>,
        type_same_lookup: &HashMap<&'a Self::Node, HashSet<&'a Self::Node>>,
    ) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        init_global_logger_once("logs/hyper-simulation.log");

        // This variant receives the h_v-compatible pairs precomputed by the
        // caller, but uses the same id-based dependency worklist.
        let mut id_to_u: HashMap<usize, &'a Self::Node> = HashMap::new();
        let mut id_to_v: HashMap<usize, &'a Self::Node> = HashMap::new();

        // Delta anchors and their frozen HC/D-match dependencies.
        let mut hc_map: HashMap<(usize, usize), Vec<((usize, usize), HashSet<(usize, usize)>)>> =
            HashMap::new();

        // Pi is initialized directly from the supplied h_v lookup.
        let mut pi: HashSet<(usize, usize)> = HashSet::new();

        // ==========================================
        // Phase 1: Declarative Initialization
        // ==========================================

        // Freeze HC and D-match before starting the monotone deletion phase.
        for u in self.nodes() {
            id_to_u.insert(u.id(), u);

            if let Some(type_same_vs) = type_same_lookup.get(u) {
                for v in type_same_vs {
                    id_to_v.insert(v.id(), v);

                    let sematic_clusters = delta.get_sematic_clusters(u, v);
                    let mut valid = true;
                    let mut local_clusters = Vec::new();

                    for (cluster_u, cluster_v) in sematic_clusters {
                        let cu_id = cluster_u.id;
                        let cv_id = cluster_v.id;
                        let d_match_set = d_match.d_match(cluster_u, cluster_v);

                        // Anchor-membership invariant: the anchor must occur in D-match.
                        if !d_match_set.contains(&(u.id(), v.id())) {
                            valid = false;
                            break; // All registered HCs are conjunctive dependencies.
                        }
                        local_clusters.push(((cu_id, cv_id), d_match_set.clone()));
                    }

                    if valid {
                        pi.insert((u.id(), v.id()));
                        hc_map.insert((u.id(), v.id()), local_clusters);
                    }
                }
            }
        }

        info!(
            "Initialized Pi from the supplied h_v lookup; Pi size: {}",
            pi.len()
        );

        // Cache one D-match per logical HC pair.
        let mut a_cluster_d_match: HashMap<(usize, usize), HashSet<(usize, usize)>> =
            HashMap::new();

        // Build HC-to-anchor and pair-to-HC reverse dependency indices.
        // D_cluster[(Cu, Cv)] -> { (u, v) \in Pi }
        let mut d_cluster: HashMap<(usize, usize), HashSet<(usize, usize)>> = HashMap::new();
        // D_pair[(u', v')] -> { (Cu, Cv) \in A_cluster }
        let mut d_pair: HashMap<(usize, usize), HashSet<(usize, usize)>> = HashMap::new();

        for (&(u_id, v_id), clusters) in &hc_map {
            for ((cu_id, cv_id), d_match_set) in clusters {
                let c_pair = (*cu_id, *cv_id);

                // Record the Delta anchor for this HC.
                d_cluster.entry(c_pair).or_default().insert((u_id, v_id));

                // Record D-match dependencies only once per logical HC.
                if !a_cluster_d_match.contains_key(&c_pair) {
                    a_cluster_d_match.insert(c_pair, d_match_set.clone());

                    for &(up_id, vp_id) in d_match_set {
                        d_pair.entry((up_id, vp_id)).or_default().insert(c_pair);
                    }
                }
            }
        }

        info!("Built HC and D-match dependency indices");

        // V_C contains HCs whose complete D-match is present in Pi.
        let mut v_c: HashSet<(usize, usize)> = HashSet::new();
        for (c_pair, d_match_set) in &a_cluster_d_match {
            // D-match closure invariant: every required pair must survive in Pi.
            if d_match_set.is_subset(&pi) {
                v_c.insert(*c_pair);
            }
        }

        info!("Initialized valid HC set V_C");

        // Seed the deletion worklist with invalid anchors.
        let mut q: VecDeque<(usize, usize)> = VecDeque::new();
        let mut pi_retained = pi.clone();

        for &(u_id, v_id) in &pi {
            let mut all_in_vc = true;
            if let Some(clusters) = hc_map.get(&(u_id, v_id)) {
                for (c_pair, _) in clusters {
                    if !v_c.contains(c_pair) {
                        all_in_vc = false;
                        break;
                    }
                }
            }

            if !all_in_vc {
                q.push_back((u_id, v_id));
                pi_retained.remove(&(u_id, v_id)); // Pi = Pi \ Q
            }
        }
        pi = pi_retained;

        info!("Seeded invalid anchor worklist");

        // ==========================================
        // Phase 2: Cascade deletions via the queue
        // ==========================================
        while let Some((up_id, vp_id)) = q.pop_front() {
            // A removed pair invalidates every HC whose D-match requires it.
            if let Some(dependent_clusters) = d_pair.get(&(up_id, vp_id)) {
                for c_pair in dependent_clusters {
                    // Each HC transitions from valid to invalid at most once.
                    if v_c.contains(c_pair) {
                        v_c.remove(c_pair); // V_c = V_c \ {(Cu, Cv)}

                        // Cascade to all anchors associated with this HC.
                        if let Some(dependent_node_pairs) = d_cluster.get(c_pair) {
                            for node_pair in dependent_node_pairs {
                                if pi.contains(node_pair) {
                                    pi.remove(node_pair);
                                    q.push_back(*node_pair);
                                }
                            }
                        }
                    }
                }
            }
        }

        info!("Completed Hyper Simulation worklist");

        // ==========================================
        // Phase 3: Construct Output
        // ==========================================
        // Convert the final id relation back to graph nodes.
        let mut result: HashMap<&'a Self::Node, HashSet<&'a Self::Node>> =
            self.nodes().map(|u| (u, HashSet::new())).collect();

        for (u_id, v_id) in pi {
            // Every surviving id came from the precomputed lookup.
            let u_node = id_to_u[&u_id];
            let v_node = id_to_v[&v_id];
            if let Some(set) = result.get_mut(u_node) {
                set.insert(v_node);
            }
        }

        result
    }

    fn get_hyper_simulation_effect_by_id(
        &'a self,
        hc_map: &HashMap<(usize, usize), Vec<((usize, usize), HashSet<(usize, usize)>)>>,
    ) -> HashSet<(usize, usize)> {
        // ==========================================
        // Precomputed id-only input
        // ==========================================
        // hc_map[(u,v)] lists every logical HC pair associated with the
        // candidate anchor and the HC's D-match relation.  The caller has
        // already applied h_v, constructed Delta, and frozen D-match.  This
        // function therefore performs only the deterministic fixed point in
        // three phases: initialize Pi, initialize V_C, and cascade deletions.

        init_global_logger_once("logs/hyper-simulation.log");

        // Pi is the current candidate relation.
        let mut pi: HashSet<(usize, usize)> = HashSet::new();

        // ==========================================
        // Phase 1: Initialize Pi from hc_map
        // ==========================================
        for &(u_id, v_id) in hc_map.keys() {
            pi.insert((u_id, v_id));
        }

        info!("Initialized id-only Pi; Pi size: {}", pi.len());

        // Cache one D-match per logical HC pair.
        let mut a_cluster_d_match: HashMap<(usize, usize), HashSet<(usize, usize)>> =
            HashMap::new();

        // D_cluster maps an HC to its Delta anchors.
        let mut d_cluster: HashMap<(usize, usize), HashSet<(usize, usize)>> = HashMap::new();
        // D_pair maps a required D-match pair to dependent HCs.
        let mut d_pair: HashMap<(usize, usize), HashSet<(usize, usize)>> = HashMap::new();

        for (&(u_id, v_id), clusters) in hc_map.iter() {
            for ((cu_id, cv_id), d_match_set) in clusters {
                let c_pair = (*cu_id, *cv_id);

                // Record the anchor for this logical HC.
                d_cluster.entry(c_pair).or_default().insert((u_id, v_id));

                // Freeze reverse D-match dependencies once per HC.
                if !a_cluster_d_match.contains_key(&c_pair) {
                    a_cluster_d_match.insert(c_pair, d_match_set.clone());

                    for &(up_id, vp_id) in d_match_set {
                        d_pair.entry((up_id, vp_id)).or_default().insert(c_pair);
                    }
                }
            }
        }

        info!("Built id-only HC dependency indices");

        // Initialize HCs whose complete D-match is present in Pi.
        let mut v_c: HashSet<(usize, usize)> = HashSet::new();
        for (c_pair, d_match_set) in &a_cluster_d_match {
            // D(Cq,Cd) must be a subset of Pi.
            if d_match_set.is_subset(&pi) {
                v_c.insert(*c_pair);
            }
        }

        info!("Initialized valid HC set V_C");

        // Seed anchors whose HCs are already invalid.
        let mut q: VecDeque<(usize, usize)> = VecDeque::new();
        let mut pi_retained = pi.clone();

        for &(u_id, v_id) in &pi {
            let mut all_in_vc = true;
            if let Some(clusters) = hc_map.get(&(u_id, v_id)) {
                for (c_pair, _) in clusters {
                    if !v_c.contains(c_pair) {
                        all_in_vc = false;
                        break;
                    }
                }
            }

            if !all_in_vc {
                q.push_back((u_id, v_id));
                pi_retained.remove(&(u_id, v_id)); // Pi = Pi \ Q
            }
        }
        pi = pi_retained;

        info!("Seeded id-only deletion worklist");

        // ==========================================
        // Phase 2: Cascade deletions via the queue
        // ==========================================
        while let Some((up_id, vp_id)) = q.pop_front() {
            // Invalidate HCs depending on the removed pair.
            if let Some(dependent_clusters) = d_pair.get(&(up_id, vp_id)) {
                for c_pair in dependent_clusters {
                    // Each HC becomes invalid at most once.
                    if v_c.contains(c_pair) {
                        v_c.remove(c_pair); // V_c = V_c \ {(Cu, Cv)}

                        // Cascade to each anchor that requires the HC.
                        if let Some(dependent_node_pairs) = d_cluster.get(c_pair) {
                            for node_pair in dependent_node_pairs {
                                if pi.contains(node_pair) {
                                    pi.remove(node_pair);
                                    q.push_back(*node_pair);
                                }
                            }
                        }
                    }
                }
            }
        }

        info!("Completed id-only Hyper Simulation worklist");

        // ==========================================
        // Phase 3: Return ID-based Result
        // ==========================================
        pi
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
/// Serializable event journal emitted by the trace-oriented reference solver.
pub struct HyperSimulationTrace {
    events: Vec<HSEvent>,
}

impl HyperSimulationTrace {
    fn new() -> Self {
        HyperSimulationTrace { events: Vec::new() }
    }

    fn add_base_event(&mut self, sc_id: usize, d_match: HashSet<(usize, usize)>) {
        let event = HSEvent::Base(sc_id, d_match);
        self.events.push(event);
    }
    fn add_derivation_event(&mut self, sc_id: usize, uncoverd: HashSet<(usize, usize)>) {
        let event = HSEvent::Derivation(sc_id, uncoverd);
        self.events.push(event);
    }
}

impl IntoIterator for HyperSimulationTrace {
    type Item = HSEvent;
    type IntoIter = std::vec::IntoIter<HSEvent>;

    fn into_iter(self) -> Self::IntoIter {
        self.events.into_iter()
    }
}

impl<'a> IntoIterator for &'a HyperSimulationTrace {
    type Item = &'a HSEvent;
    type IntoIter = std::slice::Iter<'a, HSEvent>;

    fn into_iter(self) -> Self::IntoIter {
        self.events.iter()
    }
}

impl TraceLog for HyperSimulationTrace {
    fn store_trace_file(self, filename: &'static str) -> Result<(), Box<dyn Error>> {
        // use bincode to save the HyperSimulationTrace.
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &self)?;
        Ok(())
    }

    fn get_trace(filename: &'static str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);
        let file_decoded: HyperSimulationTrace = bincode::deserialize_from(&mut reader)?;
        Ok(file_decoded)
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
/// An anchor-membership failure or a derived D-match closure failure.
pub enum HSEvent {
    Base(usize, HashSet<(usize, usize)>),       // current D-Match
    Derivation(usize, HashSet<(usize, usize)>), // D-Match \ Sim
}
