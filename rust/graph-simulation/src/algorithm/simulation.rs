//! Classical graph simulation and related labelled variants.
//!
//! Each implementation computes a greatest relation by removing candidate
//! pairs that cannot match the required predecessor or successor structure.

use graph_base::interfaces::graph::{Adjacency, AdjacencyInv, Graph, Directed};
use graph_base::interfaces::labeled::{Labeled, LabeledAdjacency};

use std::cell::RefCell;
use std::collections::{HashSet, HashMap};
pub trait Simulation<'a> {
    /// Node type whose references are stored in the returned relation.
    type Node: 'a;

    fn get_simulation(&'a self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;

    fn get_simulation_inter(&'a self, other: &'a Self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;

    fn get_simulation_native(&'a self, other: &'a Self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;

    fn get_simulation_of_node_edge(&'a self, other: &'a Self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;

    fn get_simulation_of_edge(&'a self, other: &'a Self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;
    
    fn has_simulation(sim: HashMap<&'a Self::Node, HashSet<&'a Self::Node>>) -> bool;
}

impl<'a, 'b, T> Simulation<'a> for T
where 
    T: Graph<'a> + Adjacency<'a> + AdjacencyInv<'a> + Labeled<'a> + Directed + LabeledAdjacency<'a>,
    T::Node: 'a, T::Edge: 'a,
    'b: 'a
{
    type Node = T::Node;

    fn get_simulation(&'a self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        let mut simulation: HashMap<&'a <T as Graph<'_>>::Node, HashSet<&'a <T as Graph<'_>>::Node>> = HashMap::new();
        let remove = RefCell::new(HashMap::new());
        let (adj, adj_inv) = (self.get_adj(), self.get_adj_inv());

        let pre_V = self.nodes().map(|v| self.get_post(&adj, v).collect::<HashSet<_>>()).reduce(|acc, x| acc.union(&x).cloned().collect()).unwrap();

        for v in self.nodes() {
            let sim_v: HashSet<_> = if self.get_post(&adj, v).count() != 0 {
                self.nodes().filter(|u| self.label_same(v, u)).collect()
            } else {
                self.nodes().filter(|u| self.label_same(v, u) && self.get_post(&adj,u).count() != 0).collect()
            };
            simulation.insert(v, sim_v.clone());

            let pre_sim_v = sim_v.into_iter().map(|u| self.get_pre(&adj_inv, u).collect::<HashSet<_>>()).reduce(|acc, x| acc.union(&x).cloned().collect()).unwrap();
            let res: HashSet<_> = pre_V.clone().into_iter().filter(|u| !pre_sim_v.contains(u)).collect();
            remove.borrow_mut().insert(v, res);
        }

        let legal_v = || {
            for v in self.nodes() {
                if remove.borrow().get(v).unwrap().len() != 0 {
                    return Some(v);
                }
            }
            None
        };

        while let Some(v) = legal_v() {
            for u in self.get_pre(&adj_inv,v) {
                for w in remove.borrow().get(v).unwrap() {
                    if simulation.get(u).unwrap().contains(w) {
                        simulation.get_mut(u).unwrap().remove(w);
                        for w_prime in self.get_pre(&adj_inv, w) {
                            if self.get_post(&adj, w_prime).collect::<HashSet<_>>().intersection(simulation.get(u).unwrap()).count() == 0 {
                                remove.borrow_mut().get_mut(u).unwrap().insert(w_prime);
                            }
                        }
                    }

                }
            }
            remove.borrow_mut().get_mut(v).unwrap().clear();
        }

        simulation
    }

    fn get_simulation_inter(&'a self, other: &'a Self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        let mut simulation: HashMap<&'a <T as Graph<'_>>::Node, HashSet<&'a <T as Graph<'_>>::Node>> = HashMap::new();
        let remove = RefCell::new(HashMap::new());
        let (adj, adj_inv) = (self.get_adj(), self.get_adj_inv());
        let (adj_other, adj_inv_other) = (other.get_adj(), other.get_adj_inv());

        let pre_V = other.nodes().map(|v| other.get_pre(&adj_inv_other, v).collect::<HashSet<_>>()).reduce(|acc, x| acc.union(&x).cloned().collect()).unwrap();
        
        for v in self.nodes() {
            let sim_v: HashSet<_> = if self.get_post(&adj, v).count() == 0 {
                other.nodes().filter(|u| self.label_same(v, u)).collect()
            } else {
                other.nodes().filter(|u| self.label_same(v, u) && other.get_post(&adj_other,u).count() != 0).collect()
            };
            simulation.insert(v, sim_v.clone());
            
            let pre_sim_v = sim_v.clone().iter().map(|u| other.get_pre(&adj_inv_other, u).collect::<HashSet<_>>()).reduce(|acc, x| acc.union(&x).cloned().collect()).unwrap_or(HashSet::new());
            let res: HashSet<_> = pre_V.difference(&pre_sim_v).copied().collect();
            remove.borrow_mut().insert(v, res);
        }   

        let legal_v = || {
            for v in self.nodes() {
                if remove.borrow().get(v).unwrap().len() != 0 {
                    return Some(v);
                }
            }
            None
        };

        while let Some(v) = legal_v() {
            for u in self.get_pre(&adj_inv,v) {
                let mut remove_u_add = HashSet::new();
                for w in remove.borrow().get(v).unwrap() {
                    if simulation.get(u).unwrap().contains(w) {
                        simulation.get_mut(u).unwrap().remove(w);
                        for w_prime in other.get_pre(&adj_inv_other, w) {
                            if other.get_post(&adj_other, w_prime).collect::<HashSet<_>>().intersection(simulation.get(u).unwrap()).count() == 0 {
                                remove_u_add.insert(w_prime);
                            }
                        }
                    }
                }
                remove.borrow_mut().get_mut(u).unwrap().extend(remove_u_add);
            }
            remove.borrow_mut().get_mut(v).unwrap().clear();
        }
        simulation
    }

    fn get_simulation_native(&'a self, other: &'a Self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        let mut simulation: HashMap<&'a <T as Graph<'_>>::Node, HashSet<&'a <T as Graph<'_>>::Node>> = HashMap::new();
        let (adj_other, _) = (other.get_adj(), other.get_adj_inv());
        
        for v in self.nodes() {
            let sim_v: HashSet<_> = other.nodes().filter(|u| self.label_same(v, u)).collect();
            simulation.insert(v, sim_v.clone());
        }

        let mut changed = true;
        while changed {
            changed = false;
            for (u, u_prime) in self.get_edges_pair() {
                let mut sim_u_remove = HashSet::new();
                for v in simulation.get(u).unwrap() {
                    let mut v_need_remove = true;
                    for v_prime in other.get_post(&adj_other, v) {
                        if simulation.get(u_prime).unwrap().contains(v_prime) {
                            v_need_remove = false;
                            break;
                        }
                    }
                    if v_need_remove {
                        sim_u_remove.insert(v.clone());
                        changed = true;
                    }
                }
                for v in sim_u_remove {
                    simulation.get_mut(u).unwrap().remove(v);
                }
            }
        }

        
        simulation
    }

    fn get_simulation_of_node_edge(&'a self, other: &'a Self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        let mut simulation: HashMap<&'a <T as Graph<'_>>::Node, HashSet<&'a <T as Graph<'_>>::Node>> = HashMap::new();
        let (adj_other, _) = (other.get_labeled_adj(), other.get_adj_inv());
        
        for v in self.nodes() {
            let sim_v: HashSet<_> = other.nodes().filter(|u| self.label_same(v, u)).collect();
            simulation.insert(v, sim_v.clone());
        }

        let mut changed = true;
        while changed {
            changed = false;
            for (u,  u_edge, u_prime) in self.get_edges_pair_with_edge() {
                let mut sim_u_remove = HashSet::new();
                for v in simulation.get(u).unwrap() {
                    let mut v_need_remove = true;
                    for (v_prime, v_edge) in other.get_labeled_post(&adj_other, v) {
                        if self.edge_label_same(u_edge, v_edge) && simulation.get(u_prime).unwrap().contains(v_prime) {
                            v_need_remove = false;
                            break;
                        }
                    }
                    if v_need_remove {
                        sim_u_remove.insert(v.clone());
                        changed = true;
                    }
                }
                for v in sim_u_remove {
                    simulation.get_mut(u).unwrap().remove(v);
                }
            }
        }

        
        simulation
    }

    fn get_simulation_of_edge(&'a self, other: &'a Self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {
        let mut simulation: HashMap<&'a <T as Graph<'_>>::Node, HashSet<&'a <T as Graph<'_>>::Node>> = HashMap::new();
        let (adj_other, _) = (other.get_labeled_adj(), other.get_adj_inv());
        
        for v in self.nodes() {
            let sim_v: HashSet<_> = other.nodes().collect();
            simulation.insert(v, sim_v.clone());
        }

        let mut changed = true;
        while changed {
            changed = false;
            for (u,  u_edge, u_prime) in self.get_edges_pair_with_edge() {
                let mut sim_u_remove = HashSet::new();
                for v in simulation.get(u).unwrap() {
                    let mut v_need_remove = true;
                    for (v_prime, v_edge) in other.get_labeled_post(&adj_other, v) {
                        if self.edge_node_label_same(u, u_edge, u_prime, v, v_edge, v_prime) && simulation.get(u_prime).unwrap().contains(v_prime) {
                            v_need_remove = false;
                            break;
                        }
                    }
                    if v_need_remove {
                        sim_u_remove.insert(v.clone());
                        changed = true;
                    }
                }
                for v in sim_u_remove {
                    simulation.get_mut(u).unwrap().remove(v);
                }
            }
        }

        
        simulation
    }

    fn has_simulation(sim: HashMap<&'a Self::Node, HashSet<&'a Self::Node>>) -> bool {
        sim.iter().all(|(_, sim_v)| {
            sim_v.len() != 0
        })
    }
}

pub trait HyperSimulation<'a> {
    /// Node type used by this legacy graph-oriented helper trait.
    type Node: 'a;

    fn get_hyper_simulation_inter(&'a self, other: &'a Self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;
}
