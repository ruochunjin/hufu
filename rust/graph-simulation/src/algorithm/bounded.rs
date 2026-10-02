//! Bounded graph simulation with caller-defined pairwise distance limits.

use graph_base::interfaces::graph::{Adjacency, AdjacencyInv, Degree, Directed, Graph};
use graph_base::interfaces::labeled::Labeled;

use std::collections::{HashSet, HashMap};

pub trait BoundedSimulation<'a> {
    /// Node type participating in the bounded relation.
    type Node: 'a;
    fn get_bounded_simulation(&'a self, other: &'a Self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>>;
}

pub trait Bounded<'a>: Graph<'a> {
    /// Maximum admissible path distance for a candidate pair `(u, v)`.
    fn get_bound(&'a self, u: &'a Self::Node, v: &'a Self::Node) -> usize;
}

impl<'a, 'b, T> BoundedSimulation<'a> for T 
where
    T: Graph<'a> + Bounded<'a> + Degree<'a> + Labeled<'a> + Adjacency<'a> + Degree<'a> + AdjacencyInv<'a> + Directed + 'b,
    T::Node: 'a, T::Edge: 'a,
{
    type Node = T::Node;

    fn get_bounded_simulation(&'a self, other: &'a Self) -> HashMap<&'a Self::Node, HashSet<&'a Self::Node>> {

        let adj_self = self.get_adj();
        let adj_other = other.get_adj();
        let adj_self_inv = self.get_adj_inv();
        // let adj_other_inv = other.get_adj_inv();



        // anc(get_bound(u_prime, u), u_prime, v) that returns v_prime in anc if:
        // 1. label_same(u_prime, v_prime)
        // 2. len(v_prime/.../v) <= get_bound(u_prime, u)
        // dec(get_bound(u, u_prime), u_prime, v) that returns v_prime in dec if:
        // 1. label_same(u_prime, v_prime)
        // 2. len(v/.../v_prime) <= get_bound(u, u_prime)
        // they are HashMap<(usize, &Node, &Node), HashSet<&Node>>

        // We firstly compute the distance matrix M of (V_other, E_other)
        let mut distance: HashMap<(&T::Node, &T::Node), usize> = HashMap::new();
        for u in other.nodes() {
            let mut queue: Vec<(&T::Node, usize)> = vec![(u, 0)];
            let mut visited: HashSet<&T::Node> = HashSet::new();
            visited.insert(u);
            while !queue.is_empty() {
                let (current, dist) = queue.remove(0);
                distance.insert((u, current), dist);
                for neighbor in other.get_post(&adj_other, current) {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor);
                        queue.push((neighbor, dist + 1));
                    }
                }
            }
        }
        
        let mut anc: HashMap<(usize, &T::Node, &T::Node), HashSet<&T::Node>> = HashMap::new();
        let mut dec: HashMap<(usize, &T::Node, &T::Node), HashSet<&T::Node>> = HashMap::new();
        
        
        // Then we compute anc and dec based on distance matrix

        // compute anc
        // anc(bound, u_prime, v) := {v_prime | label_same(u_prime, v_prime) and distance(v_prime, v) <= bound}
        // where u_prime is node from self, v_prime and v are nodes from other
        for u_prime in self.nodes() {
            for u in self.get_post(&adj_self, u_prime) {
                let bound = self.get_bound(u_prime, u);
                for v in other.nodes() {
                    let mut anc_set: HashSet<&T::Node> = HashSet::new();
                    for v_prime in other.nodes() {
                        if self.label_same(u_prime, v_prime) {
                            if let Some(&dist) = distance.get(&(v_prime, v)) {
                                if dist <= bound {
                                    anc_set.insert(v_prime);
                                }
                            }
                        }
                    }
                    anc.insert((bound, u_prime, v), anc_set);
                }
            }
        }

        // compute dec
        // dec(bound, u_prime, v) := {v_prime | label_same(u_prime, v_prime) and distance(v, v_prime) <= bound}
        // where u_prime is node from self, v_prime and v are nodes from other
        // We need to compute dec for all possible (u', u) pairs to get all required bounds
        for u in self.nodes() {
            for u_prime in self.get_post(&adj_self, u) {
                let bound = self.get_bound(u, u_prime);
                for v in other.nodes() {
                    let mut dec_set: HashSet<&T::Node> = HashSet::new();
                    for v_prime in other.nodes() {
                        if self.label_same(u_prime, v_prime) {
                            if let Some(&dist) = distance.get(&(v, v_prime)) {  
                                if dist <= bound {
                                    dec_set.insert(v_prime);
                                }
                            }
                        }
                    }
                    dec.insert((bound, u_prime, v), dec_set);
                }
            }
        }

        let self_out_degree = self.get_out_degree();
        let other_out_degree = other.get_out_degree();

        // sim(u) := {v | v in V_other and label_same(u, v) and out_degree(v) != 0 if out_degree(u) != 0}
        let mut sim = HashMap::new();
        for u in self.nodes() {
            let mut candidates: HashSet<&'a T::Node> = HashSet::new();
            for v in other.nodes() {
                if self.label_same(&u, v) {
                    if self.out_degree(&self_out_degree,&u) != 0 {
                        if other.out_degree(&other_out_degree,&v) != 0 {
                            candidates.insert(v);
                        }
                    } else {
                        candidates.insert(v);
                    }
                }
            }
            sim.insert(u, candidates);
        }

        // presim(u) := {v | v in V_other and there NOT exists (u_prime, u) in E_self 
        // s.t. ((1) v_prime in sim(u), (2) label_same(u_prime, v), and (3) len(v/.../v_prime) <= get_bound(u_prime, u))}
        let mut presim = HashMap::new();
        for u in self.nodes() {
            let mut candidates: HashSet<&'a T::Node> = HashSet::new();
            'v_loop: for v in sim.get(&u).unwrap() {
                if other.out_degree(&other_out_degree, v) == 0 {
                    continue;
                }
                // `v` enters presim(u) when no witness v' exists for any
                // predecessor u' with (u', u) in the query graph.
                for u_prime in self.get_pre(&adj_self_inv, u) {
                    // Condition (2): only label-compatible predecessors can
                    // invalidate this candidate.
                    if !self.label_same(u_prime, v) {
                        continue; // This predecessor cannot exclude v.
                    }
                    
                    let bound = self.get_bound(&u_prime, &u);
                    // Search for a data witness v' satisfying all conditions:
                    // (1) v' in sim(u)
                    // (2) label_same(u_prime, v), already checked above;
                    // (3) len(v/.../v') <= bound, represented by `dec`.
                    if let Some(dec_set) = dec.get(&(bound, &u_prime, v)) {
                        // `dec_set` contains compatible nodes within the bound;
                        // intersect it with the current sim(u).
                        let has_match = dec_set.iter().any(|v_prime| {
                            sim.get(&u).unwrap().contains(v_prime)
                        });
                        // A witness prevents v from entering presim(u).
                        if has_match {
                            continue 'v_loop;
                        }
                    }
                    // A missing/empty dec set provides no witness; continue.
                }
                // No predecessor found a witness, so schedule v for removal.
                candidates.insert(v);
            }
            presim.insert(u, candidates);
        }
        
        // while (there exists a node u ∈ V_self with premv(u) != ∅) do 
        //     for (each (u′, u) ∈ E_self and each z ∈ premv(u) ∩ sim(u′)) do 
        //         sim(u′) := sim(u′) \ {z};  
        //         if (sim(u′) = ∅) then return ∅; 
        //             for each u′′ with (u′′, u′) ∈ E_self do 
        //                 for (z′ ∈ anc(get_bound(u′′, u′), u′′, z) ∧ z′ /∈ premv(u′)) do 
        //                     if (dec(get_bound(u′′, u′), u′, z′) ∩ sim(u′) = ∅) 
        //                         then premv(u′) := premv(u′) ∪ {z′}; 
        //     premv(u) := ∅;

        loop {
            // 1. Select a query node with pending removals.
            let Some(u) = self.nodes().find(|node| !presim.get(node).unwrap().is_empty()) else {
                break;
            };
            
            // 2. Clone pending removals before mutating the maps.
            let premv_u = presim.get(&u).unwrap().clone();
            
            // 3. Collect all predecessors u' with (u', u) in the query graph.
            let u_primes: Vec<_> = self.get_pre(&adj_self_inv, &u).collect();
            
            for u_prime in u_primes {
                // 4. Restrict removals to candidates still present in sim(u').
                let sim_u_prime = sim.get(&u_prime).unwrap();
                let to_remove: Vec<_> = premv_u.intersection(sim_u_prime).cloned().collect();
                
                for z in to_remove {
                    // 5. Mutate sim only after all borrowed intersections end.
                    sim.get_mut(&u_prime).unwrap().remove(&z);
                    
                    if sim.get(&u_prime).unwrap().is_empty() {
                        return HashMap::new();
                    }
                    
                    // 6. Propagate the change to predecessors u'' of u'.
                    let u_double_primes: Vec<_> = self.get_pre(&adj_self_inv, &u_prime).collect();
                    
                    // 7. Accumulate presim updates before mutating the map.
                    let mut updates: Vec<(&T::Node, &T::Node)> = Vec::new();
                    
                    // Snapshot presim(u') to avoid scheduling duplicates.
                    let presim_u_prime = presim.get(&u_prime).unwrap().clone();
                    
                    for u_double_prime in u_double_primes {
                        let bound = self.get_bound(&u_double_prime, &u_prime);
                        
                        if let Some(anc_set) = anc.get(&(bound, &u_double_prime, &z)) {
                            // Materialize the ancestor candidates for stable borrowing.
                            let anc_vec: Vec<_> = anc_set.iter().cloned().collect();
                            
                            // Keep z' in anc(...) that is not already in presim(u').
                            for z_prime in anc_vec.iter() {
                                if !presim_u_prime.contains(z_prime) {
                                    // Schedule z' only when dec(...) intersects no sim(u').
                                    if let Some(dec_set) = dec.get(&(bound, &u_prime, z_prime)) {
                                        let sim_u_prime_set = sim.get(&u_prime).unwrap();
                                        let has_intersection = dec_set.iter().any(|v| sim_u_prime_set.contains(v));
                                        
                                        if !has_intersection {
                                            updates.push((u_double_prime, *z_prime));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // 8. Apply accumulated updates in one mutation phase.
                    for (u_double_prime, z_prime) in updates {
                        presim.get_mut(&u_double_prime).unwrap().insert(z_prime);
                    }
                }
            }
            
            // 9. Mark this pending-removal set as processed.
            presim.get_mut(&u).unwrap().clear();
        }

        sim
    }
}
