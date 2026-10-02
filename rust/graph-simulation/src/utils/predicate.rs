
//! Cached synthetic label predicates used by examples and benchmarks.
//!
//! These helpers are not part of the HyperMatch semantic estimators; they
//! provide repeatable match relations for exercising the generic algorithms.

use std::{collections::{HashMap, HashSet}, hash::Hash};
use rand_pcg::Pcg64;
use lazy_static::lazy_static;
use crate::utils::validation::Node;
use serde::{Serialize, Deserialize};
use fxhash::FxHashMap;
use rand::{prelude::*, rng};
use std::sync::RwLock;
lazy_static!{
    static ref l_save: RwLock<LSave> = RwLock::new(LSave::from_file());
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
struct Hyperedge((HashSet<Node>, HashSet<Node>));

impl Hash for Hyperedge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for node in &self.0 .0 {
            node.hash(state);
        }
        for node in &self.0 .1 {
            node.hash(state);
        }
    }
}

#[derive(Serialize, Deserialize)]
struct LSave {
    l_predicate_node: FxHashMap<(Node, Node), bool>,
    l_predicate_node_set: FxHashMap<Hyperedge, bool>,
    l_match: FxHashMap<Hyperedge, HashMap<Node, Node>>,
}

impl LSave {
    pub fn from_file() -> Self {
        let file = std::fs::File::open("lsave_backup.json").expect("Unable to open file");
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader).expect("Unable to parse JSON")
    }

    fn l_predicate_node(&mut self, x: &Node, y: &Node, p: f64) -> bool {
        if let Some(&result) = self.l_predicate_node.get(&(x.clone(), y.clone())) {
            return result;
        }
        let mut rng = rng();
        let result = rng.random_bool(p);
        self.l_predicate_node.insert((x.clone(), y.clone()), result);
        result
    }

    fn l_predicate_node_set(&mut self, x: &HashSet<Node>, y: &HashSet<Node>, p: f64) -> bool {
        let hyperedge = Hyperedge((x.clone(), y.clone()));
        if let Some(&result) = self.l_predicate_node_set.get(&hyperedge) {
            return result;
        }
        let mut rng = rng();
        let result = rng.random_bool(p);
        self.l_predicate_node_set.insert(hyperedge, result);
        result
    }

    fn l_match(&mut self, x: &HashSet<Node>, y: &HashSet<Node>, p: f64) -> HashMap<Node, Node> {
        let hyperedge = Hyperedge((x.clone(), y.clone()));
        if let Some(result) = self.l_match.get(&hyperedge) {
            return result.clone();
        }
        let mut rng = rng();
        let mut used = HashSet::new();
        let mut result = HashMap::new();
        for node_x in x {
            let mut min = 1.0;
            let mut min_node = None;
            for node_y in y {
                if !used.contains(node_y) {
                    let value = node_x.clone() ^ node_y.clone();
                    if value < min {
                        min = value;
                        min_node = Some(node_y);
                    }
                }
                if let Some(node_y) = min_node {
                    used.insert(node_y.clone());
                    result.insert(node_x.clone(), node_y.clone());
                }
            }
        }
        let mut final_result = HashMap::new();
        for (key, value) in result {
            if rng.random_bool(p) {
                final_result.insert(key, value);
            }
        }
        self.l_match.insert(hyperedge, final_result.clone());
        final_result
    }
}

impl Drop for LSave {
    fn drop(&mut self) {
        let file = std::fs::File::create("lsave_backup.json").expect("Unable to create file");
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, self).expect("Unable to write JSON");
    }
}

pub fn l_predicate_node(x: &Node, y: &Node, p: f64) -> bool {
    // Delegate to the cached set predicate so node and hyperedge tests share
    // exactly the same stochastic label semantics.
    l_save.write().unwrap().l_predicate_node(x, y, p)
}

pub fn l_predicate_node_set(x: &HashSet<Node>, y: &HashSet<Node>, p: f64) -> bool {
    l_save.write().unwrap().l_predicate_node_set(x, y, p)
}

pub fn l_match(x: &HashSet<Node>, y: &HashSet<Node>, p: f64) -> HashMap<Node, Node> {
    l_save.write().unwrap().l_match(x, y, p)
}
