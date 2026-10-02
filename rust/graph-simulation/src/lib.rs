//! Generic graph- and hypergraph-simulation algorithms used by HyperMatch.
//!
//! The crate contains no model code: callers supply graph structure, labels,
//! `h_v`, HC dependencies, and D-match relations through the public traits.

pub mod algorithm;
pub mod utils;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

use crate::algorithm::simulation::Simulation;

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let graph = graph::labeled_graph::StandardLabeledGraph::new();
//         // let sim = graph.get_simulation();
//     }
    
// }
