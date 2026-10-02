//! Integration examples for native, inter-graph, and isomorphism predicates.

use std::collections::{HashMap, HashSet};
use std::env;

use graph_base::impls::standard::StandardLabeledGraph;
use graph_base::interfaces::graph::Graph;
use graph_simulation::algorithm::simulation::Simulation;
use std::fs;

fn display_sim<'a, N: 'a + std::fmt::Display + Eq + std::hash::Hash>(
    sim: &HashMap<&N, HashSet<&N>>,
) {
    for (k, v) in sim {
        print!("{} -> {{", k);
        for u in v {
            print!("{}, ", u);
        }
        println!("}}");
    }
}

#[test]
#[ignore = "requires external benchmark fixtures under data/label_graph/simulation_test"]
fn test_simulation() {
    let paths = fs::read_dir(format!(
        "{}/data/label_graph/simulation_test/",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("Unable to read directory");
    for path in paths {
        if let Ok(entry) = path {
            let path = entry.path();
            if path.is_file() {
                let graph_name = path.file_name().unwrap().to_str().unwrap();
                let content = fs::read_to_string(&path).expect("Unable to read file");

                let mut lines = content.split_whitespace();
                let is_true = lines.next().unwrap() == "t";
                let n1: usize = lines.next().unwrap().parse().unwrap();
                let m1: usize = lines.next().unwrap().parse().unwrap();
                let _: usize = lines.next().unwrap().parse().unwrap();

                let mut graph1 = StandardLabeledGraph::new();
                for _ in 0..n1 {
                    let node: u64 = lines.next().unwrap().parse().unwrap();
                    let label: String = lines.next().unwrap().parse().unwrap();
                    graph1.add_node(node, label);
                }

                for _ in 0..m1 {
                    let source: u64 = lines.next().unwrap().parse().unwrap();
                    let destination: u64 = lines.next().unwrap().parse().unwrap();
                    graph1.add_edge(source, destination);
                }

                let n2: usize = lines.next().unwrap().parse().unwrap();
                let m2: usize = lines.next().unwrap().parse().unwrap();
                let _: usize = lines.next().unwrap().parse().unwrap();

                let mut graph2 = StandardLabeledGraph::new();
                for _ in 0..n2 {
                    let node: u64 = lines.next().unwrap().parse().unwrap();
                    let label: String = lines.next().unwrap().parse().unwrap();
                    graph2.add_node(node, label);
                }

                for _ in 0..m2 {
                    let source: u64 = lines.next().unwrap().parse().unwrap();
                    let destination: u64 = lines.next().unwrap().parse().unwrap();
                    graph2.add_edge(source, destination);
                }

                let sim = graph1.get_simulation_inter(&graph2);
                let has_sim = StandardLabeledGraph::has_simulation(sim);

                match (is_true, has_sim) {
                    (true, true) => assert!(true),
                    (false, false) => assert!(true),
                    (true, false) => {
                        println!(
                            "{}: Test failed at: Expected isomorphic, got no simulation",
                            graph_name
                        );
                        assert!(false);
                    }
                    (false, true) => {
                        println!(
                            "{}: Test warm: graphs no isomorphic, got simulation",
                            graph_name
                        );
                        // assert!(true);
                    }
                }
            }
        }
    }
    // assert!(false);
}

#[test]
#[ignore = "requires external benchmark fixtures under data/label_graph/simulation_test"]
fn simulation_same() {
    let paths = fs::read_dir(format!(
        "{}/data/label_graph/simulation_test/",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("Unable to read directory");
    for path in paths {
        if let Ok(entry) = path {
            let path = entry.path();
            if path.is_file() {
                let graph_name = path.file_name().unwrap().to_str().unwrap();
                let content = fs::read_to_string(&path).expect("Unable to read file");

                let mut lines = content.split_whitespace();
                let is_true = lines.next().unwrap() == "t";
                let n1: usize = lines.next().unwrap().parse().unwrap();
                let m1: usize = lines.next().unwrap().parse().unwrap();
                let _: usize = lines.next().unwrap().parse().unwrap();

                let mut graph1 = StandardLabeledGraph::new();
                for _ in 0..n1 {
                    let node: u64 = lines.next().unwrap().parse().unwrap();
                    let label: String = lines.next().unwrap().parse().unwrap();
                    graph1.add_node(node, label);
                }

                for _ in 0..m1 {
                    let source: u64 = lines.next().unwrap().parse().unwrap();
                    let destination: u64 = lines.next().unwrap().parse().unwrap();
                    graph1.add_edge(source, destination);
                }

                let n2: usize = lines.next().unwrap().parse().unwrap();
                let m2: usize = lines.next().unwrap().parse().unwrap();
                let _: usize = lines.next().unwrap().parse().unwrap();

                let mut graph2 = StandardLabeledGraph::new();
                for _ in 0..n2 {
                    let node: u64 = lines.next().unwrap().parse().unwrap();
                    let label: String = lines.next().unwrap().parse().unwrap();
                    graph2.add_node(node, label);
                }

                for _ in 0..m2 {
                    let source: u64 = lines.next().unwrap().parse().unwrap();
                    let destination: u64 = lines.next().unwrap().parse().unwrap();
                    graph2.add_edge(source, destination);
                }

                let sim1 = graph1.get_simulation_inter(&graph2);
                let sim2 = graph1.get_simulation_native(&graph2);

                println!(
                    "graphs {} is {}",
                    graph_name,
                    if is_true {
                        "isomorphic"
                    } else {
                        "not isomorphic"
                    }
                );
                println!("sim1:");
                display_sim(&sim1);
                println!("sim2:");
                display_sim(&sim2);

                println!("graph1:\n{}", graph1);
                println!("graph2:\n{}", graph2);

                for u in graph1.nodes() {
                    let sim1_u = sim1.get(u).unwrap();
                    let sim2_u = sim2.get(u).unwrap();

                    // println!("{}: {} == {}", graph_name, sim1_u, sim2_u);

                    assert!(sim1_u == sim2_u);
                }
            }
        }
    }
}
