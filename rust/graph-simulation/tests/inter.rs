//! Fixture-backed regression test for simulation between two labelled graphs.

use graph_base::impls::standard::StandardLabeledGraph;
use graph_simulation::algorithm::simulation::Simulation;
use std::fs;

#[test]
#[ignore = "requires external benchmark fixtures under data/label_graph/simulation_test"]
fn simulation_inter_test() {
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
            }
        }
    }
}
