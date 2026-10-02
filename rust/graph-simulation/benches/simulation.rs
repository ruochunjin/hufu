//! Criterion benchmarks for native and inter-graph simulation workloads.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use graph_base::impls::standard::StandardLabeledGraph;
use graph_simulation::algorithm::simulation::Simulation;
use std::fs;

fn simulation_inter_test() {
    let paths = fs::read_dir(format!("{}/data/label_graph/simulation_test/", env!("CARGO_MANIFEST_DIR"))).expect("Unable to read directory");
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
                    let has_sim=  StandardLabeledGraph::has_simulation(sim);
            }
        }
    }
}

fn simulation_native_test() {
    let paths = fs::read_dir(format!("{}/data/label_graph/simulation_test/", env!("CARGO_MANIFEST_DIR"))).expect("Unable to read directory");
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

                    let sim = graph1.get_simulation_native(&graph2);
                    let has_sim=  StandardLabeledGraph::has_simulation(sim);
            }
        }
    }
}

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n-1) + fibonacci(n-2),
    }
}

pub fn simulation(c: &mut Criterion) {
    c.bench_function("simulation_inter", |b| b.iter(|| {
        simulation_inter_test();
    }));
    c.bench_function("simulation_native", |b| b.iter(|| {
        simulation_native_test();
    }));
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, simulation);
criterion_main!(benches);
