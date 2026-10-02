//! PyO3 extension exposing graph and Hyper Simulation algorithms to Python.
//!
//! Python-facing types intentionally use stable integer identifiers so one
//! logical HC can be registered once and associated with multiple Delta pairs.

pub mod graph;
pub mod utils;

use pyo3::prelude::*;


#[pymodule]
fn simulation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // graph::register_graph_module(m)?;
    m.add_class::<graph::hypergraph::Hypergraph>()?;
    m.add_class::<graph::hypergraph::Node>()?;
    m.add_class::<graph::hypergraph::Hyperedge>()?;
    m.add_class::<graph::hypergraph::LMatchImpl>()?;
    m.add_class::<graph::hypergraph::DeltaPy>()?;
    m.add_class::<graph::hypergraph::DMatchImpl>()?;
    m.add_class::<graph::hypergraph::Event>()?;

    m.add_function(wrap_pyfunction!(graph::networkx_graph::get_simulation_inter, m)?)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::get_simulation_inter_fn, m)?)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::is_simulation_isomorphic, m)?)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::is_simulation_isomorphic_fn, m)?)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::is_simulation_isomorphic_of_edge_fn, m)?)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::is_simulation_isomorphic_of_node_edge_fn, m)?)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::get_bounded_simulation, m)?)?;
    Ok(())
}
