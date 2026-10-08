#![allow(clippy::all, clippy::nursery, clippy::pedantic)]

//! Consolidated GPU integration-test harness.

// Explicitly inactive fixture translation for differential comparison. The active database never
// calls this helper or constructs the resident fixture that supplies its input.
fn canonical_fixture(
    legacy: &irongraph::graph::legacy::GraphStore,
) -> irongraph::Result<irongraph::graph::GraphStore> {
    use irongraph::graph::{EdgeInput, GraphMutation, GraphStore, NodeInput};
    let graph = GraphStore::default();
    for (id, name) in legacy.catalog().labels() {
        graph.apply(GraphMutation::DeclareLabel {
            name: name.to_owned(),
            id,
        })?;
    }
    for (id, name) in legacy.catalog().properties() {
        graph.apply(GraphMutation::DeclareProperty {
            name: name.to_owned(),
            id,
        })?;
    }
    for (id, name) in legacy.catalog().relationship_types() {
        graph.apply(GraphMutation::DeclareRelationshipType {
            name: name.to_owned(),
            id,
        })?;
    }
    for node in legacy.nodes() {
        graph.insert_node(NodeInput {
            id: node.id(),
            layer: node.layer(),
            revision: node.revision(),
            labels: node.labels().to_vec(),
            properties: node.properties(),
        })?;
    }
    for edge in legacy.edges() {
        graph.insert_edge(EdgeInput {
            id: edge.id(),
            source: edge.source(),
            target: edge.target(),
            relationship_type: edge.relationship_type(),
            layer: edge.layer(),
            revision: edge.revision(),
            properties: edge.properties(),
        })?;
    }
    Ok(graph)
}

fn canonical_temporal_fixture(
    legacy: &irongraph::graph::legacy::TemporalStore,
) -> irongraph::Result<irongraph::graph::TemporalStore> {
    let encoded = postcard::to_stdvec(legacy)
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    postcard::from_bytes(&encoded).map_err(|error| irongraph::Error::internal(error.to_string()))
}

#[path = "../tests/gpu_float_arithmetic_differential.rs"]
mod gpu_float_arithmetic_differential;
#[path = "../tests/gpu_graph_algorithm_contract.rs"]
mod gpu_graph_algorithm_contract;
#[path = "../tests/gpu_mutation_intents.rs"]
mod gpu_mutation_intents;
#[path = "../tests/gpu_pattern1_bound_endpoint_pairs.rs"]
mod gpu_pattern1_bound_endpoint_pairs;
#[path = "../tests/gpu_pattern1_resident_predicates.rs"]
mod gpu_pattern1_resident_predicates;
#[path = "../tests/gpu_pattern_parallel_compaction.rs"]
mod gpu_pattern_parallel_compaction;
#[path = "../tests/gpu_resident_quantifier_entity_source_cpu.rs"]
mod gpu_resident_quantifier_entity_source_cpu;
#[path = "../tests/gpu_resident_quantifier_entity_source_metal.rs"]
mod gpu_resident_quantifier_entity_source_metal;
#[path = "../tests/gpu_resident_quantifier_parallelism_metal.rs"]
mod gpu_resident_quantifier_parallelism_metal;
#[path = "../tests/gpu_resident_quantifier_program_metal.rs"]
mod gpu_resident_quantifier_program_metal;
#[path = "../tests/gpu_resident_row_numeric_adversarial.rs"]
mod gpu_resident_row_numeric_adversarial;
#[path = "../tests/gpu_resident_row_program_cpu.rs"]
mod gpu_resident_row_program_cpu;
#[path = "../tests/gpu_resident_row_program_metal.rs"]
mod gpu_resident_row_program_metal;
#[path = "../tests/gpu_resident_row_string_arena.rs"]
mod gpu_resident_row_string_arena;
#[path = "../tests/gpu_row_create_metal.rs"]
mod gpu_row_create_metal;
#[path = "../tests/gpu_row_match_merge_relationship_cpu.rs"]
mod gpu_row_match_merge_relationship_cpu;
#[path = "../tests/gpu_row_mutation_cpu.rs"]
mod gpu_row_mutation_cpu;
#[path = "../tests/gpu_scalar_status_propagation.rs"]
mod gpu_scalar_status_propagation;
#[path = "../tests/gpu_variable_length_path_program.rs"]
mod gpu_variable_length_path_program;
#[path = "../tests/graph_query_gpu.rs"]
mod graph_query_gpu;
