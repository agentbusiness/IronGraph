#![allow(clippy::all, clippy::nursery, clippy::pedantic)]

//! Consolidated GPU integration-test harness.
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
