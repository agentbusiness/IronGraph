#![allow(clippy::all, clippy::nursery, clippy::pedantic)]

//! Consolidated graph and storage integration-test harness.

#[path = "../tests/canonical_property_shape_validation.rs"]
mod canonical_property_shape_validation;
#[path = "../tests/mixed_property_types.rs"]
mod mixed_property_types;

#[path = "../tests/document_properties.rs"]
mod document_properties;
#[path = "../tests/document_property_wire_round_trip.rs"]
mod document_property_wire_round_trip;
#[path = "../tests/durable_sync_barrier.rs"]
mod durable_sync_barrier;
#[path = "../tests/graph_algorithm_procedures.rs"]
mod graph_algorithm_procedures;
#[path = "../tests/no_fixed_query_ceiling.rs"]
mod no_fixed_query_ceiling;
#[path = "../tests/reachability_endpoint_only.rs"]
mod reachability_endpoint_only;
#[path = "../tests/temporal_duration_arithmetic_cpu.rs"]
mod temporal_duration_arithmetic_cpu;
#[path = "../tests/temporal_index_lifecycle.rs"]
mod temporal_index_lifecycle;
#[path = "../tests/temporal_projection_cpu.rs"]
mod temporal_projection_cpu;
#[path = "../tests/temporal_truncate_cpu.rs"]
mod temporal_truncate_cpu;
