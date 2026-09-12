#![allow(clippy::all, clippy::nursery, clippy::pedantic)]

//! Consolidated Cypher integration-test harness.
#[path = "../tests/aggregate_over_aggregate_declines_to_the_host.rs"]
mod aggregate_over_aggregate_declines_to_the_host;
#[path = "../tests/cypher_aggregate_expressions.rs"]
mod cypher_aggregate_expressions;
#[path = "../tests/cypher_aggregation_semantic_validation_cpu.rs"]
mod cypher_aggregation_semantic_validation_cpu;
#[path = "../tests/cypher_binding_kinds.rs"]
mod cypher_binding_kinds;
#[path = "../tests/cypher_bound_write_pattern_binder.rs"]
mod cypher_bound_write_pattern_binder;
#[path = "../tests/cypher_collected_entity_mutation_cpu.rs"]
mod cypher_collected_entity_mutation_cpu;
#[path = "../tests/cypher_comparison_chain_parser.rs"]
mod cypher_comparison_chain_parser;
#[path = "../tests/cypher_comprehension_aggregation_binding.rs"]
mod cypher_comprehension_aggregation_binding;
#[path = "../tests/cypher_delete_constraint_semantics.rs"]
mod cypher_delete_constraint_semantics;
#[path = "../tests/cypher_delete_tck_semantics_cpu.rs"]
mod cypher_delete_tck_semantics_cpu;
#[path = "../tests/cypher_deleted_entity_access_cpu.rs"]
mod cypher_deleted_entity_access_cpu;
#[path = "../tests/cypher_duration_binding.rs"]
mod cypher_duration_binding;
#[path = "../tests/cypher_dynamic_index_cpu.rs"]
mod cypher_dynamic_index_cpu;
#[path = "../tests/cypher_empty_variable_length_range_cpu.rs"]
mod cypher_empty_variable_length_range_cpu;
#[path = "../tests/cypher_entity_label_predicate_binding.rs"]
mod cypher_entity_label_predicate_binding;
#[path = "../tests/cypher_entity_list_provenance_binding.rs"]
mod cypher_entity_list_provenance_binding;
#[path = "../tests/cypher_existential_subquery_cpu.rs"]
mod cypher_existential_subquery_cpu;
#[path = "../tests/cypher_graph5_entity_label_predicate_cpu.rs"]
mod cypher_graph5_entity_label_predicate_cpu;
#[path = "../tests/cypher_graph_function_runtime_errors.rs"]
mod cypher_graph_function_runtime_errors;
#[path = "../tests/cypher_graph_function_static_binding.rs"]
mod cypher_graph_function_static_binding;
#[path = "../tests/cypher_lexer_error_details.rs"]
mod cypher_lexer_error_details;
#[path = "../tests/cypher_list_slice_cpu.rs"]
mod cypher_list_slice_cpu;
#[path = "../tests/cypher_match_relationship_uniqueness_cpu.rs"]
mod cypher_match_relationship_uniqueness_cpu;
#[path = "../tests/cypher_merge_entity_map_path_null_cpu.rs"]
mod cypher_merge_entity_map_path_null_cpu;
#[path = "../tests/cypher_modulo_cpu.rs"]
mod cypher_modulo_cpu;
#[path = "../tests/cypher_nan_equality.rs"]
mod cypher_nan_equality;
#[path = "../tests/cypher_native_bound_relationship_merge_route.rs"]
mod cypher_native_bound_relationship_merge_route;
#[path = "../tests/cypher_native_computed_order_route.rs"]
mod cypher_native_computed_order_route;
#[path = "../tests/cypher_native_create_node_route.rs"]
mod cypher_native_create_node_route;
#[path = "../tests/cypher_native_create_post_write_route.rs"]
mod cypher_native_create_post_write_route;
#[path = "../tests/cypher_native_create_relationship_route.rs"]
mod cypher_native_create_relationship_route;
#[path = "../tests/cypher_native_create_temporal_array_route.rs"]
mod cypher_native_create_temporal_array_route;
#[path = "../tests/cypher_native_delete5_selector_route.rs"]
mod cypher_native_delete5_selector_route;
#[path = "../tests/cypher_native_delete_continuation_route.rs"]
mod cypher_native_delete_continuation_route;
#[path = "../tests/cypher_native_delete_remaining_contract.rs"]
mod cypher_native_delete_remaining_contract;
#[path = "../tests/cypher_native_dynamic_property_route.rs"]
mod cypher_native_dynamic_property_route;
#[path = "../tests/cypher_native_entity_keys_pattern_comprehension_route.rs"]
mod cypher_native_entity_keys_pattern_comprehension_route;
#[path = "../tests/cypher_native_entity_label_predicate_output_route.rs"]
mod cypher_native_entity_label_predicate_output_route;
#[path = "../tests/cypher_native_fused_aggregation_tck_route.rs"]
mod cypher_native_fused_aggregation_tck_route;
#[path = "../tests/cypher_native_label_comprehension_membership_route.rs"]
mod cypher_native_label_comprehension_membership_route;
#[path = "../tests/cypher_native_list12_remaining_manifest.rs"]
mod cypher_native_list12_remaining_manifest;
#[path = "../tests/cypher_native_list_add_route.rs"]
mod cypher_native_list_add_route;
#[path = "../tests/cypher_native_list_size_route.rs"]
mod cypher_native_list_size_route;
#[path = "../tests/cypher_native_list_slice_bounds_route.rs"]
mod cypher_native_list_slice_bounds_route;
#[path = "../tests/cypher_native_literal_map_set_route.rs"]
mod cypher_native_literal_map_set_route;
#[path = "../tests/cypher_native_map_aggregate_freeze_route.rs"]
mod cypher_native_map_aggregate_freeze_route;
#[path = "../tests/cypher_native_map_keys_route.rs"]
mod cypher_native_map_keys_route;
#[path = "../tests/cypher_native_map_property_route.rs"]
mod cypher_native_map_property_route;
#[path = "../tests/cypher_native_merge9_interoperation_route.rs"]
mod cypher_native_merge9_interoperation_route;
#[path = "../tests/cypher_native_merge_on_create_route.rs"]
mod cypher_native_merge_on_create_route;
#[path = "../tests/cypher_native_merge_on_match_literal_route.rs"]
mod cypher_native_merge_on_match_literal_route;
#[path = "../tests/cypher_native_mutation_continuation_route.rs"]
mod cypher_native_mutation_continuation_route;
#[path = "../tests/cypher_native_mutation_route.rs"]
mod cypher_native_mutation_route;
#[path = "../tests/cypher_native_null_property_mutation_route.rs"]
mod cypher_native_null_property_mutation_route;
#[path = "../tests/cypher_native_optional_conformance_route.rs"]
mod cypher_native_optional_conformance_route;
#[path = "../tests/cypher_native_ordered_create_merge_route.rs"]
mod cypher_native_ordered_create_merge_route;
#[path = "../tests/cypher_native_pattern_comprehension_route.rs"]
mod cypher_native_pattern_comprehension_route;
#[path = "../tests/cypher_native_pattern_pair_predicate_route.rs"]
mod cypher_native_pattern_pair_predicate_route;
#[path = "../tests/cypher_native_pattern_predicate_route.rs"]
mod cypher_native_pattern_predicate_route;
#[path = "../tests/cypher_native_percentile_aggregation_route.rs"]
mod cypher_native_percentile_aggregation_route;
#[path = "../tests/cypher_native_post_write_route.rs"]
mod cypher_native_post_write_route;
#[path = "../tests/cypher_native_procedure_table_route.rs"]
mod cypher_native_procedure_table_route;
#[path = "../tests/cypher_native_quantifier_conformance_route.rs"]
mod cypher_native_quantifier_conformance_route;
#[path = "../tests/cypher_native_relationship_remove_continuation_route.rs"]
mod cypher_native_relationship_remove_continuation_route;
#[path = "../tests/cypher_native_relationship_type_route.rs"]
mod cypher_native_relationship_type_route;
#[path = "../tests/cypher_native_scalar_precedence_route.rs"]
mod cypher_native_scalar_precedence_route;
#[path = "../tests/cypher_native_segmented_aggregation_execution.rs"]
mod cypher_native_segmented_aggregation_execution;
#[path = "../tests/cypher_native_segmented_aggregation_route.rs"]
mod cypher_native_segmented_aggregation_route;
#[path = "../tests/cypher_native_set1_list_remaining_contract.rs"]
mod cypher_native_set1_list_remaining_contract;
#[path = "../tests/cypher_native_set_property_route.rs"]
mod cypher_native_set_property_route;
#[path = "../tests/cypher_native_simple_case_route.rs"]
mod cypher_native_simple_case_route;
#[path = "../tests/cypher_native_string_predicate_route.rs"]
mod cypher_native_string_predicate_route;
#[path = "../tests/cypher_native_string_predicate_scalar_route.rs"]
mod cypher_native_string_predicate_scalar_route;
#[path = "../tests/cypher_native_substring_scalar_route.rs"]
mod cypher_native_substring_scalar_route;
#[path = "../tests/cypher_native_temporal_arithmetic_route.rs"]
mod cypher_native_temporal_arithmetic_route;
#[path = "../tests/cypher_native_temporal_clock_null_route.rs"]
mod cypher_native_temporal_clock_null_route;
#[path = "../tests/cypher_native_temporal_comparison_route.rs"]
mod cypher_native_temporal_comparison_route;
#[path = "../tests/cypher_native_temporal_duration_order_route.rs"]
mod cypher_native_temporal_duration_order_route;
#[path = "../tests/cypher_native_temporal_property_order_route.rs"]
mod cypher_native_temporal_property_order_route;
#[path = "../tests/cypher_native_temporal_serialization_route.rs"]
mod cypher_native_temporal_serialization_route;
#[path = "../tests/cypher_native_temporal_unwind_route.rs"]
mod cypher_native_temporal_unwind_route;
#[path = "../tests/cypher_native_type_conversion4_boolean_property_to_string_route.rs"]
mod cypher_native_type_conversion4_boolean_property_to_string_route;
#[path = "../tests/cypher_native_type_conversion4_to_string_route.rs"]
mod cypher_native_type_conversion4_to_string_route;
#[path = "../tests/cypher_native_unwind_match_merge_relationship_route.rs"]
mod cypher_native_unwind_match_merge_relationship_route;
#[path = "../tests/cypher_native_unwind_merge_route.rs"]
mod cypher_native_unwind_merge_route;
#[path = "../tests/cypher_native_variable_length_match_route.rs"]
mod cypher_native_variable_length_match_route;
#[path = "../tests/cypher_native_variable_path_outputs_route.rs"]
mod cypher_native_variable_path_outputs_route;
#[path = "../tests/cypher_native_with_order_consistency_manifest.rs"]
mod cypher_native_with_order_consistency_manifest;
#[path = "../tests/cypher_native_with_order_scalar_consistency_route.rs"]
mod cypher_native_with_order_scalar_consistency_route;
#[path = "../tests/cypher_native_with_order_temporal_consistency_route.rs"]
mod cypher_native_with_order_temporal_consistency_route;
#[path = "../tests/cypher_node_label_predicate_parser.rs"]
mod cypher_node_label_predicate_parser;
#[path = "../tests/cypher_optional_match_where_cpu.rs"]
mod cypher_optional_match_where_cpu;
#[path = "../tests/cypher_order_by_aggregation_scope.rs"]
mod cypher_order_by_aggregation_scope;
#[path = "../tests/cypher_parser_delete_targets.rs"]
mod cypher_parser_delete_targets;
#[path = "../tests/cypher_parser_merge_direction.rs"]
mod cypher_parser_merge_direction;
#[path = "../tests/cypher_parser_pattern_semantics.rs"]
mod cypher_parser_pattern_semantics;
#[path = "../tests/cypher_parser_precedence.rs"]
mod cypher_parser_precedence;
#[path = "../tests/cypher_parser_radix_literals.rs"]
mod cypher_parser_radix_literals;
#[path = "../tests/cypher_parser_relationship_errors.rs"]
mod cypher_parser_relationship_errors;
#[path = "../tests/cypher_parser_set_targets.rs"]
mod cypher_parser_set_targets;
#[path = "../tests/cypher_pattern1_predicate_cpu.rs"]
mod cypher_pattern1_predicate_cpu;
#[path = "../tests/cypher_pattern1_predicate_frontend.rs"]
mod cypher_pattern1_predicate_frontend;
#[path = "../tests/cypher_pattern2_comprehension_cpu.rs"]
mod cypher_pattern2_comprehension_cpu;
#[path = "../tests/cypher_pattern2_comprehension_frontend.rs"]
mod cypher_pattern2_comprehension_frontend;
#[path = "../tests/cypher_percentile_preflight_parity.rs"]
mod cypher_percentile_preflight_parity;
#[path = "../tests/cypher_percentile_range_cpu.rs"]
mod cypher_percentile_range_cpu;
#[path = "../tests/cypher_procedure_catalog.rs"]
mod cypher_procedure_catalog;
#[path = "../tests/cypher_projection_boundary_binding.rs"]
mod cypher_projection_boundary_binding;
#[path = "../tests/cypher_projection_source_text.rs"]
mod cypher_projection_source_text;
#[path = "../tests/cypher_quantifier_static_binding.rs"]
mod cypher_quantifier_static_binding;
#[path = "../tests/cypher_random.rs"]
mod cypher_random;
#[path = "../tests/cypher_remaining_runtime_semantics_cpu.rs"]
mod cypher_remaining_runtime_semantics_cpu;
#[path = "../tests/cypher_remove_null_semantics_cpu.rs"]
mod cypher_remove_null_semantics_cpu;
#[path = "../tests/cypher_resident_procedure_table.rs"]
mod cypher_resident_procedure_table;
#[path = "../tests/cypher_set_semantics_cpu.rs"]
mod cypher_set_semantics_cpu;
#[path = "../tests/cypher_skip_limit_binding.rs"]
mod cypher_skip_limit_binding;
#[path = "../tests/cypher_static_source_shape_binding.rs"]
mod cypher_static_source_shape_binding;
#[path = "../tests/cypher_string_predicate_null_semantics_cpu.rs"]
mod cypher_string_predicate_null_semantics_cpu;
#[path = "../tests/cypher_temporal5_property_access_cpu.rs"]
mod cypher_temporal5_property_access_cpu;
#[path = "../tests/cypher_temporal6_duration_rendering_cpu.rs"]
mod cypher_temporal6_duration_rendering_cpu;
#[path = "../tests/cypher_temporal_clock_function_binding.rs"]
mod cypher_temporal_clock_function_binding;
#[path = "../tests/cypher_union_mode_parser.rs"]
mod cypher_union_mode_parser;
#[path = "../tests/cypher_union_schema_planner.rs"]
mod cypher_union_schema_planner;
#[path = "../tests/cypher_variable_length_aggregate_route.rs"]
mod cypher_variable_length_aggregate_route;
#[path = "../tests/cypher_variable_length_relationship_binding.rs"]
mod cypher_variable_length_relationship_binding;
#[path = "../tests/cypher_with_order_by_binding.rs"]
mod cypher_with_order_by_binding;
#[path = "../tests/cypher_with_where_cpu.rs"]
mod cypher_with_where_cpu;
#[path = "../tests/cypher_write_pattern_binding.rs"]
mod cypher_write_pattern_binding;
#[path = "../tests/opencypher_conformance.rs"]
mod opencypher_conformance;
#[path = "../tests/opencypher_tck.rs"]
mod opencypher_tck;
#[path = "../tests/temporal_property_read_agrees_across_paths.rs"]
mod temporal_property_read_agrees_across_paths;
