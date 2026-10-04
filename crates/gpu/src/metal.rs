//! Metal-first complete-residency backend using Candle's production Metal kernels.

use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
    sync::Arc,
};

use candle_core::{Device, Tensor};
use parking_lot::ReentrantMutexGuard;
use tokio_util::sync::CancellationToken;

use crate::{
    Bookmark, Error, ErrorCode, ProjectId, Result,
    graph::{IvfPqBuildPlan, IvfPqConfig, IvfPqIndex, LayerMask, PersistentMap, VectorIndex},
    types::{LabelId, PropertyId, ScalarValue},
};

use super::{
    BackendKind, CompareOp, DeviceMemoryGovernor, DistanceBatch, ExecutionBackend,
    ResidentBooleanAggregateProgramRequest, ResidentBooleanProgramRequest,
    ResidentBooleanProgramResult, ResidentCreateNodeRequest, ResidentCreateNodeResult,
    ResidentDeleteRequest, ResidentDeleteResult, ResidentGraphProcedure,
    ResidentGraphProcedureRequest, ResidentGraphProcedureResult, ResidentGroup,
    ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
    ResidentMultiwayIntersectionRequest, ResidentNodeGroupPipelineRequest,
    ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentNullableRelationRequest,
    ResidentNullableRelationResult, ResidentPatternCountRequest, ResidentPatternCountResult,
    ResidentPatternPairPredicateRequest, ResidentPatternPairPredicateResult,
    ResidentPatternPredicateRequest, ResidentPatternPredicateResult, ResidentProcedureTableRequest,
    ResidentProcedureTableResult, ResidentProjectDelta, ResidentProjectImage,
    ResidentQuantifierProgramRequest, ResidentQuantifierProgramResult, ResidentRangeProgramRequest,
    ResidentRangeProgramResult, ResidentRowMutationRequest, ResidentRowMutationResult,
    ResidentRowProgramRequest, ResidentRowProgramResult, ResidentScalarInput,
    ResidentScalarProgramRequest, ResidentScalarProgramResult, ResidentSegmentedAggregationRequest,
    ResidentSegmentedAggregationResult, ResidentSortRequest, ResidentSortResult,
    ResidentTemporalArithmeticRequest, ResidentTemporalArithmeticResult,
    ResidentTemporalPipelineRequest, ResidentTemporalPipelineResult,
    ResidentTemporalValueProgramRequest, ResidentTemporalValueProgramResult,
    ResidentToBooleanRequest, ResidentToBooleanResult, ResidentVariablePathRequest,
    ResidentVariablePathResult, ResidentVectorQuery, ResidentVectorResult,
    accelerator::{
        CandleIvfPqBuildKernel, CandleResident, candle_error, exact_l2,
        execute_metal_boolean_aggregate_program, execute_metal_boolean_program,
        execute_metal_create_node, execute_metal_procedure_table, execute_metal_quantifier_program,
        execute_metal_range_program, execute_metal_scalar_program,
        execute_metal_segmented_aggregation, execute_metal_temporal_arithmetic_program,
        execute_metal_temporal_value_program, execute_metal_to_boolean,
        execute_metal_variable_path, filter_tensor, intersect_sorted_node_sets,
        metal_breadth_first_scratch_bytes, metal_degree_scratch_bytes,
        metal_depth_first_scratch_bytes, metal_graph_metrics_scratch_bytes,
        metal_kcore_scratch_bytes, metal_louvain_scratch_bytes, metal_pagerank_scratch_bytes,
        metal_pages::{BranchPages, Capture, ExclusiveWrite},
        metal_quantifier_program_scratch_bytes, metal_resident_row_program_scratch_bytes,
        metal_scc_scratch_bytes, metal_shortest_path_scratch_bytes,
        metal_temporal_arithmetic_scratch_bytes, metal_unit_dijkstra_scratch_bytes,
        metal_wcc_scratch_bytes, metal_weighted_dijkstra_scratch_bytes,
        prepare_metal_graph_algorithm_kernels, prepare_metal_segmented_aggregation,
        prepare_metal_sort_kernels, prepare_metal_temporal_value_program,
    },
    device::construct_metal_device,
    ensure_graph_execution, ensure_not_cancelled, metal_sort_scratch_bytes, operator_scratch_bytes,
    validate_matrix, vector_query_scratch_bytes,
};

/// Production Metal backend. Construction fails unless a real Metal device is available.
pub struct MetalBackend {
    device: Device,
    /// Serialises every command stream on `device`. See [`crate::metal_gate`]: Candle's Metal
    /// backend shares one buffer pool pair and one `MTLResidencySet` per device and mutates them
    /// without a covering lock, so two threads on one device is undefined behaviour — and
    /// `pin_project` hands this exact device to every concurrent read.
    gate: &'static parking_lot::ReentrantMutex<()>,
    resident: BTreeMap<ProjectId, Arc<CandleResident>>,
    /// The device's maximum single-buffer length, used as the native command-buffer scratch ceiling
    /// so a high-memory GPU is allowed the largest command it can physically hold.
    max_buffer_length: usize,
    /// Native allocations/pages private to this pinned branch. None uses the conservative
    /// complete-footprint fallback after explicit cold replacement of an existing branch.
    private_pages: Option<BranchPages>,
    host: BTreeMap<ProjectId, HostFootprint>,
    private_host: HostFootprint,
    // Release retained data before its accounting token.
    governor: DeviceMemoryGovernor,
}

#[derive(Clone, Default)]
struct HostFootprint {
    cold_bytes: usize,
    owners: PersistentMap<HostOwner>,
    cold_temporal_owners: PersistentMap<()>,
    private_bytes: usize,
    history_entries: u64,
    dictionary_base: [(usize, usize); 2],
    dictionary_bytes: usize,
}

#[derive(Clone, Copy, Default)]
struct HostOwner {
    fixed: usize,
    payload: usize,
}

impl HostOwner {
    fn bytes(self) -> usize {
        self.fixed.saturating_add(self.payload)
    }
}

fn scalar_host_bytes(value: &ScalarValue) -> usize {
    match value {
        ScalarValue::String(value) => value.len(),
        ScalarValue::Bytes(value) => value.len(),
        ScalarValue::List(value) => value.as_bytes().len(),
        ScalarValue::Map(value) => value.as_bytes().len(),
        ScalarValue::ZonedDateTime { timezone, .. } => 32 + timezone.len(),
        _ => 32,
    }
}

impl HostFootprint {
    fn dictionary_stats(columns: &crate::graph::PropertyColumns) -> (usize, usize) {
        let dictionary = columns.string_dictionary();
        (dictionary.len(), dictionary.byte_len())
    }

    fn branch(resident: &CandleResident) -> Self {
        Self {
            dictionary_base: resident
                .shared_graph_backing()
                .map_or([(0, 0); 2], |backing| {
                    [
                        Self::dictionary_stats(&backing.node_properties),
                        Self::dictionary_stats(&backing.edge_properties),
                    ]
                }),
            ..Self::default()
        }
    }

    fn update_dictionaries(&mut self, resident: &CandleResident) {
        let Some(backing) = resident.shared_graph_backing() else {
            return;
        };
        self.dictionary_bytes = [
            Self::dictionary_stats(&backing.node_properties),
            Self::dictionary_stats(&backing.edge_properties),
        ]
        .into_iter()
        .zip(self.dictionary_base)
        .fold(
            0_usize,
            |bytes, ((entries, payload), (base_entries, base_payload))| {
                let added = entries.saturating_sub(base_entries);
                bytes
                    .saturating_add(added.saturating_mul(512))
                    .saturating_add(payload.saturating_sub(base_payload))
                    .saturating_add(usize::from(added != 0) * 64 * 1024)
            },
        );
    }

    fn cold(image: &ResidentProjectImage) -> Self {
        let mut cold_temporal_owners = PersistentMap::default();
        for column in &image.temporal_canonical {
            let key = (7_u128 << 120)
                | ((column.entity_kind as u128) << 112)
                | (u128::from(column.property.0) << 64)
                | u128::from(column.target);
            cold_temporal_owners.insert_cow(key, ());
            cold_temporal_owners.insert_cow(Self::group_key(key, column.target), ());
        }
        let dictionary_base = [
            Self::dictionary_stats(&image.graph.node_properties),
            Self::dictionary_stats(&image.graph.edge_properties),
        ];
        let rows = image
            .graph
            .node_ids
            .len()
            .saturating_add(image.graph.edge_ids.len())
            .saturating_add(
                image
                    .indexes
                    .vectors
                    .iter()
                    .map(|column| column.entity_ids.len())
                    .sum::<usize>(),
            )
            .saturating_add(
                image
                    .temporal
                    .columns
                    .iter()
                    .map(|column| column.entity_ids.len())
                    .sum::<usize>(),
            );
        // Canonical value bytes share native allocations after rebind. Identity radix nodes,
        // page directories and container roots remain independently retained host allocations.
        Self {
            cold_bytes: rows
                .saturating_mul(512)
                .saturating_add(
                    image
                        .resident_bytes()
                        .div_ceil(16 * 1024)
                        .saturating_mul(512),
                )
                .saturating_add(64 * 1024)
                .saturating_add(
                    dictionary_base
                        .iter()
                        .map(|(entries, _)| entries.saturating_mul(512))
                        .sum::<usize>(),
                ),
            dictionary_base,
            cold_temporal_owners,
            ..Self::default()
        }
    }

    fn bytes(&self) -> usize {
        self.cold_bytes
            .saturating_add(self.private_bytes)
            .saturating_add(self.dictionary_bytes)
    }

    fn cold_staging_bytes(&self, image: &ResidentProjectImage) -> usize {
        // Upload retains these already allocated persistent identity maps by cloning their
        // roots. Keep them in total resident accounting, but reserve only new allocations
        // against the host's currently available physical memory. Missing maps are rebuilt
        // during upload and therefore still need their complete staging allowance.
        let shared_nodes = if image.node_id_rows.len() == image.graph.node_ids.len() {
            image.node_id_rows.len()
        } else {
            0
        };
        let shared_edges = if image.edge_id_rows.len() == image.graph.edge_ids.len() {
            image.edge_id_rows.len()
        } else {
            0
        };
        self.bytes().saturating_sub(
            shared_nodes
                .saturating_add(shared_edges)
                .saturating_mul(512),
        )
    }

    fn set(&mut self, key: u128, fixed: usize, payload: usize) {
        // Removing a value does not remove its column's null/validity pages. Keep the
        // affected lane allowance while releasing the previous variable payload estimate.
        // Dense row bits branch first; the logical owner domains remain in retirement events.
        let key = key.reverse_bits();
        let previous = self.owners.get(key).copied().unwrap_or_default();
        let next = HostOwner {
            fixed: fixed.max(previous.fixed),
            payload,
        };
        self.owners.insert_cow(key, next);
        self.private_bytes = self
            .private_bytes
            .saturating_sub(previous.bytes())
            .saturating_add(next.bytes());
    }

    fn group_key(key: u128, dense: u64) -> u128 {
        (key & !u128::from(u64::MAX)) | (1_u128 << 119) | u128::from(dense / 256)
    }

    fn apply(
        &mut self,
        delta: &ResidentProjectDelta,
        schema: (usize, usize),
        previous: &CandleResident,
        resident: &CandleResident,
        retain_history: bool,
    ) -> Vec<(u128, usize)> {
        let mut changed = Vec::new();
        // Every retirement event describes the pre-publication generation, including when
        // several rows in this batch create the same new group.
        let prior_owners = retain_history.then(|| self.owners.clone());
        // Appending dense value pages can still replace existing page-directory ancestors.
        // These bounded directory paths are separate from the exact identity-map changes.
        for (domain, changed_rows, old_rows, lanes) in [
            (
                1_u128,
                delta.graph.nodes.len(),
                previous.node_count(),
                schema.0 + 1,
            ),
            (
                2,
                delta.graph.edges.len(),
                previous.edge_count(),
                schema.1 + 1,
            ),
            (3, delta.graph.outgoing.len(), previous.node_count(), 1),
            (4, delta.graph.incoming.len(), previous.node_count(), 1),
        ] {
            if retain_history && changed_rows != 0 {
                changed.push((
                    (domain << 120) | (1_u128 << 118),
                    if old_rows != 0 { lanes * 64 * 1024 } else { 0 },
                ));
            }
        }
        for vector in delta.vectors.iter().filter(|_| retain_history) {
            let property = match vector {
                crate::graph::ResolvedVectorMutation::Upsert { property, .. }
                | crate::graph::ResolvedVectorMutation::Remove { property, .. } => *property,
            };
            changed.push((
                (5_u128 << 120) | (1_u128 << 118) | (u128::from(property.0) << 64),
                if previous.vector_row_count(property) == 0 {
                    0
                } else {
                    64 * 1024
                },
            ));
        }
        let mut changed_groups = BTreeMap::<u128, usize>::new();
        let mut set = |this: &mut Self,
                       key: u128,
                       dense: u64,
                       fixed: usize,
                       payload: usize,
                       existing: (bool, bool)| {
            // A 256-row group shares its column-page/path allowance. Payload and owner
            // metadata remain individually replaceable; ingest must not allocate one
            // full host-page allowance for every row.
            let group = Self::group_key(key, dense);
            match changed_groups.entry(group) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    if let Some(prior_owners) = &prior_owners {
                        let prior_group = prior_owners
                            .get(group.reverse_bits())
                            .copied()
                            .unwrap_or(HostOwner {
                                fixed: if existing.1 { fixed } else { 0 },
                                payload: 0,
                            });
                        changed.push((group, prior_group.bytes()));
                    }
                    this.set(group, fixed, 0);
                    entry.insert(fixed);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) if fixed > *entry.get() => {
                    entry.insert(fixed);
                    this.set(group, fixed, 0);
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
            if let Some(prior_owners) = &prior_owners {
                changed.push((
                    key,
                    prior_owners
                        .get(key.reverse_bits())
                        .copied()
                        .map_or(if existing.0 { 512 } else { 0 }, HostOwner::bytes),
                ));
            }
            this.set(key, 512, payload);
        };
        for node in &delta.graph.nodes {
            let bytes = node
                .properties
                .iter()
                .fold(node.labels.len() * 8, |bytes, (_, value)| {
                    bytes.saturating_add(scalar_host_bytes(value))
                });
            set(
                self,
                (1_u128 << 120) | u128::from(node.dense),
                u64::from(node.dense),
                (schema.0.max(node.properties.len()) + 1) * 64 * 1024,
                bytes,
                (
                    (node.dense as usize) < previous.node_count(),
                    (node.dense as usize / 256) * 256 < previous.node_count(),
                ),
            );
        }
        for edge in &delta.graph.edges {
            let bytes = edge.properties.iter().fold(0_usize, |bytes, (_, value)| {
                bytes.saturating_add(scalar_host_bytes(value))
            });
            set(
                self,
                (2_u128 << 120) | u128::from(edge.dense),
                u64::from(edge.dense),
                (schema.1.max(edge.properties.len()) + 1) * 64 * 1024,
                bytes,
                (
                    (edge.dense as usize) < previous.edge_count(),
                    (edge.dense as usize / 256) * 256 < previous.edge_count(),
                ),
            );
        }
        for (domain, rows) in [
            (3_u128, &delta.graph.outgoing),
            (4_u128, &delta.graph.incoming),
        ] {
            for row in rows {
                set(
                    self,
                    (domain << 120) | u128::from(row.dense),
                    u64::from(row.dense),
                    64 * 1024,
                    (row.neighbors.len() + row.edges.len()) * 4,
                    (
                        (row.dense as usize) < previous.node_count(),
                        (row.dense as usize / 256) * 256 < previous.node_count(),
                    ),
                );
            }
        }
        for vector in &delta.vectors {
            let (property, entity, payload) = match vector {
                crate::graph::ResolvedVectorMutation::Upsert {
                    property,
                    entity_id,
                    coordinates,
                    ..
                } => (*property, *entity_id, coordinates.len() * 2),
                crate::graph::ResolvedVectorMutation::Remove {
                    property,
                    entity_id,
                    ..
                } => (*property, *entity_id, 0),
            };
            let dense = u64::from(resident.vector_dense_row(property, entity).unwrap_or(0));
            set(
                self,
                (5_u128 << 120) | (u128::from(property.0) << 64) | u128::from(entity),
                dense,
                128 * 1024,
                payload,
                (
                    previous.vector_dense_row(property, entity).is_some(),
                    (dense as usize / 256) * 256 < previous.vector_row_count(property),
                ),
            );
        }
        for sample in &delta.temporal {
            let key = (7_u128 << 120)
                | ((sample.entity_kind as u128) << 112)
                | (u128::from(sample.sample.property.0) << 64)
                | u128::from(sample.target);
            let existing = (
                self.cold_temporal_owners.get(key).is_some(),
                self.cold_temporal_owners
                    .get(Self::group_key(key, sample.target))
                    .is_some(),
            );
            set(self, key, sample.target, 64 * 1024, 0, existing);
            self.set(
                (6_u128 << 120) | u128::from(self.history_entries),
                1024,
                scalar_host_bytes(&sample.sample.value),
            );
            self.history_entries += 1;
        }
        changed
    }
}

fn host_delta_retirement(
    current: &CandleResident,
    host: Option<&HostFootprint>,
    delta: &ResidentProjectDelta,
) -> usize {
    let mut bytes = delta.staging_bytes().saturating_mul(2);
    if let Some(backing) = current.shared_graph_backing() {
        for (domain, rows, adjacency) in [
            (3_u128, &delta.graph.outgoing, &backing.outgoing),
            (4_u128, &delta.graph.incoming, &backing.incoming),
        ] {
            for row in rows {
                let cold = adjacency.row(row.dense).map_or(0, |row| row.len() * 8);
                let prior = host
                    .and_then(|host| {
                        host.owners
                            .get(((domain << 120) | u128::from(row.dense)).reverse_bits())
                    })
                    .copied()
                    .map_or(0, HostOwner::bytes);
                bytes = bytes.saturating_add(cold.max(prior));
            }
        }
        for (rows, columns) in [
            (
                delta
                    .graph
                    .nodes
                    .iter()
                    .map(|node| node.dense)
                    .collect::<Vec<_>>(),
                &backing.node_properties,
            ),
            (
                delta
                    .graph
                    .edges
                    .iter()
                    .map(|edge| edge.dense)
                    .collect::<Vec<_>>(),
                &backing.edge_properties,
            ),
        ] {
            for row in rows {
                bytes = bytes.saturating_add(64 * 1024);
                for property in columns.property_ids() {
                    if let Some(value) = columns.get(row, property) {
                        bytes = bytes.saturating_add(64 * 1024 + scalar_host_bytes(&value));
                    }
                }
            }
        }
    }
    bytes.saturating_add(
        (delta.graph.outgoing.len()
            + delta.graph.incoming.len()
            + delta.vectors.len()
            + delta.temporal.len())
        .saturating_mul(128 * 1024),
    )
}

fn identity_path_changes(
    previous: &CandleResident,
    staged: &CandleResident,
    delta: &ResidentProjectDelta,
) -> Vec<([u64; 6], usize)> {
    let project = delta.project.0.as_u128();
    let mut changes = Vec::new();
    let mut maps = BTreeSet::new();
    for node in &delta.graph.nodes {
        if node.dense as usize >= previous.node_count() {
            maps.insert((0_u8, PropertyId(0)));
        }
    }
    for edge in &delta.graph.edges {
        if edge.dense as usize >= previous.edge_count() {
            maps.insert((1, PropertyId(0)));
        }
    }
    for mutation in &delta.vectors {
        let (property, entity) = match mutation {
            crate::graph::ResolvedVectorMutation::Upsert {
                property,
                entity_id,
                ..
            }
            | crate::graph::ResolvedVectorMutation::Remove {
                property,
                entity_id,
                ..
            } => (*property, *entity_id),
        };
        if previous.vector_dense_row(property, entity).is_none()
            && staged.vector_dense_row(property, entity).is_some()
        {
            maps.insert((2, property));
        }
    }
    for (domain, property) in maps {
        changes.extend(
            staged
                .identity_map_changes(previous, domain, property)
                .into_iter()
                .map(|(prefix, depth, bytes, _)| {
                    (
                        [
                            3 | (u64::from(domain) << 8) | (u64::from(depth) << 16),
                            (project >> 64) as u64,
                            project as u64,
                            property.0,
                            (prefix >> 64) as u64,
                            prefix as u64,
                        ],
                        bytes,
                    )
                }),
        );
    }
    changes
}

const METAL_KERNEL_SOURCES: [(&str, &str); 6] = [
    (
        "operators.metal",
        include_str!("../../../kernels/metal/operators.metal"),
    ),
    (
        "graph_paths.metal",
        include_str!("../../../kernels/metal/graph_paths.metal"),
    ),
    (
        "graph_components_metrics.metal",
        include_str!("../../../kernels/metal/graph_components_metrics.metal"),
    ),
    (
        "graph_louvain.metal",
        include_str!("../../../kernels/metal/graph_louvain.metal"),
    ),
    (
        "dictionary_order.rs",
        include_str!("accelerator/dictionary_order.rs"),
    ),
    (
        "temporal_order.rs",
        include_str!("accelerator/temporal_order.rs"),
    ),
];

fn metal_kernel_sources_hash(sources: &[(&str, &str)]) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    for (name, source) in sources {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&(source.len() as u64).to_le_bytes());
        hasher.update(source.as_bytes());
    }
    hasher.finalize()
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalBackend")
            .field("project_count", &self.resident.len())
            .field("resident_revision", &self.resident_revision())
            .finish_non_exhaustive()
    }
}

/// Incremental peak introduced by device-authored mutation effects: canonical membership/value
/// lanes, one effect bit per attempted intent, and the additional sealed descriptor/receipt for
/// every command. The ordinary pipeline and continuation reservations account for the pre-effect
/// packet; this delta keeps admission honest after widening that packet.
fn metal_mutation_effect_scratch_bytes(request: &ResidentNodePipelineRequest) -> Result<usize> {
    let Some(program) = &request.mutation else {
        return Ok(0);
    };
    let overflow = || {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "Metal mutation effect scratch accounting overflow",
        )
    };
    let static_membership_lanes = program
        .commands
        .iter()
        .try_fold(0_usize, |total, command| {
            let lanes = match &command.operation {
                super::ResidentMutationOperation::AddLabels { labels, .. }
                | super::ResidentMutationOperation::RemoveLabels { labels, .. } => labels.len(),
                super::ResidentMutationOperation::RemoveProperty { .. } => 1,
                _ => 0,
            };
            total.checked_add(lanes).ok_or_else(overflow)
        })?;
    let static_effect_lanes = if static_membership_lanes == 0 {
        0
    } else {
        static_membership_lanes
            .checked_add(1)
            .ok_or_else(overflow)?
    };
    // Typed SET widened each target from identity-only (three lanes) to identity plus the
    // canonical destination cell (three additional lanes).
    let typed_current_lanes = program
        .commands
        .iter()
        .filter(|command| {
            matches!(
                command.operation,
                super::ResidentMutationOperation::SetProperty { .. }
            )
        })
        .count()
        .checked_mul(3)
        .ok_or_else(overflow)?;
    let canonical_words = request
        .max_output_rows
        .checked_mul(static_effect_lanes.max(typed_current_lanes))
        .ok_or_else(overflow)?;
    let intent_words = program.max_intents;
    // Nine output words, six typed descriptor words, and one static command word per effect
    // obligation. Taking all three is conservative across the mutually exclusive packet paths.
    let receipt_words = program
        .commands
        .len()
        .checked_mul(16)
        .ok_or_else(overflow)?;
    canonical_words
        .checked_add(intent_words)
        .and_then(|words| words.checked_add(receipt_words))
        .and_then(|words| words.checked_mul(mem::size_of::<u64>()))
        .and_then(|bytes| bytes.checked_add(64))
        .ok_or_else(overflow)
}

impl MetalBackend {
    pub fn new(
        device_ordinal: usize,
        memory_limit_bytes: usize,
        reserved_bytes: usize,
    ) -> Result<Self> {
        Self::with_governor(
            device_ordinal,
            DeviceMemoryGovernor::new(memory_limit_bytes, reserved_bytes),
        )
    }

    pub fn with_governor(
        device_ordinal: usize,
        mut governor: DeviceMemoryGovernor,
    ) -> Result<Self> {
        let ordinal = u32::try_from(device_ordinal).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "Metal device ordinal exceeds u32",
            )
        })?;
        let device = construct_metal_device(ordinal)
            .map_err(|error| Error::new(ErrorCode::GpuAdmissionFailure, error))?;
        let recommended = device
            .as_metal_device()
            .map_err(candle_error)?
            .metal_device()
            .recommended_max_working_set_size();
        if recommended > 0 {
            governor.cap_limit(recommended)?;
        }
        // The cap above is computed once, here, from what the DEVICE can hold. It is correct at this
        // instant and never revisited, so a process that starts on an idle machine keeps the whole
        // budget after other tenants have taken the memory. Measured on a 48 GB host: 37.4 GB
        // admitted against 19.3 GB actually free, with swap 91% used — 1.9x over-admission, which is
        // more than enough to have the process killed while its own ledger still says there is room.
        //
        // The probe closes that by bounding each admission's INCREASE against free memory at the
        // moment of admission. It lives here rather than in the execution crate because reading host
        // memory is platform-specific and that crate is not.
        governor.set_host_available_probe(std::sync::Arc::new(host_available_bytes));
        // The device's recommended working set is the largest amount of memory it wants in use, so
        // it is also the natural ceiling for one command's scratch buffer: a big-memory GPU is
        // allowed a proportionally big command instead of a fixed sub-hardware constant.
        let max_buffer_length = usize::try_from(recommended).unwrap_or(usize::MAX);
        prepare_metal_sort_kernels(&device)?;
        prepare_metal_graph_algorithm_kernels(&device)?;
        let gate = crate::metal_gate::metal_device_gate(&device).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "Metal backend constructed a non-Metal Candle device",
            )
        })?;
        Ok(Self {
            device,
            gate,
            governor,
            resident: BTreeMap::new(),
            max_buffer_length,
            private_pages: None,
            host: BTreeMap::new(),
            private_host: HostFootprint::default(),
        })
    }

    /// Hash of every backend-specific Metal source library shipped for tuned deployments.
    ///
    /// Keep the library name and byte length in the transcript so moving code between source
    /// units, adding a new unit, or changing any graph kernel invalidates the same production
    /// fingerprint used by deployment/cache compatibility checks.
    #[must_use]
    pub fn kernel_source_hash() -> blake3::Hash {
        metal_kernel_sources_hash(&METAL_KERNEL_SOURCES)
    }

    /// Builds deterministic flat IVF-PQ pages with device-side assignment and bounded admitted
    /// scratch. The returned pages remain derived and are published by the normal index lifecycle.
    pub fn build_ivf_pq(&self, source: &VectorIndex, config: IvfPqConfig) -> Result<IvfPqIndex> {
        self.build_ivf_pq_cancellable(source, config, &CancellationToken::new())
    }

    fn build_ivf_pq_cancellable(
        &self,
        source: &VectorIndex,
        config: IvfPqConfig,
        cancellation: &CancellationToken,
    ) -> Result<IvfPqIndex> {
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        let plan = IvfPqBuildPlan::for_shape(source.len() as u64, source.dimension(), config)?;
        let scratch = usize::try_from(plan.peak_scratch_bytes).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "IVF-PQ scratch estimate exceeds this process address space",
            )
        })?;
        let derived = usize::try_from(plan.derived_bytes).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "IVF-PQ derived pages exceed this process address space",
            )
        })?;
        let _derived = self.governor.reserve_staging(derived)?;
        let _reservation = self.governor.reserve_scratch(scratch)?;
        let distance_tile = plan.assignment_tile_bytes.max(size_of::<f32>());
        let mut kernel = CandleIvfPqBuildKernel::new(&self.device, distance_tile)?;
        IvfPqIndex::build_with_kernel_cancellable(source, config, &mut kernel, cancellation)
    }

    fn project(&self, project: ProjectId) -> Result<&CandleResident> {
        self.resident.get(&project).map(Arc::as_ref).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!("project {project} has no complete Metal-resident image"),
            )
        })
    }

    /// Holds this backend's Candle device for one whole operation, readback included.
    ///
    /// Candle sweeps and frees its buffer pools inside the readback (`flush_and_wait_current` ->
    /// `drop_unused_buffers`), so a guard that ends before the readback does not serialise the
    /// part that actually races. The gate is reentrant: a gated method may call another.
    fn gate(&self) -> ReentrantMutexGuard<'static, ()> {
        self.gate.lock()
    }

    fn total_after(&self, project: ProjectId, replacement_bytes: usize) -> usize {
        self.resident
            .iter()
            .filter(|(id, _)| **id != project)
            .map(|(id, resident)| {
                resident
                    .allocated_bytes
                    .saturating_add(self.host.get(id).map_or(0, HostFootprint::bytes))
            })
            .fold(replacement_bytes, usize::saturating_add)
    }
}

impl ExecutionBackend for MetalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn build_ivf_pq(
        &self,
        source: &VectorIndex,
        config: IvfPqConfig,
        cancellation: &CancellationToken,
    ) -> Result<IvfPqIndex> {
        self.build_ivf_pq_cancellable(source, config, cancellation)
    }

    fn available_query_scratch_bytes(&self) -> usize {
        self.governor.available_scratch_bytes()
    }

    fn reserve_query_scratch(&self, bytes: usize) -> Result<super::ScratchReservation> {
        self.governor.reserve_scratch(bytes)
    }

    fn resident_project_bytes(&self, project: ProjectId) -> Option<usize> {
        // This interface reports the device image. Independent host allocations are
        // still included in governor admission and retained-generation accounting.
        self.resident
            .get(&project)
            .map(|resident| resident.allocated_bytes)
    }

    fn shared_project_backing(&self, project: ProjectId) -> Option<super::ResidentSharedBacking> {
        let resident = self.resident.get(&project)?;
        Some(super::ResidentSharedBacking {
            graph: resident.shared_graph_backing()?,
            temporal: resident.shared_temporal_backings(),
            vectors: resident.shared_vector_backings(),
        })
    }

    fn pin_project(&self, project: ProjectId) -> Result<Box<dyn ExecutionBackend>> {
        let resident = self.resident.get(&project).cloned().ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!("project {project} has no complete Metal-resident image"),
            )
        })?;
        let private_pages = if self.governor.is_shared_branch() {
            self.private_pages.clone()
        } else {
            Some(BranchPages::default())
        };
        let governor = self
            .governor
            .pin_shared_generation(self.governor.current_generation()?)?;
        let private_host = if self.governor.is_shared_branch() {
            self.private_host.clone()
        } else {
            HostFootprint::branch(&resident)
        };
        Ok(Box::new(Self {
            device: self.device.clone(),
            gate: self.gate,
            governor,
            resident: BTreeMap::from([(project, resident)]),
            max_buffer_length: self.max_buffer_length,
            private_pages,
            host: self
                .host
                .get(&project)
                .cloned()
                .map(|host| BTreeMap::from([(project, host)]))
                .unwrap_or_default(),
            private_host,
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        let _gate = self.gate();
        let host = HostFootprint::cold(&image);
        let existing_host_bytes = host.bytes().saturating_sub(host.cold_staging_bytes(&image));
        let planned =
            CandleResident::planned_bytes(&image).saturating_add(host.cold_staging_bytes(&image));
        let staging = self.governor.reserve_staging(planned)?;
        let project = image.project;
        let staged = CandleResident::upload(image, &self.device)?;
        let total = self.total_after(project, staged.allocated_bytes.saturating_add(host.bytes()));
        self.device.synchronize().map_err(candle_error)?;
        let mut governor = self.governor.clone();
        governor.publish_staging_with_existing_allocations(
            staging,
            staged.allocated_bytes,
            total,
            existing_host_bytes,
            || {
                let old = self.resident.insert(project, Arc::new(staged));
                drop(old);
                Ok(())
            },
        )?;
        self.governor = governor;
        self.private_pages = None;
        self.private_host = HostFootprint::default();
        self.host.insert(project, host);
        Ok(())
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        let _gate = self.gate();
        let planned = images.iter().try_fold(0_usize, |total, image| {
            total
                .checked_add(
                    CandleResident::planned_bytes(image)
                        .saturating_add(HostFootprint::cold(image).cold_staging_bytes(image)),
                )
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "replacement Metal project bytes overflow",
                    )
                })
        })?;
        let staging = self.governor.reserve_staging(planned)?;
        let mut replacement = BTreeMap::new();
        let mut host = BTreeMap::new();
        let mut shared_host_bytes = 0_usize;
        for image in images {
            let project = image.project;
            let footprint = HostFootprint::cold(&image);
            shared_host_bytes = shared_host_bytes.saturating_add(
                footprint
                    .bytes()
                    .saturating_sub(footprint.cold_staging_bytes(&image)),
            );
            host.insert(project, footprint);
            let resident = CandleResident::upload(image, &self.device)?;
            if replacement.insert(project, Arc::new(resident)).is_some() {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "snapshot contains a duplicate project ID",
                ));
            }
        }
        let actual = replacement
            .iter()
            .try_fold(0_usize, |total, (project, resident)| {
                total
                    .checked_add(
                        resident
                            .allocated_bytes
                            .saturating_add(host[project].bytes()),
                    )
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "replacement Metal project bytes overflow",
                        )
                    })
            })?;
        let mut governor = self.governor.clone();
        self.device.synchronize().map_err(candle_error)?;
        governor.publish_staging_with_existing_allocations(
            staging,
            actual.saturating_sub(shared_host_bytes),
            actual,
            shared_host_bytes,
            || {
                let old = mem::replace(&mut self.resident, replacement);
                drop(old);
                Ok(())
            },
        )?;
        self.governor = governor;
        self.private_pages = None;
        self.private_host = HostFootprint::default();
        self.host = host;
        Ok(())
    }

    fn apply_project_delta(&mut self, delta: ResidentProjectDelta) -> Result<()> {
        let _gate = self.gate();
        let project = delta.project;
        let current = self.resident.get(&project).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!("project {project} has no complete Metal-resident image"),
            )
        })?;
        let retain_history =
            !self.governor.is_shared_branch() && self.governor.has_live_shared_pins()?;
        let host_complete = current.delta_publish_is_host_complete(&delta);
        let retired_host = host_delta_retirement(current, self.host.get(&project), &delta);
        let (planned_bytes, plan) = current.prepare_delta_staging(&delta)?;
        let staged_bytes = planned_bytes.saturating_add(retired_host);
        let reservation = self.governor.reserve_staging(staged_bytes)?;
        let mut host = self.host.get(&project).cloned().unwrap_or_default();
        let mut private_host = self.private_host.clone();
        let capture = Capture::begin()?;
        let staged = current.stage_delta_prepared(&delta, &self.device, Some(plan))?;
        let mut pages = capture.finish()?;
        let schema = staged.shared_graph_backing().map_or((0, 0), |backing| {
            (
                backing.node_properties.property_ids().count(),
                backing.edge_properties.property_ids().count(),
            )
        });
        let shared_host_owners = private_host.owners.shared_with(&host.owners)
            && private_host.private_bytes == host.private_bytes
            && private_host.history_entries == host.history_entries;
        let host_changes = host.apply(&delta, schema, current, &staged, retain_history);
        host.update_dictionaries(&staged);
        if self.governor.is_shared_branch() {
            if shared_host_owners {
                // The owner delta is identical. Share its immutable index root while
                // retaining this branch's separate cold and dictionary allowances.
                private_host.owners = host.owners.clone();
                private_host.private_bytes = host.private_bytes;
                private_host.history_entries = host.history_entries;
            } else {
                private_host.apply(&delta, schema, current, &staged, false);
            }
            private_host.update_dictionaries(&staged);
        }
        let mut private_pages = self.private_pages.clone();
        if let Some(branch) = &mut private_pages {
            branch.apply(&pages);
        }
        let total = self.total_after(project, staged.allocated_bytes.saturating_add(host.bytes()));
        if !host_complete {
            self.device.synchronize().map_err(candle_error)?;
        }
        let mut governor = self.governor.clone();
        // Branches charge private_pages/private_host, so retirement records are
        // needed only when the root publishes a new generation for pinned readers.
        let (births, mut changes) = if retain_history {
            pages.retirement_events()
        } else {
            (Vec::new(), Vec::new())
        };
        if retain_history {
            changes.extend(identity_path_changes(current, &staged, &delta));
            let project_key = project.0.as_u128();
            if let Some(previous_host) = self.host.get(&project) {
                changes.extend(
                    host.owners
                        .changed_path_nodes(&previous_host.owners)
                        .into_iter()
                        .map(|(prefix, depth, old_bytes, _)| {
                            (
                                [
                                    4,
                                    (project_key >> 64) as u64,
                                    project_key as u64,
                                    u64::from(depth),
                                    (prefix >> 64) as u64,
                                    prefix as u64,
                                ],
                                old_bytes,
                            )
                        }),
                );
            }
            changes.extend(host_changes.into_iter().map(|(key, bytes)| {
                (
                    [
                        2,
                        (project_key >> 64) as u64,
                        project_key as u64,
                        (key >> 64) as u64,
                        0,
                        key as u64,
                    ],
                    bytes,
                )
            }));
        }
        governor.publish_shared_page_staging(
            reservation,
            staged_bytes,
            total,
            &births,
            &changes,
            private_pages
                .as_ref()
                .map(|pages| pages.bytes().saturating_add(private_host.bytes())),
            || {
                let old = self.resident.insert(project, Arc::new(staged));
                drop(old);
                Ok(())
            },
        )?;
        pages.commit();
        self.governor = governor;
        self.private_pages = private_pages;
        self.private_host = private_host;
        self.host.insert(project, host);
        Ok(())
    }

    #[allow(unsafe_code)]
    unsafe fn apply_project_delta_exclusive(&mut self, delta: ResidentProjectDelta) -> Result<()> {
        let _gate = self.gate();
        if self.governor.has_live_shared_pins()?
            || self
                .resident
                .get(&delta.project)
                .is_none_or(|resident| Arc::strong_count(resident) != 1)
        {
            return self.apply_project_delta(delta);
        }
        self.device.synchronize().map_err(candle_error)?;
        let _exclusive = ExclusiveWrite::begin();
        self.apply_project_delta(delta)
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        let _gate = self.gate();
        let total = self
            .resident
            .iter()
            .filter(|(resident_project, _)| **resident_project != project)
            .map(|(id, resident)| {
                resident
                    .allocated_bytes
                    .saturating_add(self.host.get(id).map_or(0, HostFootprint::bytes))
            })
            .fold(0_usize, usize::saturating_add);
        let staging = self.governor.reserve_staging(0)?;
        let mut governor = self.governor.clone();
        self.device.synchronize().map_err(candle_error)?;
        governor.publish_staging(staging, 0, total, || {
            let old = self.resident.remove(&project);
            drop(old);
            Ok(())
        })?;
        self.governor = governor;
        self.private_pages = None;
        self.private_host = HostFootprint::default();
        self.host.remove(&project);
        Ok(())
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        for resident in self.resident.values_mut() {
            let resident = Arc::make_mut(resident);
            resident.bookmark = bookmark;
        }
    }

    fn resident_revision(&self) -> Option<u64> {
        self.resident
            .values()
            .map(|resident| resident.revision)
            .max()
    }

    fn resident_graph_revision(&self, project: ProjectId) -> Option<u64> {
        self.resident
            .get(&project)
            .map(|resident| resident.revision)
    }

    fn resident_bookmark(&self, project: ProjectId) -> Option<Bookmark> {
        self.resident
            .get(&project)
            .map(|resident| resident.bookmark)
    }

    fn scan_nodes(
        &self,
        project: ProjectId,
        label: Option<LabelId>,
        layers: LayerMask,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        let _gate = self.gate();
        self.project(project)?
            .scan_nodes(label, layers, cancellation)
    }

    fn filter_node_i64(
        &self,
        project: ProjectId,
        property: PropertyId,
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        let _gate = self.gate();
        self.project(project)?
            .filter_node_i64(property, operation, operand, cancellation)
    }

    fn expand_project_out(
        &self,
        project: ProjectId,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let _gate = self.gate();
        self.project(project)?
            .expand_out(&self.device, sources, cancellation)
    }

    fn expand_project_in(
        &self,
        project: ProjectId,
        targets: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let _gate = self.gate();
        self.project(project)?
            .expand_in(&self.device, targets, cancellation)
    }

    fn expand_project_out_bounded(
        &self,
        project: ProjectId,
        sources: &[u32],
        maximum_rows: usize,
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let _gate = self.gate();
        self.project(project)?
            .expand_out_bounded(sources, maximum_rows, cancellation)
    }

    fn expand_project_in_bounded(
        &self,
        project: ProjectId,
        targets: &[u32],
        maximum_rows: usize,
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let _gate = self.gate();
        self.project(project)?
            .expand_in_bounded(targets, maximum_rows, cancellation)
    }

    fn search_vectors(
        &self,
        request: &ResidentVectorQuery,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        let _gate = self.gate();
        let resident = self.project(request.project)?;
        let input_rows = resident
            .node_count()
            .checked_add(resident.vector_row_count(request.property))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal vector-pipeline scratch row count overflow",
                )
            })?;
        let output_rows = request
            .limit
            .checked_mul(request.query_count)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal vector result shape overflow",
                )
            })?;
        let _scratch = self.governor.reserve_scratch(vector_query_scratch_bytes(
            input_rows,
            resident.vector_dimension(request.property),
            request.query_count,
            output_rows,
        )?)?;
        resident.search_vectors(&self.device, request, cancellation)
    }

    fn intersect_sorted_node_sets(
        &self,
        request: &ResidentMultiwayIntersectionRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        let _gate = self.gate();
        self.project(request.project)?;
        let input_rows = request.sorted_sets.iter().try_fold(0_usize, |total, set| {
            total.checked_add(set.len()).ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal multiway-intersection input size overflow",
                )
            })
        })?;
        let maximum_output = request
            .sorted_sets
            .iter()
            .map(Vec::len)
            .min()
            .unwrap_or(0)
            .min(request.max_output_rows.saturating_add(1));
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            160,
            maximum_output,
            24,
        )?)?;
        intersect_sorted_node_sets(&self.device, request, cancellation)
    }

    fn sort_rows(
        &self,
        request: &ResidentSortRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        let _gate = self.gate();
        request.validate()?;
        let _scratch = self
            .governor
            .reserve_scratch(metal_sort_scratch_bytes(request)?)?;
        self.project(request.project)?
            .sort_rows(&self.device, request, cancellation)
    }

    fn join_node_i64(
        &self,
        request: &ResidentJoinRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        let _gate = self.gate();
        let input_rows = request
            .left_rows
            .len()
            .checked_add(request.right_rows.len())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal join scratch accounting overflow",
                )
            })?;
        let maximum_output = request
            .left_rows
            .len()
            .checked_mul(request.right_rows.len())
            .map_or(request.max_pairs, |pairs| pairs.min(request.max_pairs));
        let maximum_output = maximum_output.min(u32::MAX as usize - 1);
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            256,
            maximum_output,
            64,
        )?)?;
        self.project(request.project)?
            .join_node_i64(&self.device, request, cancellation)
    }

    fn group_node_i64(
        &self,
        request: &ResidentGroupRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        let _gate = self.gate();
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            request.rows.len(),
            384,
            request.rows.len(),
            96,
        )?)?;
        self.project(request.project)?
            .group_node_i64(&self.device, request, cancellation)
    }

    fn execute_node_pipeline(
        &self,
        request: &ResidentNodePipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        let _gate = self.gate();
        let resident = self.project(request.project)?;
        let relationship_count = request.relationship_column_count();
        let maximum_rows =
            request.maximum_pipeline_rows(resident.node_count(), resident.edge_count())?;
        let projected_width = request
            .integer_projections
            .len()
            .saturating_mul(size_of::<i64>() + size_of::<u8>())
            .saturating_add(request.property_null_projections.len())
            .saturating_add(
                relationship_count
                    .saturating_mul(2)
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(size_of::<u32>());
        let input_rows = resident
            .node_count()
            .checked_add(maximum_rows)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal resident-pipeline scratch row count overflow",
                )
            })?;
        let pipeline_scratch =
            operator_scratch_bytes(input_rows, 160, maximum_rows, projected_width)?;
        let effect_scratch = metal_mutation_effect_scratch_bytes(request)?;
        let segmented_value_scratch = request
            .mutation
            .as_ref()
            .and_then(|program| program.continuation.as_ref())
            .and_then(|continuation| continuation.segmented_value_program.as_ref())
            .map_or(Ok(0), |program| {
                prepare_metal_segmented_aggregation(resident, program)
                    .map(|prepared| prepared.scratch_bytes())
            })?;
        let unit_value_scratch = request
            .mutation
            .as_ref()
            .and_then(|program| program.continuation.as_ref())
            .and_then(|continuation| continuation.value_program.as_ref())
            .map_or(Ok(0), |program| {
                metal_quantifier_program_scratch_bytes(resident, program)
            })?;
        let scratch_bytes = pipeline_scratch
            .checked_add(request.mutation_continuation_scratch_bytes()?)
            .and_then(|bytes| bytes.checked_add(effect_scratch))
            .and_then(|bytes| bytes.checked_add(segmented_value_scratch))
            .and_then(|bytes| bytes.checked_add(unit_value_scratch))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal mutation/effect scratch reservation overflow",
                )
            })?;
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        resident.execute_node_pipeline(&self.device, request, cancellation)
    }

    fn execute_row_program(
        &self,
        request: &ResidentRowProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        request.validate()?;
        let resident = self.project(request.project)?;
        let maximum_rows = request
            .input
            .maximum_pipeline_rows(resident.node_count(), resident.edge_count())?;
        let admitted_rows = maximum_rows.min(request.input.max_output_rows);
        let relationship_count = request.input.relationship_column_count();
        let projected_width = relationship_count
            .checked_mul(2)
            .and_then(|columns| columns.checked_mul(size_of::<u32>()))
            .and_then(|bytes| bytes.checked_add(size_of::<u32>()))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal typed-row wrapped-pipeline width overflow",
                )
            })?;
        let pipeline_input_rows =
            resident
                .node_count()
                .checked_add(maximum_rows)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "Metal typed-row wrapped-pipeline row count overflow",
                    )
                })?;
        let pipeline_scratch =
            operator_scratch_bytes(pipeline_input_rows, 160, maximum_rows, projected_width)?;
        let row_scratch = metal_resident_row_program_scratch_bytes(request, admitted_rows)?;
        let combined_scratch = pipeline_scratch.checked_add(row_scratch).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Metal typed-row combined scratch accounting overflow",
            )
        })?;
        let mut scratch = self.governor.reserve_scratch(combined_scratch)?;
        resident.execute_row_program(&self.device, request, &mut scratch, cancellation)
    }

    fn supports_native_quantifier_program(&self) -> bool {
        true
    }

    fn supports_native_quantifier_entity_source(&self) -> bool {
        true
    }

    fn execute_quantifier_program(
        &self,
        request: &ResidentQuantifierProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentQuantifierProgramResult> {
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        request.validate()?;
        let resident = self.project(request.generation.project)?;
        if resident.bookmark != request.generation.bookmark
            || resident.revision != request.generation.graph_revision
            || resident.layout_version != request.generation.layout_version
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier program expected a different Metal bookmark, graph revision, or layout version",
            ));
        }
        let scratch_bytes = metal_quantifier_program_scratch_bytes(resident, request)?;
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        execute_metal_quantifier_program(&self.device, resident, request, cancellation)
    }

    fn supports_nullable_relation_existing_relationship(&self) -> bool {
        true
    }

    fn supports_nullable_relation_relationship_endpoint_seed(&self) -> bool {
        true
    }

    fn supports_nullable_relation_predicates(&self) -> bool {
        true
    }

    fn supports_nullable_relation_scope_limit(&self) -> bool {
        true
    }

    fn supports_nullable_relation_string_property_equality(&self) -> bool {
        true
    }

    fn execute_nullable_relation(
        &self,
        request: &ResidentNullableRelationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNullableRelationResult> {
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        request.validate()?;
        let resident = self.project(request.generation.project)?;
        let scratch_bytes = resident.metal_nullable_relation_scratch_bytes(request)?;
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        resident.execute_nullable_relation(&self.device, request, cancellation)
    }

    fn execute_delete_pipeline(
        &self,
        request: &ResidentDeleteRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentDeleteResult> {
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        request.validate()?;
        let resident = self.project(request.project)?;
        let scratch_bytes = resident.metal_delete_scratch_bytes(request)?;
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        resident.execute_delete_pipeline_metal(&self.device, request, cancellation)
    }

    fn execute_row_mutation(
        &self,
        request: &ResidentRowMutationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        request.validate()?;
        let resident = self.project(request.generation.project)?;
        let scratch_bytes = resident.metal_row_mutation_scratch_bytes(request)?;
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        resident.execute_row_mutation_metal(&self.device, request, cancellation)
    }

    fn execute_pattern_predicate(
        &self,
        request: &ResidentPatternPredicateRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentPatternPredicateResult> {
        let _gate = self.gate();
        let resident = self.project(request.project)?;
        let _scratch = self
            .governor
            .reserve_scratch(resident.metal_pattern_predicate_scratch_bytes(request)?)?;
        resident.execute_pattern_predicate(&self.device, request, cancellation)
    }

    fn execute_pattern_count(
        &self,
        request: &ResidentPatternCountRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentPatternCountResult> {
        let _gate = self.gate();
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        self.project(request.project)?
            .execute_pattern_count(&self.device, request, cancellation)
    }

    fn execute_pattern_predicate_pairs(
        &self,
        request: &ResidentPatternPairPredicateRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentPatternPairPredicateResult> {
        let _gate = self.gate();
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        self.project(request.project)?
            .execute_pattern_predicate_pairs(&self.device, request, cancellation)
    }

    fn supports_native_variable_path(&self) -> bool {
        true
    }

    fn execute_variable_path(
        &self,
        request: &ResidentVariablePathRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVariablePathResult> {
        let _gate = self.gate();
        request.validate()?;
        // This is where a real Metal command buffer is built, so the fixed command-buffer scratch
        // ABI ceiling applies here (and only here — host executors have no such bound).
        request.enforce_native_scratch_abi(self.max_buffer_length)?;
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_variable_path(
            &self.device,
            self.project(request.project)?,
            request,
            cancellation,
        )
    }

    fn supports_native_to_boolean(&self) -> bool {
        true
    }

    fn execute_to_boolean(
        &self,
        request: &ResidentToBooleanRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentToBooleanResult> {
        let _gate = self.gate();
        let scratch = request.inputs.iter().try_fold(0_usize, |bytes, input| {
            let payload = match input {
                ResidentScalarInput::String(value) => value.len(),
                ResidentScalarInput::Null
                | ResidentScalarInput::Boolean(_)
                | ResidentScalarInput::Other => 0,
            };
            bytes
                .checked_add(payload.saturating_add(size_of::<u32>() + 2))
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "Metal toBoolean scratch accounting overflow",
                    )
                })
        })?;
        let _scratch = self.governor.reserve_scratch(scratch)?;
        execute_metal_to_boolean(&self.device, request, cancellation)
    }

    fn supports_native_create_node(&self) -> bool {
        true
    }

    fn execute_create_node(
        &self,
        request: &ResidentCreateNodeRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentCreateNodeResult> {
        let _gate = self.gate();
        request.validate()?;
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_create_node(
            &self.device,
            self.project(request.project)?,
            request,
            cancellation,
        )
    }

    fn supports_native_temporal_value_program(&self) -> bool {
        true
    }

    fn supports_native_temporal_arithmetic_program(&self) -> bool {
        true
    }

    fn execute_temporal_arithmetic_program(
        &self,
        request: &ResidentTemporalArithmeticRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalArithmeticResult> {
        let _gate = self.gate();
        request.validate()?;
        ensure_not_cancelled(cancellation)?;
        let _scratch = self
            .governor
            .reserve_scratch(metal_temporal_arithmetic_scratch_bytes(request)?)?;
        execute_metal_temporal_arithmetic_program(
            &self.device,
            self.project(request.generation.project)?,
            request,
            cancellation,
        )
    }

    fn execute_temporal_value_program(
        &self,
        request: &ResidentTemporalValueProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalValueProgramResult> {
        let _gate = self.gate();
        request.validate()?;
        let prepared = prepare_metal_temporal_value_program(request)?;
        let _scratch = self.governor.reserve_scratch(prepared.scratch_bytes())?;
        execute_metal_temporal_value_program(&self.device, request, &prepared, cancellation)
    }

    fn supports_native_boolean_program(&self) -> bool {
        true
    }

    fn execute_boolean_program(
        &self,
        request: &ResidentBooleanProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentBooleanProgramResult> {
        let _gate = self.gate();
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_boolean_program(&self.device, request, cancellation)
    }

    fn supports_native_scalar_program(&self) -> bool {
        true
    }

    fn execute_scalar_program(
        &self,
        request: &ResidentScalarProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentScalarProgramResult> {
        let _gate = self.gate();
        request.validate()?;
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_scalar_program(&self.device, request, cancellation)
    }

    fn supports_native_boolean_aggregate_program(&self) -> bool {
        true
    }

    fn execute_boolean_aggregate_program(
        &self,
        request: &ResidentBooleanAggregateProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentBooleanProgramResult> {
        let _gate = self.gate();
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_boolean_aggregate_program(&self.device, request, cancellation)
    }

    fn supports_native_segmented_aggregation(&self) -> bool {
        true
    }

    fn execute_segmented_aggregation(
        &self,
        request: &ResidentSegmentedAggregationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSegmentedAggregationResult> {
        let _gate = self.gate();
        request.validate()?;
        let resident = self.project(request.project)?;
        let prepared = prepare_metal_segmented_aggregation(resident, request)?;
        let _scratch = self.governor.reserve_scratch(prepared.scratch_bytes())?;
        execute_metal_segmented_aggregation(
            &self.device,
            resident,
            request,
            &prepared,
            cancellation,
        )
    }

    fn supports_native_procedure_table(&self) -> bool {
        true
    }

    fn execute_procedure_table(
        &self,
        request: &ResidentProcedureTableRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentProcedureTableResult> {
        let _gate = self.gate();
        request.validate()?;
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_procedure_table(&self.device, request, cancellation)
    }

    fn supports_native_range_program(&self) -> bool {
        true
    }

    fn execute_range_program(
        &self,
        request: &ResidentRangeProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRangeProgramResult> {
        let _gate = self.gate();
        request.validate()?;
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_range_program(&self.device, request, cancellation)
    }

    fn execute_node_group_pipeline(
        &self,
        request: &ResidentNodeGroupPipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        let _gate = self.gate();
        let resident = self.project(request.input.project)?;
        let input_rows = resident
            .node_count()
            .checked_add(
                request
                    .input
                    .expansion
                    .as_ref()
                    .map_or(0, |_| resident.edge_count()),
            )
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal resident-group scratch row count overflow",
                )
            })?;
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            448,
            request.max_groups.min(input_rows),
            96,
        )?)?;
        resident.execute_node_group_pipeline(&self.device, request, cancellation)
    }

    fn execute_node_group_pipelines(
        &self,
        first: &ResidentNodeGroupPipelineRequest,
        additional: &[ResidentNodeGroupPipelineRequest],
        cancellation: &CancellationToken,
    ) -> Result<Vec<Vec<ResidentGroup>>> {
        let _gate = self.gate();
        let resident = self.project(first.input.project)?;
        let input_rows = resident
            .node_count()
            .checked_add(
                first
                    .input
                    .expansion
                    .as_ref()
                    .map_or(0, |_| resident.edge_count()),
            )
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal resident-group scratch row count overflow",
                )
            })?;
        let maximum_groups = std::iter::once(first)
            .chain(additional)
            .map(|request| request.max_groups)
            .max()
            .unwrap_or(0)
            .min(input_rows);
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            448,
            maximum_groups,
            96,
        )?)?;
        resident.execute_node_group_pipelines(&self.device, first, additional, cancellation)
    }

    fn execute_temporal_pipeline(
        &self,
        request: &ResidentTemporalPipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalPipelineResult> {
        let _gate = self.gate();
        let resident = self.project(request.input.project)?;
        let input_rows = resident
            .node_count()
            .checked_add(
                request
                    .input
                    .expansion
                    .as_ref()
                    .map_or(0, |_| resident.edge_count()),
            )
            .and_then(|rows| {
                rows.checked_add(resident.temporal_row_count(request.target, request.property))
            })
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal resident-temporal scratch row count overflow",
                )
            })?;
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            416,
            request.max_output_rows.min(input_rows),
            64,
        )?)?;
        resident.execute_temporal_pipeline(&self.device, request, cancellation)
    }

    fn execute_graph_procedure(
        &self,
        request: &ResidentGraphProcedureRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentGraphProcedureResult> {
        let _gate = self.gate();
        ensure_graph_execution(cancellation, request.deadline)?;
        let resident = self.project(request.project)?;
        let scratch_bytes = match request.procedure {
            ResidentGraphProcedure::Degree => metal_degree_scratch_bytes(resident.node_count())?,
            ResidentGraphProcedure::BreadthFirst { .. } => {
                metal_breadth_first_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::DepthFirst { .. } => {
                metal_depth_first_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::ShortestPath { .. } => {
                metal_shortest_path_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::DijkstraWeighted { .. } => {
                metal_weighted_dijkstra_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::DijkstraUnit { .. } => {
                metal_unit_dijkstra_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::Louvain => {
                metal_louvain_scratch_bytes(resident.node_count(), resident.edge_count())?
            }
            ResidentGraphProcedure::PageRank { .. } => {
                metal_pagerank_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::WeaklyConnectedComponents => {
                metal_wcc_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::StronglyConnectedComponents => {
                metal_scc_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::TriangleCount
            | ResidentGraphProcedure::ClusteringCoefficient => {
                metal_graph_metrics_scratch_bytes(resident.node_count(), resident.edge_count())?
            }
            ResidentGraphProcedure::KCore => {
                metal_kcore_scratch_bytes(resident.node_count(), resident.edge_count())?
            }
        };
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        resident.execute_graph_procedure(&self.device, request, cancellation)
    }

    fn filter_i64(
        &self,
        values: &[i64],
        validity: &[bool],
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        let _gate = self.gate();
        if values.len() != validity.len() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "filter value and validity lengths differ",
            ));
        }
        if values.is_empty() {
            return Ok(Vec::new());
        }
        ensure_not_cancelled(cancellation)?;
        let _scratch = self.governor.reserve_scratch(
            values
                .len()
                .saturating_mul(size_of::<i64>() + size_of::<u8>() * 2),
        )?;
        let values =
            Tensor::from_slice(values, values.len(), &self.device).map_err(candle_error)?;
        let validity = Tensor::from_slice(
            &validity
                .iter()
                .map(|valid| u8::from(*valid))
                .collect::<Vec<_>>(),
            validity.len(),
            &self.device,
        )
        .map_err(candle_error)?;
        filter_tensor(&values, &validity, operation, operand, cancellation)
    }

    fn expand_out(
        &self,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let _gate = self.gate();
        self.expand_project_out(ProjectId(uuid::Uuid::nil()), sources, cancellation)
    }

    fn exact_l2(
        &self,
        matrix: &[f32],
        rows: usize,
        dimension: usize,
        query: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        let _gate = self.gate();
        validate_matrix(matrix, rows, dimension, query)?;
        ensure_not_cancelled(cancellation)?;
        let _scratch = self.governor.reserve_scratch(
            matrix
                .len()
                .saturating_add(query.len())
                .saturating_add(rows)
                .saturating_mul(size_of::<f32>()),
        )?;
        let distances = exact_l2(&self.device, matrix, rows, dimension, query)?;
        ensure_not_cancelled(cancellation)?;
        Ok(DistanceBatch { distances })
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use crate::graph::{IvfPqConfig, Similarity, VectorIndex};

    use super::*;

    #[test]
    fn host_batch_groups_charge_once_and_keep_retained_accounting() -> Result<()> {
        use crate::graph::{GraphStore, NodeInput};
        use crate::{Layer, NodeId};
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        let mut graph = GraphStore::default();
        let value = graph.catalog_mut().intern_property("value")?;
        let body = graph.catalog_mut().intern_property("body")?;
        let dirty: Arc<str> = "dirty body ".repeat(32_768).into();
        for row in 0..64_u64 {
            graph.insert_node(NodeInput {
                id: NodeId(row + 1),
                layer: Layer::Observed,
                revision: 1,
                labels: vec![],
                properties: vec![
                    (value, ScalarValue::Integer(row as i64)),
                    (body, ScalarValue::String(dirty.clone())),
                ],
            })?;
        }
        let image = ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?));
        let mut host = HostFootprint::cold(&image);
        let pinned = host.clone();
        let original = CandleResident::upload(image, &device)?;
        for row in 64..320_u64 {
            graph.insert_node(NodeInput {
                id: NodeId(row + 1),
                layer: Layer::Observed,
                revision: 2,
                labels: vec![],
                properties: vec![
                    (value, ScalarValue::Integer(row as i64)),
                    (body, ScalarValue::String(dirty.clone())),
                ],
            })?;
        }
        let delta = ResidentProjectDelta {
            project: original.project,
            bookmark: Bookmark { term: 1, index: 2 },
            graph: graph.device_delta(2)?,
            temporal: vec![],
            vectors: vec![],
            invalidate_derived: false,
        };
        let (_, plan) = original.prepare_delta_staging(&delta)?;
        let staged = original.stage_delta_prepared(&delta, &device, Some(plan))?;
        let mut without_history = host.clone();
        assert!(
            without_history
                .apply(&delta, (2, 0), &original, &staged, false)
                .is_empty()
        );
        let events = host.apply(&delta, (2, 0), &original, &staged, true);
        assert_eq!(without_history.private_bytes, host.private_bytes);
        assert_eq!(
            without_history
                .owners
                .iter()
                .map(|(key, value)| (key, value.fixed, value.payload))
                .collect::<Vec<_>>(),
            host.owners
                .iter()
                .map(|(key, value)| (key, value.fixed, value.payload))
                .collect::<Vec<_>>()
        );
        let groups = events
            .iter()
            .filter(|(key, _)| key & (1_u128 << 119) != 0)
            .collect::<Vec<_>>();
        assert_eq!(groups.len(), 2, "one retirement record per changed group");
        let expected = 2 * 3 * 64 * 1024 + 256 * (512 + 32 + dirty.len());
        assert_eq!(host.private_bytes, expected);
        assert_eq!(pinned.private_bytes, 0);
        assert_eq!(pinned.owners.len(), 0);
        let events = host.apply(&delta, (2, 0), &original, &staged, true);
        assert_eq!(
            host.private_bytes, expected,
            "repeated replacement cannot grow its charge"
        );
        assert_eq!(
            events
                .iter()
                .filter(|(key, bytes)| key & (1_u128 << 119) != 0 && *bytes == 3 * 64 * 1024)
                .count(),
            2
        );
        Ok(())
    }

    #[test]
    #[allow(unsafe_code)]
    fn exclusive_publication_preserves_older_pins_and_updates_unpinned_cells() -> Result<()> {
        use crate::graph::{GraphStore, NodeInput, TypedColumn};
        use crate::{Layer, NodeId};
        let _guard = crate::metal_test_guard();
        let mut graph = GraphStore::default();
        let value = graph.catalog_mut().intern_property("value")?;
        let mirror = graph.catalog_mut().intern_property("mirror")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![],
            properties: vec![
                (value, ScalarValue::Integer(7)),
                (mirror, ScalarValue::Integer(11)),
            ],
        })?;
        let image = ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?));
        let project = image.project;
        let host = HostFootprint::cold(&image);
        assert_eq!(host.bytes() - host.cold_staging_bytes(&image), 512);
        let mut missing = image.clone();
        missing.node_id_rows = PersistentMap::default();
        assert_eq!(host.cold_staging_bytes(&missing), host.bytes());
        let mut backend = MetalBackend::new(0, 64 * 1024 * 1024, 0)?;
        backend.admit_project(image)?;
        let original = backend.pin_project(project)?;
        graph.set_node_property(NodeId(1), value, ScalarValue::Integer(19), 2)?;
        backend.apply_project_delta(ResidentProjectDelta {
            project,
            bookmark: Bookmark { term: 1, index: 2 },
            graph: graph.device_delta(2)?,
            vectors: vec![],
            temporal: vec![],
            invalidate_derived: true,
        })?;
        assert_eq!(Arc::strong_count(&backend.resident[&project]), 1);
        graph.set_node_property(NodeId(1), mirror, ScalarValue::Integer(23), 3)?;
        // SAFETY: this fixture has no concurrent host accesses. The backend must still reject
        // in-place writes when an older execution generation owns unchanged native lanes.
        unsafe {
            backend.apply_project_delta_exclusive(ResidentProjectDelta {
                project,
                bookmark: Bookmark { term: 1, index: 3 },
                graph: graph.device_delta(3)?,
                vectors: vec![],
                temporal: vec![],
                invalidate_derived: true,
            })?;
        }
        let original_backing = original
            .shared_project_backing(project)
            .ok_or_else(|| Error::internal("missing pinned backing"))?;
        let read = |backing: &crate::graph::GraphSharedBacking, property| {
            let Some(TypedColumn::Integer { values, .. }) =
                backing.node_properties.column(property)
            else {
                panic!("integer fixture")
            };
            values[0]
        };
        assert_eq!(read(&original_backing.graph, value), 7);
        assert_eq!(read(&original_backing.graph, mirror), 11);
        drop(original_backing);
        drop(original);
        assert!(!backend.governor.has_live_shared_pins()?);
        graph.set_node_property(NodeId(1), mirror, ScalarValue::Integer(31), 4)?;
        // SAFETY: every external host view and execution pin was dropped above.
        unsafe {
            backend.apply_project_delta_exclusive(ResidentProjectDelta {
                project,
                bookmark: Bookmark { term: 1, index: 4 },
                graph: graph.device_delta(4)?,
                vectors: vec![],
                temporal: vec![],
                invalidate_derived: true,
            })?;
        }
        let after = backend
            .shared_project_backing(project)
            .ok_or_else(|| Error::internal("missing published backing"))?;
        assert_eq!(read(&after.graph, mirror), 31);
        drop(after);
        // A reader born after an unpinned publication retains that new value,
        // even though the earlier write needed no retirement history.
        let later = backend.pin_project(project)?;
        graph.set_node_property(NodeId(1), mirror, ScalarValue::Integer(47), 5)?;
        // SAFETY: the older execution pin forces the safe staged publication path.
        unsafe {
            backend.apply_project_delta_exclusive(ResidentProjectDelta {
                project,
                bookmark: Bookmark { term: 1, index: 5 },
                graph: graph.device_delta(5)?,
                vectors: vec![],
                temporal: vec![],
                invalidate_derived: true,
            })?;
        }
        let later_backing = later.shared_project_backing(project).unwrap();
        assert_eq!(read(&later_backing.graph, mirror), 31);
        assert!(backend.governor.snapshot().pinned_generation_bytes > 0);
        Ok(())
    }

    #[test]
    fn surgical_host_births_exclude_pre_ingest_readers() -> Result<()> {
        use crate::graph::{GraphStore, NodeInput};
        use crate::{Layer, NodeId};
        let _guard = crate::metal_test_guard();
        let mut graph = GraphStore::default();
        let property = graph.catalog_mut().intern_property("value")?;
        let id = |row: u64| {
            NodeId(
                row.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(1_u64 << 63),
            )
        };
        for row in 0..255_u64 {
            graph.insert_node(NodeInput {
                id: id(row),
                layer: Layer::Observed,
                revision: 1,
                labels: vec![],
                properties: vec![(property, ScalarValue::Integer(row as i64))],
            })?;
        }
        let project = ProjectId(uuid::Uuid::nil());
        let mut backend = MetalBackend::new(0, 256 * 1024 * 1024, 0)?;
        backend.admit_project(ResidentProjectImage::graph_only(Arc::new(
            graph.snapshot()?,
        )))?;
        let previous = backend.resident[&project].clone();
        let mut host = backend.host[&project].clone();
        for row in 255..768_u64 {
            graph.insert_node(NodeInput {
                id: id(row),
                layer: Layer::Observed,
                revision: 2,
                labels: vec![],
                properties: vec![(property, ScalarValue::Integer(row as i64))],
            })?;
        }
        let mut delta = ResidentProjectDelta {
            project,
            bookmark: Bookmark { term: 1, index: 2 },
            graph: graph.device_delta(2)?,
            vectors: vec![],
            temporal: vec![],
            invalidate_derived: true,
        };
        backend.apply_project_delta(delta.clone())?;
        let grown = backend.resident[&project].clone();
        let paths = grown.identity_map_changes(&previous, 0, PropertyId(0));
        assert!(
            paths.iter().filter(|(_, _, old, _)| *old > 0).count() > 8,
            "random high IDs retire multiple old lookup leaves"
        );
        let path_events = identity_path_changes(&previous, &grown, &delta);
        assert_eq!(path_events.len(), paths.len());
        let expected_old = paths.iter().map(|(_, _, old, _)| old).sum::<usize>();
        let mut lookup_governor = DeviceMemoryGovernor::new(16 * 1024 * 1024, 0);
        let before_lookup_birth = lookup_governor.pin_shared_generation(0)?;
        let reservation = lookup_governor.reserve_staging(0)?;
        lookup_governor.publish_shared_page_staging(
            reservation,
            0,
            0,
            &[],
            &path_events,
            None,
            || Ok(()),
        )?;
        assert_eq!(
            lookup_governor.snapshot().pinned_generation_bytes,
            expected_old
        );
        let born_paths = paths
            .iter()
            .filter(|(_, _, old, new)| *old == 0 && *new != 0)
            .map(|(prefix, depth, _, new)| {
                (
                    [
                        3 | (u64::from(*depth) << 16),
                        0,
                        0,
                        0,
                        (prefix >> 64) as u64,
                        *prefix as u64,
                    ],
                    *new,
                )
            })
            .collect::<Vec<_>>();
        assert!(
            !born_paths.is_empty(),
            "split sibling births must be recorded"
        );
        let reservation = lookup_governor.reserve_staging(0)?;
        lookup_governor.publish_shared_page_staging(
            reservation,
            0,
            0,
            &[],
            &born_paths,
            None,
            || Ok(()),
        )?;
        assert_eq!(
            lookup_governor.snapshot().pinned_generation_bytes,
            expected_old
        );
        let after_lookup_birth =
            lookup_governor.pin_shared_generation(lookup_governor.current_generation()?)?;
        let reservation = lookup_governor.reserve_staging(0)?;
        lookup_governor.publish_shared_page_staging(
            reservation,
            0,
            0,
            &[],
            &born_paths,
            None,
            || Ok(()),
        )?;
        assert_eq!(
            lookup_governor.snapshot().pinned_generation_bytes,
            expected_old + born_paths.iter().map(|(_, bytes)| bytes).sum::<usize>()
        );
        drop(after_lookup_birth);
        assert_eq!(
            lookup_governor.snapshot().pinned_generation_bytes,
            expected_old
        );
        drop(before_lookup_birth);
        assert_eq!(lookup_governor.snapshot().pinned_generation_bytes, 0);
        let changes = host.apply(&delta, (1, 0), &previous, &grown, true);
        let new_node_changes = |changes: Vec<(u128, usize)>| {
            changes
                .into_iter()
                .filter(|(key, _)| key >> 120 == 1 && key & (1_u128 << 118) == 0)
                .map(|(key, bytes)| ([2, 0, 0, (key >> 64) as u64, 0, key as u64], bytes))
                .collect::<Vec<_>>()
        };
        let changes = new_node_changes(changes);
        let old_group = [
            2,
            0,
            0,
            ((1_u128 << 120 | 1_u128 << 119) >> 64) as u64,
            0,
            0,
        ];
        assert!(
            changes
                .iter()
                .any(|(key, bytes)| *key == old_group && *bytes > 0),
            "a partially filled cold page keeps its old metadata bound"
        );
        let births = changes
            .into_iter()
            .filter(|(key, _)| *key != old_group)
            .collect::<Vec<_>>();
        assert_eq!(births.len(), 513 + 2, "one birth per row and new group");
        assert_eq!(
            births
                .iter()
                .map(|(key, _)| key)
                .collect::<BTreeSet<_>>()
                .len(),
            births.len()
        );
        assert!(
            births.iter().all(|(_, bytes)| *bytes == 0),
            "every row and both new page groups are born in this batch"
        );
        let mut governor = DeviceMemoryGovernor::new(16 * 1024 * 1024, 0);
        let oldest = governor.pin_shared_generation(0)?;
        let reservation = governor.reserve_staging(0)?;
        governor.publish_shared_page_staging(reservation, 0, 0, &[], &births, None, || Ok(()))?;
        graph.set_node_property(id(256), property, ScalarValue::Integer(-3), 3)?;
        delta.graph = graph.device_delta(3)?;
        delta.bookmark.index = 3;
        let replaced = new_node_changes(host.apply(&delta, (1, 0), &grown, &grown, true));
        assert!(replaced.iter().all(|(_, bytes)| *bytes > 0));
        let reservation = governor.reserve_staging(0)?;
        governor.publish_shared_page_staging(reservation, 0, 0, &[], &replaced, None, || Ok(()))?;
        assert_eq!(governor.snapshot().pinned_generation_bytes, 0);
        let later = governor.pin_shared_generation(governor.current_generation()?)?;
        let reservation = governor.reserve_staging(0)?;
        governor.publish_shared_page_staging(reservation, 0, 0, &[], &replaced, None, || Ok(()))?;
        assert!(governor.snapshot().pinned_generation_bytes > 0);
        drop(later);
        assert_eq!(governor.snapshot().pinned_generation_bytes, 0);
        drop(oldest);
        Ok(())
    }

    #[test]
    fn surgical_metal_pins_share_pages_and_branch_churn_plateaus() -> Result<()> {
        use crate::graph::{GraphStore, NodeInput};
        use crate::{Layer, NodeId};

        let _guard = crate::metal_test_guard();
        let mut graph = GraphStore::default();
        let property = graph.catalog_mut().intern_property("value")?;
        for row in 0_u64..8192 {
            graph.insert_node(NodeInput {
                id: NodeId(row + 1),
                layer: Layer::Observed,
                revision: 1,
                labels: vec![],
                properties: vec![(
                    property,
                    ScalarValue::Integer((row as i64).wrapping_mul(7919)),
                )],
            })?;
        }
        let project = ProjectId(uuid::Uuid::nil());
        let mut backend = MetalBackend::new(0, 256 * 1024 * 1024, 0)?;
        backend.admit_project(ResidentProjectImage::graph_only(Arc::new(
            graph.snapshot()?,
        )))?;
        let observer = backend.governor.clone();
        let reader = backend.pin_project(project)?;
        let mut branch = backend.pin_project(project)?;
        assert_eq!(observer.snapshot().pinned_generation_bytes, 0);
        let delta = |graph: &mut GraphStore, revision| -> Result<ResidentProjectDelta> {
            graph.set_node_property(
                NodeId(1),
                property,
                ScalarValue::Integer(-(revision as i64)),
                revision,
            )?;
            Ok(ResidentProjectDelta {
                project,
                bookmark: Bookmark {
                    term: 1,
                    index: revision,
                },
                graph: graph.device_delta(revision)?,
                temporal: vec![],
                vectors: vec![],
                invalidate_derived: true,
            })
        };
        branch.apply_project_delta(delta(&mut graph, 2)?)?;
        let private = observer.snapshot().pinned_generation_bytes;
        assert!(
            private > 0 && private < 1024 * 1024,
            "branch unexpectedly retained {private} bytes"
        );
        let old_branch = branch.pin_project(project)?;
        for revision in 3..35 {
            branch.apply_project_delta(delta(&mut graph, revision)?)?;
            assert_eq!(observer.snapshot().pinned_generation_bytes, 2 * private);
        }
        let cancellation = CancellationToken::new();
        assert_eq!(
            reader.filter_node_i64(project, property, CompareOp::Eq, 0, &cancellation)?,
            vec![0]
        );
        assert_eq!(
            old_branch.filter_node_i64(project, property, CompareOp::Eq, -2, &cancellation)?,
            vec![0]
        );
        assert_eq!(
            branch.filter_node_i64(project, property, CompareOp::Eq, -34, &cancellation)?,
            vec![0]
        );
        drop(old_branch);
        assert_eq!(observer.snapshot().pinned_generation_bytes, private);
        let root_delta = delta(&mut graph, 35)?;
        backend.apply_project_delta(root_delta)?;
        assert!(observer.snapshot().pinned_generation_bytes > private);
        let root_retained = observer.snapshot().pinned_generation_bytes;
        // A branch pinned after root writes has a different private-owner baseline.
        // It must charge its own delta rather than inherit the root's private allowance.
        let mut late_branch = backend.pin_project(project)?;
        late_branch.apply_project_delta(delta(&mut graph, 36)?)?;
        assert_eq!(
            late_branch.filter_node_i64(project, property, CompareOp::Eq, -36, &cancellation)?,
            vec![0]
        );
        assert_eq!(
            backend.filter_node_i64(project, property, CompareOp::Eq, -35, &cancellation)?,
            vec![0]
        );
        assert!(observer.snapshot().pinned_generation_bytes > root_retained);
        drop(late_branch);
        assert_eq!(observer.snapshot().pinned_generation_bytes, root_retained);
        for revision in 36..68 {
            backend.apply_project_delta(delta(&mut graph, revision)?)?;
            assert_eq!(observer.snapshot().pinned_generation_bytes, root_retained);
        }
        let intermediate = backend.pin_project(project)?;
        backend.apply_project_delta(delta(&mut graph, 68)?)?;
        let with_intermediate = observer.snapshot().pinned_generation_bytes;
        assert!(with_intermediate > root_retained);
        for revision in 69..85 {
            backend.apply_project_delta(delta(&mut graph, revision)?)?;
            assert_eq!(
                observer.snapshot().pinned_generation_bytes,
                with_intermediate
            );
        }
        assert_eq!(
            intermediate.filter_node_i64(project, property, CompareOp::Eq, -67, &cancellation)?,
            vec![0]
        );
        drop(intermediate);
        assert_eq!(observer.snapshot().pinned_generation_bytes, root_retained);
        assert_eq!(
            reader.filter_node_i64(project, property, CompareOp::Eq, 0, &cancellation)?,
            vec![0]
        );
        let prior = branch.resident_bookmark(project);
        observer.cap_limit(observer.snapshot().admitted_bytes())?;
        assert!(branch.apply_project_delta(delta(&mut graph, 85)?).is_err());
        assert_eq!(branch.resident_bookmark(project), prior);
        assert_eq!(
            branch.filter_node_i64(project, property, CompareOp::Eq, -34, &cancellation)?,
            vec![0]
        );
        drop(branch);
        drop(reader);
        assert_eq!(observer.snapshot().pinned_generation_bytes, 0);
        eprintln!(
            "surgical Metal: dirty_rows=8192 branch_private_bytes={private}; repeated writes plateau"
        );
        Ok(())
    }

    #[test]
    fn production_kernel_fingerprint_covers_every_metal_library() {
        assert_eq!(METAL_KERNEL_SOURCES.len(), 6);
        assert_eq!(
            MetalBackend::kernel_source_hash(),
            metal_kernel_sources_hash(&METAL_KERNEL_SOURCES)
        );
        for changed in 0..METAL_KERNEL_SOURCES.len() {
            let mut sources = METAL_KERNEL_SOURCES;
            sources[changed].1 = "deliberately changed source";
            assert_ne!(
                MetalBackend::kernel_source_hash(),
                metal_kernel_sources_hash(&sources),
                "source library {changed} is absent from the production fingerprint"
            );
        }
    }

    #[test]
    #[ignore = "requires an available physical Metal device"]
    fn metal_builds_flat_ivf_pq_pages_with_bounded_device_assignment() -> Result<()> {
        let mut vectors = VectorIndex::new(4, Similarity::Euclidean)?;
        for row in 0..64_u64 {
            let value = row as f32 / 64.0;
            vectors.upsert(row + 1, &[value, 1.0 - value, value * value, 0.5], row + 1)?;
        }
        let backend = MetalBackend::new(0, 256 * 1024 * 1024, 8 * 1024 * 1024)?;
        let index = backend.build_ivf_pq(
            &vectors,
            IvfPqConfig {
                size_class_version: crate::graph::IVF_PQ_SIZE_CLASS_VERSION,
                coarse_centroids: 4,
                subquantizers: 2,
                bits_per_code: 4,
                probes: 2,
                candidate_budget: 32,
                iterations: 3,
                seed: 7,
            },
        )?;
        let hits = index.search(&vectors, &[0.5, 0.5, 0.25, 0.5], 4)?;
        assert_eq!(hits.len(), 4);
        Ok(())
    }
}

/// Bytes the host still has free, or `None` when it cannot be determined.
///
/// Free + inactive + speculative: inactive and speculative pages are reclaimable on demand, so
/// counting only `free` would understate what is available by most of the file cache and refuse
/// work that would have fit.
///
/// **The page size is read, never assumed.** Apple Silicon uses 16 KiB pages where x86 used 4 KiB,
/// and hardcoding 4096 here understates availability by exactly 4x — which is precisely the error
/// that produced a wrong headline number while this was being characterised.
///
/// `None` on any failure. The caller treats that as "no opinion" and admits as it would without a
/// probe: a memory check that cannot read memory must not become a new way to refuse work.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(unsafe_code)]
fn host_available_bytes() -> Option<usize> {
    unsafe extern "C" {
        static mach_task_self_: libc::mach_port_t;
        fn mach_host_self() -> libc::mach_port_t;
        fn mach_port_deallocate(task: libc::mach_port_t, name: libc::mach_port_t) -> i32;
    }
    // SAFETY: the kernel writes at most `count` integer words into this initialized
    // ABI-sized structure. Each acquired host send right is released before returning.
    let (status, count, stats, page_size) = unsafe {
        let mut stats: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        let host = mach_host_self();
        let status = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&mut stats as *mut libc::vm_statistics64).cast(),
            &mut count,
        );
        mach_port_deallocate(mach_task_self_, host);
        (status, count, stats, libc::vm_page_size)
    };
    // Mach's free_count already includes speculative pages; vm_stat subtracts them
    // from its "Pages free" display. Require the three leading counters we use.
    (status == libc::KERN_SUCCESS && count >= 3 && page_size > 0).then(|| {
        (stats.free_count as usize)
            .saturating_add(stats.inactive_count as usize)
            .saturating_mul(page_size)
    })
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn host_available_bytes() -> Option<usize> {
    None
}

#[cfg(test)]
mod host_memory_tests {
    #[test]
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn the_host_probe_reads_a_believable_number_from_the_real_machine() {
        // This exists because the bug it guards against already happened: the page size was assumed
        // to be 4096 while Apple Silicon uses 16384, understating availability by exactly 4x and
        // producing a wrong headline figure. An assumed-too-LARGE page size is detectable — it makes
        // "available" exceed physical RAM — so bound it against the real hardware size rather than
        // against a constant.
        let physical = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .and_then(|text| text.trim().parse::<usize>().ok())
            .expect("hw.memsize");
        let available = super::host_available_bytes().expect("probe should read this machine");
        assert!(
            available > 0 && available <= physical,
            "probe reported {available} bytes available on a machine with {physical} bytes of RAM"
        );
        // Not an assertion — the value itself, so a human can compare it against `vm_stat` when this
        // is being trusted for the first time on new hardware.
        println!(
            "host_available_bytes = {available} ({:.1} GB) of {:.1} GB physical",
            available as f64 / 2f64.powi(30),
            physical as f64 / 2f64.powi(30)
        );
    }
}
