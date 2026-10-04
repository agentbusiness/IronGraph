//! Persistent derived row directories and page-local publication.
use super::*;

#[derive(Clone, Default)]
pub(super) struct StaleRows {
    positions: PersistentMap<usize>,
    rows: PagedVec<u32>,
    len: usize,
    pending: BTreeMap<usize, u32>,
}

impl StaleRows {
    #[cfg(test)]
    pub(super) fn new() -> Self {
        Self::default()
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = &u32> {
        self.rows.iter().take(self.len)
    }
    pub(super) fn insert(&mut self, row: u32) -> bool {
        if self.positions.get(u128::from(row)).is_some() {
            return false;
        }
        let position = self.len;
        if position == self.rows.len() {
            self.rows.push(row);
        } else {
            self.rows[position] = row;
        }
        self.positions.insert_cow(u128::from(row), position);
        self.len += 1;
        self.pending.insert(position, row);
        true
    }
    pub(super) fn remove(&mut self, row: &u32) -> bool {
        let Some(position) = self.positions.remove(u128::from(*row)) else {
            return false;
        };
        self.len -= 1;
        self.pending.remove(&self.len);
        if position != self.len {
            let last = self.rows[self.len];
            self.rows[position] = last;
            self.positions.insert_cow(u128::from(last), position);
            self.pending.insert(position, last);
        }
        true
    }
    #[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
    fn publish(&mut self, tensor: &mut Option<Tensor>, device: &Device) -> Result<()> {
        if self.len == 0 {
            if let Some(previous) = tensor.as_ref() {
                metal_pages::retire(previous)?;
            }
            *tensor = None;
            self.pending.clear();
            return Ok(());
        }
        let updates = self
            .pending
            .iter()
            .filter(|(row, _)| **row < self.len)
            .map(|(row, value)| (*row, *value))
            .collect::<Vec<_>>();
        let source = match tensor.as_ref() {
            Some(tensor) => tensor.clone(),
            None => metal_pages::zeros(DType::U32, 0, device)?,
        };
        let len = source.elem_count().max(self.len);
        let extended = metal_pages::extend(&source, &updates, len, device)?;
        *tensor = Some(extended.narrow(0, 0, self.len).map_err(candle_error)?);
        self.pending.clear();
        Ok(())
    }
}

impl FromIterator<u32> for StaleRows {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        let mut rows = Self::default();
        for row in iter {
            rows.insert(row);
        }
        rows.pending.clear();
        rows
    }
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
pub(super) fn staging_bytes(
    resident: &CandleResident,
    delta: &ResidentProjectDelta,
) -> Result<usize> {
    let page = metal_pages::page_bytes();
    let mut bytes = 0_usize;
    for mutation in &delta.vectors {
        let property = match mutation {
            ResolvedVectorMutation::Upsert { property, .. }
            | ResolvedVectorMutation::Remove { property, .. } => property,
        };
        let column = resident.vectors.get(property).ok_or_else(|| {
            Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "vector staging targets an absent column",
            )
        })?;
        bytes = bytes
            .saturating_add(16 * page)
            .saturating_add(column.dimension.saturating_mul(48));
    }
    bytes = bytes.saturating_add(
        (delta.graph.nodes.len() + delta.graph.edges.len())
            .saturating_mul(resident.vectors.len())
            .saturating_mul(4 * page),
    );
    let mut additions = BTreeMap::<(u8, u64, PropertyId), Vec<TemporalSample>>::new();
    for mutation in &delta.temporal {
        additions
            .entry((
                mutation.entity_kind as u8,
                mutation.target,
                mutation.sample.property,
            ))
            .or_default()
            .push(mutation.sample.clone());
    }
    for (key, samples) in additions {
        let previous = resident
            .shared_temporal
            .iter()
            .find(|backing| (backing.entity_kind as u8, backing.target, backing.property) == key)
            .ok_or_else(|| Error::internal("temporal staging targets an absent column"))?;
        let mut after = previous.clone();
        after.append_samples(&samples)?;
        let (_, property_bytes) = property_delta::changed_rows(
            &previous.values,
            &after.values,
            previous.len()..after.len(),
        )?;
        bytes = bytes
            .saturating_add(property_bytes)
            .saturating_add(12 * page)
            .saturating_add(samples.len().saturating_mul(256));
        let dictionary = resident
            .temporal_properties
            .get(&key)
            .and_then(|bundle| bundle.dictionary.as_ref())
            .ok_or_else(|| Error::internal("temporal staging dictionary absent"))?;
        if let Some(plan) = property_delta::DictionaryAppend::plan(dictionary, &after.values)? {
            bytes = bytes.saturating_add(plan.staging_bytes);
        }
        if let Some(order) = resident
            .temporal_properties
            .get(&key)
            .and_then(|bundle| bundle.temporal_metadata.as_ref())
            .and_then(|metadata| metadata.order.as_ref())
        {
            bytes =
                bytes.saturating_add(order.prepare_append(previous.len(), &after)?.staging_bytes);
        }
    }
    Ok(bytes)
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
fn swap_node_properties(
    resident: &mut CandleResident,
    bundle: &mut UploadedPropertyColumns,
) -> Result<()> {
    let dictionary = bundle
        .dictionary
        .as_mut()
        .ok_or_else(|| Error::internal("temporal property dictionary is absent"))?;
    std::mem::swap(&mut resident.node_string_dictionary, dictionary);
    macro_rules! swap { ($($resident:ident => $bundle:ident),+) => { $(std::mem::swap(&mut resident.$resident, &mut bundle.$bundle);)+ }; }
    swap!(integer_nodes=>integer,boolean_nodes=>boolean,float_nodes=>float,string_nodes=>string,date_nodes=>date,local_time_nodes=>local_time,zoned_time_nodes=>zoned_time,local_datetime_nodes=>local_datetime,zoned_datetime_nodes=>zoned_datetime,duration_nodes=>duration,mixed_nodes=>mixed,list_nodes=>lists,map_nodes=>maps,byte_nodes=>bytes,opaque_node_validity=>opaque_validity,unsupported_node_properties=>unsupported);
    Ok(())
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
pub(super) fn temporal(
    resident: &mut CandleResident,
    delta: &ResidentProjectDelta,
    device: &Device,
) -> Result<()> {
    let mut additions = BTreeMap::<(u8, u64, PropertyId), Vec<TemporalSample>>::new();
    for mutation in &delta.temporal {
        additions
            .entry((
                mutation.entity_kind as u8,
                mutation.target,
                mutation.sample.property,
            ))
            .or_default()
            .push(mutation.sample.clone());
    }
    for (key, samples) in additions {
        let position = resident
            .shared_temporal
            .iter()
            .position(|backing| {
                (backing.entity_kind as u8, backing.target, backing.property) == key
            })
            .ok_or_else(|| Error::internal("temporal delta targets an absent shared column"))?;
        let mut backing = resident.shared_temporal[position].clone();
        let old_rows = backing.len();
        let old_values = backing.values.clone();
        backing.append_samples(&samples)?;
        let (changes, _) =
            property_delta::changed_rows(&old_values, &backing.values, old_rows..backing.len())?;
        let mut bundle = resident
            .temporal_properties
            .remove(&key)
            .ok_or_else(|| Error::internal("temporal resident property lanes are absent"))?;
        // All state here belongs to an unpublished staged generation. Swap a typed view into
        // the common row publisher, restoring graph lanes on both success and failure.
        swap_node_properties(resident, &mut bundle)?;
        let result = (|| {
            if let Some(plan) = property_delta::DictionaryAppend::plan(
                &resident.node_string_dictionary,
                &backing.values,
            )? {
                let before = dictionary_bytes(&resident.node_string_dictionary);
                plan.apply(
                    &mut resident.node_string_dictionary,
                    &mut backing.values,
                    device,
                )?;
                resident.allocated_bytes = resident
                    .allocated_bytes
                    .saturating_sub(before)
                    .saturating_add(dictionary_bytes(&resident.node_string_dictionary));
            }
            property_delta::patch_property_columns(
                resident,
                true,
                &backing.values,
                &changes,
                device,
            )
        })();
        swap_node_properties(resident, &mut bundle)?;
        result?;
        let metadata = bundle
            .temporal_metadata
            .as_mut()
            .ok_or_else(|| Error::internal("temporal native metadata absent"))?;
        macro_rules! raw_lane {
            ($field:ident,$value:expr) => {{
                let updates = samples
                    .iter()
                    .enumerate()
                    .map(|(offset, sample)| (old_rows + offset, ($value)(sample)))
                    .collect();
                let (before, after) = patch(
                    &mut metadata.$field,
                    backing.len(),
                    DType::I64,
                    updates,
                    device,
                )?;
                resident.allocated_bytes = resident
                    .allocated_bytes
                    .saturating_sub(before)
                    .saturating_add(after);
            }};
        }
        raw_lane!(entity_ids, |sample: &TemporalSample| sample.entity_id
            as i64);
        raw_lane!(event_times, |sample: &TemporalSample| sample
            .event_time_nanos);
        raw_lane!(sequences, |sample: &TemporalSample| sample.sequence_index
            as i64);
        if let Some(order) = &metadata.order {
            let before = order.bytes();
            let after = order.prepare_append(old_rows, &backing)?.apply(device)?;
            resident.allocated_bytes = resident
                .allocated_bytes
                .saturating_sub(before)
                .saturating_add(after.bytes());
            metadata.order = Some(after);
        }
        if backing.value_type == TemporalType::Integer {
            let column = resident
                .temporal_integer
                .get_mut(&key)
                .ok_or_else(|| Error::internal("integer history lost its resident lanes"))?;
            macro_rules! lane {
                ($field:ident,$dtype:expr,$value:expr) => {{
                    let updates = samples
                        .iter()
                        .enumerate()
                        .map(|(offset, sample)| (old_rows + offset, ($value)(sample)))
                        .collect();
                    let (before, after) =
                        patch(&mut column.$field, backing.len(), $dtype, updates, device)?;
                    resident.allocated_bytes = resident
                        .allocated_bytes
                        .saturating_sub(before)
                        .saturating_add(after);
                }};
            }
            lane!(entity_ids, DType::I64, |sample: &TemporalSample| {
                u64_order_key(sample.entity_id)
            });
            column.event_times_nanos = metadata.event_times.clone();
            lane!(sequence_indexes, DType::I64, |sample: &TemporalSample| {
                u64_order_key(sample.sequence_index)
            });
            let values = bundle
                .integer
                .get(&backing.property)
                .ok_or_else(|| Error::internal("integer history lost its property"))?;
            column.values = values.values.clone();
            column.validity = values.validity.clone();
            column.rows = backing.len();
        }
        resident.temporal_properties.insert(key, bundle);
        resident.shared_temporal[position] = backing;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
fn dictionary_bytes(dictionary: &StringDictionary) -> usize {
    metal_pages::buffer_bytes(&dictionary.offsets)
        + dictionary
            .bytes
            .as_ref()
            .map_or(0, metal_pages::buffer_bytes)
        + dictionary
            .ranks
            .as_ref()
            .map_or(0, metal_pages::buffer_bytes)
        + dictionary
            .order
            .as_ref()
            .map_or(0, dictionary_order::LexicalOrder::bytes)
}

#[cfg(all(test, feature = "accelerator", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::NodeId;
    use crate::graph::{
        EmbeddingProfile, GraphStore, IndexCatalog, NodeInput, TemporalDeclaration, TemporalStore,
        VectorDeviceImage,
    };

    #[test]
    fn surgical_vector_history_publication_preserves_dirty_pages_and_late_samples() -> Result<()> {
        let _serial = crate::metal_test_guard();
        let Some(device) = crate::metal_test_device() else {
            return Ok(());
        };
        for dtype in [EmbeddingDType::F16, EmbeddingDType::Bf16] {
            let bits = |value: f32| match dtype {
                EmbeddingDType::F16 => f16::from_f32(value).to_bits(),
                EmbeddingDType::Bf16 => bf16::from_f32(value).to_bits(),
            };
            let mut measured = Vec::new();
            for count in [4096_usize, 32768] {
                let mut graph = GraphStore::default();
                let label = graph.catalog_mut().intern_label("Sensor")?;
                let integer = graph.catalog_mut().intern_property("reading")?;
                let text = graph.catalog_mut().intern_property("message")?;
                let vector_property = crate::graph::SEMANTIC_NODE_PROPERTY;
                for id in 1..=count as u64 {
                    graph.insert_node(NodeInput {
                        id: NodeId(id),
                        layer: Layer::Observed,
                        revision: 1,
                        labels: vec![label],
                        properties: vec![(
                            text,
                            ScalarValue::String(
                                format!("dirty-owner-{id:08}-{}", "λ".repeat(129)).into(),
                            ),
                        )],
                    })?;
                }
                let mut history = TemporalStore::default();
                for (property, value_type) in [
                    (integer, TemporalType::Integer),
                    (text, TemporalType::String),
                ] {
                    history.declare(
                        TemporalDeclaration {
                            entity_kind: EntityKind::Node,
                            target: label.0,
                            property,
                            value_type,
                            retention_nanos: i64::MAX,
                        },
                        1_000_000,
                    )?;
                }
                for row in 0..count {
                    for (property, value) in [
                        (integer, ScalarValue::Integer(row as i64)),
                        (
                            text,
                            ScalarValue::String(
                                format!("history-{row:08}-{}", "μ".repeat(129)).into(),
                            ),
                        ),
                    ] {
                        history.append(
                            EntityKind::Node,
                            label.0,
                            TemporalSample {
                                entity_id: 1 + (row % 2) as u64,
                                property,
                                event_time_nanos: row as i64 * 4,
                                sequence_index: row as u64 + 1,
                                value,
                            },
                            1_000_000,
                        )?;
                    }
                }
                let mut image = ResidentProjectImage::build(
                    ProjectId::random(),
                    Bookmark {
                        term: 1,
                        index: count as u64,
                    },
                    &graph,
                    &history,
                    &IndexCatalog::default(),
                )?;
                image.indexes.profile = Some(EmbeddingProfile::new(
                    [1; 32],
                    [2; 32],
                    384,
                    dtype,
                    false,
                    Similarity::Dot,
                )?);
                image.indexes.vectors.push(VectorDeviceImage {
                    property: vector_property,
                    dimension: 384,
                    similarity: Similarity::Dot,
                    dtype,
                    entity_ids: (1..=count as u64).collect(),
                    values: (0..count)
                        .flat_map(|row| {
                            (0..384).map(move |lane| bits(((row + lane) % 17) as f32 / 8.0))
                        })
                        .collect(),
                    versions: vec![1; count],
                    active: vec![1; count],
                });
                let original = CandleResident::upload(image, &device)?;
                let old_vector = original.vectors[&vector_property]
                    .values
                    .as_ref()
                    .expect("vector")
                    .clone();
                let old_norm = original.vectors[&vector_property]
                    .squared_norms
                    .as_ref()
                    .expect("norm")
                    .narrow(0, 2048, 1)
                    .map_err(candle_error)?
                    .to_vec1::<f32>()
                    .map_err(candle_error)?;
                let samples = vec![
                    super::super::super::ResidentTemporalDelta {
                        entity_kind: EntityKind::Node,
                        target: label.0,
                        sample: TemporalSample {
                            entity_id: 1,
                            property: integer,
                            event_time_nanos: 1,
                            sequence_index: count as u64 + 1,
                            value: ScalarValue::Integer(-991),
                        },
                    },
                    super::super::super::ResidentTemporalDelta {
                        entity_kind: EntityKind::Node,
                        target: label.0,
                        sample: TemporalSample {
                            entity_id: 1,
                            property: text,
                            event_time_nanos: 1,
                            sequence_index: count as u64 + 1,
                            value: ScalarValue::String("late history λ".repeat(41).into()),
                        },
                    },
                ];
                let delta = ResidentProjectDelta {
                    project: original.project,
                    bookmark: Bookmark {
                        term: 1,
                        index: count as u64 + 1,
                    },
                    graph: graph.device_delta(2)?,
                    temporal: samples,
                    vectors: vec![ResolvedVectorMutation::Upsert {
                        property: vector_property,
                        entity_id: 2049,
                        revision: 2,
                        coordinates: vec![bits(2.0); 384],
                    }],
                    invalidate_derived: false,
                };
                let budget = original.planned_delta_staging_bytes(&delta)?;
                let capture = metal_pages::Capture::begin()?;
                let staged = original.stage_delta(&delta, &device)?;
                let pages = capture.finish()?;
                measured.push((budget, pages.native_retired_bytes()));
                assert!(
                    pages.retired_bytes() > pages.native_retired_bytes(),
                    "host order pages must be charged too"
                );
                let norm = staged.vectors[&vector_property]
                    .squared_norms
                    .as_ref()
                    .expect("norm")
                    .narrow(0, 2048, 1)
                    .map_err(candle_error)?
                    .to_vec1::<f32>()
                    .map_err(candle_error)?;
                assert_eq!(norm, [1536.0]);
                assert_eq!(
                    old_vector
                        .narrow(0, 2048 * 384, 3)
                        .and_then(|values| values.to_dtype(DType::F32))
                        .and_then(|values| values.to_vec1::<f32>())
                        .map_err(candle_error)?,
                    [1.0, 1.125, 1.25]
                );
                let candidates =
                    Tensor::from_slice(&[0_u32, 2048], 2, &device).map_err(candle_error)?;
                let hits = score_device_vectors(
                    staged.vectors[&vector_property].values.as_ref(),
                    &staged.vectors[&vector_property],
                    &[1.0; 384],
                    1,
                    Some(&candidates),
                    &device,
                    &CancellationToken::new(),
                )?;
                assert_eq!(hits[0].entity_id, 2049);
                assert_eq!(hits[0].score, 768.0);
                assert_eq!(
                    original.vectors[&vector_property]
                        .squared_norms
                        .as_ref()
                        .expect("old norm")
                        .narrow(0, 2048, 1)
                        .map_err(candle_error)?
                        .to_vec1::<f32>()
                        .map_err(candle_error)?,
                    old_norm
                );
                assert_eq!(
                    old_vector.id(),
                    original.vectors[&vector_property]
                        .values
                        .as_ref()
                        .expect("old vector")
                        .id()
                );
                for property in [integer, text] {
                    let key = (EntityKind::Node as u8, label.0, property);
                    let backing = staged
                        .shared_temporal
                        .iter()
                        .find(|c| c.property == property)
                        .expect("history");
                    assert_eq!(backing.len(), count + 1);
                    assert_eq!(backing.event_times_nanos[count], 1);
                    assert_eq!(
                        original
                            .shared_temporal
                            .iter()
                            .find(|c| c.property == property)
                            .expect("pinned history")
                            .len(),
                        count
                    );
                    let metadata = staged.temporal_properties[&key]
                        .temporal_metadata
                        .as_ref()
                        .expect("native metadata");
                    assert_eq!(
                        metadata
                            .event_times
                            .as_ref()
                            .expect("times")
                            .narrow(0, count, 1)
                            .map_err(candle_error)?
                            .to_vec1::<i64>()
                            .map_err(candle_error)?,
                        [1]
                    );
                }
                let input = ResidentNodePipelineRequest {
                    project: original.project,
                    labels: vec![label],
                    layers: LayerMask::AUTHORITY,
                    initial_optional: false,
                    expansion: None,
                    continuations: Vec::new(),
                    correlated_optional: None,
                    relationship_null_filter: None,
                    predicates: Vec::new(),
                    property_filters: Vec::new(),
                    value_matrix: None,
                    mutation: None,
                    orders: Vec::new(),
                    offset: 0,
                    limit: 1,
                    integer_projections: Vec::new(),
                    property_null_projections: Vec::new(),
                    max_output_rows: count + 1,
                };
                let request = ResidentTemporalPipelineRequest {
                    input,
                    binding: super::super::super::ResidentNodeBinding::Start,
                    target: label.0,
                    property: integer,
                    from_nanos: 0,
                    to_nanos: count as i64 * 4 + 1,
                    bookmark_index: count as u64 + 1,
                    window: None,
                    max_output_rows: count + 1,
                };
                let results = staged.execute_temporal_pipeline(
                    &device,
                    &request,
                    &CancellationToken::new(),
                )?;
                assert_eq!(results.values.len(), count / 2 + 1);
                assert_eq!(results.event_times_nanos[1], 1);
                assert_eq!(results.values[1], -991);
                let mut invalid = delta.clone();
                invalid.vectors = vec![ResolvedVectorMutation::Upsert {
                    property: vector_property,
                    entity_id: 2049,
                    revision: 3,
                    coordinates: vec![bits(4.0); 383],
                }];
                invalid.bookmark.index += 1;
                assert!(staged.stage_delta(&invalid, &device).is_err());
                graph.insert_node(NodeInput {
                    id: NodeId(count as u64 + 1),
                    layer: Layer::Observed,
                    revision: 3,
                    labels: vec![label],
                    properties: Vec::new(),
                })?;
                let mut next = ResidentProjectDelta {
                    project: staged.project,
                    bookmark: Bookmark {
                        term: 1,
                        index: count as u64 + 2,
                    },
                    graph: graph.device_delta(3)?,
                    temporal: Vec::new(),
                    vectors: vec![
                        ResolvedVectorMutation::Remove {
                            property: vector_property,
                            entity_id: 2049,
                            revision: 3,
                        },
                        ResolvedVectorMutation::Upsert {
                            property: vector_property,
                            entity_id: count as u64 + 1,
                            revision: 3,
                            coordinates: vec![bits(3.0); 384],
                        },
                    ],
                    invalidate_derived: false,
                };
                let appended = staged.stage_delta(&next, &device)?;
                let column = &appended.vectors[&vector_property];
                assert_eq!(column.rows, count + 1);
                assert_eq!(
                    column
                        .active
                        .as_ref()
                        .expect("active")
                        .narrow(0, 2048, 1)
                        .and_then(|value| value.to_vec1::<u8>())
                        .map_err(candle_error)?,
                    [0]
                );
                assert_eq!(
                    column
                        .node_backed
                        .as_ref()
                        .expect("owner")
                        .narrow(0, count, 1)
                        .and_then(|value| value.to_vec1::<u8>())
                        .map_err(candle_error)?,
                    [1]
                );
                assert_eq!(
                    column
                        .node_rows
                        .as_ref()
                        .expect("owner row")
                        .narrow(0, count, 1)
                        .and_then(|value| value.to_vec1::<u32>())
                        .map_err(candle_error)?,
                    [count as u32]
                );
                assert_eq!(
                    column
                        .squared_norms
                        .as_ref()
                        .expect("new norm")
                        .narrow(0, count, 1)
                        .and_then(|value| value.to_vec1::<f32>())
                        .map_err(candle_error)?,
                    [3456.0]
                );
                next.bookmark.index += 1;
                next.graph = graph.device_delta(4)?;
                next.vectors = vec![ResolvedVectorMutation::Upsert {
                    property: vector_property,
                    entity_id: 2049,
                    revision: 4,
                    coordinates: vec![bits(4.0); 384],
                }];
                let restored = appended.stage_delta(&next, &device)?;
                assert_eq!(restored.vectors[&vector_property].rows, count + 1);
                assert_eq!(
                    restored.vectors[&vector_property]
                        .active
                        .as_ref()
                        .expect("restored active")
                        .narrow(0, 2048, 1)
                        .and_then(|value| value.to_vec1::<u8>())
                        .map_err(candle_error)?,
                    [1]
                );
                eprintln!(
                    "surgical derived dtype={dtype:?} rows={count} staging={budget} retired={}",
                    pages.retired_bytes()
                );
            }
            assert_eq!(
                measured[0].1, measured[1].1,
                "history/vector native dirty pages must be independent of unrelated rows"
            );
            assert!(
                measured
                    .iter()
                    .all(|(budget, _)| *budget < 256 * metal_pages::page_bytes()),
                "staging must stay within the fixed-width index path bound"
            );
        }
        Ok(())
    }
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
pub(super) fn patch<T: Copy>(
    tensor: &mut Option<Tensor>,
    len: usize,
    dtype: DType,
    updates: Vec<(usize, T)>,
    device: &Device,
) -> Result<(usize, usize)> {
    let before = tensor.as_ref().map_or(0, metal_pages::buffer_bytes);
    if updates.is_empty()
        && tensor
            .as_ref()
            .is_none_or(|tensor| tensor.elem_count() == len)
    {
        return Ok((before, before));
    }
    let source = match tensor {
        Some(tensor) => tensor.clone(),
        None => metal_pages::zeros(dtype, 0, device)?,
    };
    *tensor = Some(metal_pages::extend(&source, &updates, len, device)?);
    Ok((before, tensor.as_ref().map_or(0, metal_pages::buffer_bytes)))
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
pub(super) fn vectors(
    resident: &mut CandleResident,
    delta: &ResidentProjectDelta,
    device: &Device,
    previous_node_rows: &PersistentMap<u32>,
    previous_edge_rows: &PersistentMap<u32>,
) -> Result<()> {
    let mut affected = BTreeMap::<PropertyId, BTreeSet<u32>>::new();
    for (property, column) in &resident.vectors {
        let mapping_changed = if *property == crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY {
            delta.graph.edges.iter().any(|edge| {
                stable_id_row(&column.entity_rows, edge.id.0).is_some()
                    && stable_id_row(previous_edge_rows, edge.id.0) != Some(&edge.dense)
            })
        } else {
            delta.graph.nodes.iter().any(|node| {
                stable_id_row(&column.entity_rows, node.id.0).is_some()
                    && stable_id_row(previous_node_rows, node.id.0) != Some(&node.dense)
            })
        };
        if mapping_changed {
            affected.entry(*property).or_default();
        }
    }
    for mutation in &delta.vectors {
        let (property, entity) = match mutation {
            ResolvedVectorMutation::Upsert {
                property,
                entity_id,
                ..
            }
            | ResolvedVectorMutation::Remove {
                property,
                entity_id,
                ..
            } => (*property, *entity_id),
        };
        let backing = resident
            .shared_vectors
            .iter_mut()
            .find(|backing| backing.property == property)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "vector delta targets an absent shared column",
                )
            })?;
        let column = resident
            .vectors
            .get_mut(&property)
            .ok_or_else(|| Error::internal("shared vector has no resident column"))?;
        apply_shared_vector_mutation(backing, &mut column.entity_rows, mutation)?;
        if let Some(row) = stable_id_row(&column.entity_rows, entity) {
            affected.entry(property).or_default().insert(*row);
        }
    }
    for (property, changed) in affected {
        let backing = resident
            .shared_vectors
            .iter_mut()
            .find(|backing| backing.property == property)
            .ok_or_else(|| Error::internal("shared vector disappeared"))?;
        let column = resident
            .vectors
            .get_mut(&property)
            .ok_or_else(|| Error::internal("vector column disappeared"))?;
        let old_rows = column.rows;
        let rows = backing.entity_ids.len();
        let owners = if property == crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY {
            &resident.edge_id_rows
        } else {
            &resident.node_id_rows
        };
        let mut owner_changes = BTreeSet::new();
        if property == crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY {
            for edge in &delta.graph.edges {
                if stable_id_row(previous_edge_rows, edge.id.0) != Some(&edge.dense) {
                    if let Some(row) = stable_id_row(&column.entity_rows, edge.id.0) {
                        owner_changes.insert(*row);
                    }
                }
            }
        } else {
            for node in &delta.graph.nodes {
                if stable_id_row(previous_node_rows, node.id.0) != Some(&node.dense) {
                    if let Some(row) = stable_id_row(&column.entity_rows, node.id.0) {
                        owner_changes.insert(*row);
                    }
                }
            }
        }
        owner_changes.extend((old_rows..rows).map(|row| row as u32));
        macro_rules! patch_lane {
            ($field:ident,$len:expr,$dtype:expr,$updates:expr) => {{
                let (before, after) = patch(&mut column.$field, $len, $dtype, $updates, device)?;
                resident.allocated_bytes = resident
                    .allocated_bytes
                    .saturating_sub(before)
                    .saturating_add(after);
            }};
        }
        patch_lane!(
            entity_ids,
            rows,
            DType::I64,
            (old_rows..rows)
                .map(|row| (row, u64_order_key(backing.entity_ids[row])))
                .collect()
        );
        patch_lane!(
            node_rows,
            rows,
            DType::U32,
            owner_changes
                .iter()
                .map(|row| (
                    *row as usize,
                    stable_id_row(owners, backing.entity_ids[*row as usize])
                        .copied()
                        .unwrap_or(0)
                ))
                .collect()
        );
        patch_lane!(
            node_backed,
            rows,
            DType::U8,
            owner_changes
                .iter()
                .map(|row| (
                    *row as usize,
                    u8::from(stable_id_row(owners, backing.entity_ids[*row as usize]).is_some())
                ))
                .collect()
        );
        let dimension = column.dimension;
        let dtype = column.dtype;
        let updates = changed
            .iter()
            .flat_map(|row| {
                let start = *row as usize * dimension;
                (start..start + dimension).map(|index| (index, backing.values[index]))
            })
            .collect();
        patch_lane!(
            values,
            rows.checked_mul(dimension)
                .ok_or_else(|| Error::internal("vector shape overflow"))?,
            match dtype {
                EmbeddingDType::F16 => DType::F16,
                EmbeddingDType::Bf16 => DType::BF16,
            },
            updates
        );
        let norms = changed
            .iter()
            .map(|row| {
                let start = *row as usize * dimension;
                Ok((
                    *row as usize,
                    shared_vector_squared_norm(backing, start..start + dimension, dtype)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        patch_lane!(squared_norms, rows, DType::F32, norms);
        patch_lane!(
            versions,
            rows,
            DType::I64,
            changed
                .iter()
                .map(|row| (*row as usize, backing.versions[*row as usize] as i64))
                .collect()
        );
        patch_lane!(
            active,
            rows,
            DType::U8,
            changed
                .iter()
                .map(|row| (*row as usize, u8::from(backing.active[*row as usize])))
                .collect()
        );
        column.rows = rows;
        if let Some(index) = resident.ann.get_mut(&property) {
            let before = index
                .stale_row_ids
                .as_ref()
                .map_or(0, metal_pages::buffer_bytes);
            for row in &changed {
                if backing.active[*row as usize] {
                    index.stale_row_set.insert(*row);
                } else {
                    index.stale_row_set.remove(row);
                }
            }
            index
                .stale_row_set
                .publish(&mut index.stale_row_ids, device)?;
            resident.allocated_bytes = resident
                .allocated_bytes
                .saturating_sub(before)
                .saturating_add(
                    index
                        .stale_row_ids
                        .as_ref()
                        .map_or(0, metal_pages::buffer_bytes),
                );
        }
    }
    Ok(())
}
