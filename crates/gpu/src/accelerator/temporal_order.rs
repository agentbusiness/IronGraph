//! Stable physical temporal rows with an incremental unsigned lexicographic order index.
use super::*;
use candle_metal_kernels::metal::ComputePipeline;

const WORDS: usize = 13;
const LEAF: u32 = u32::MAX;
const KEY_BITS: usize = 256;

fn zero_vector(rows: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    // Metal rejects zero-byte resources; an empty view retains one harmless backing element.
    let tensor = Tensor::zeros(rows.max(1), dtype, device).map_err(candle_error)?;
    if rows == 0 {
        tensor.narrow(0, 0, 0).map_err(candle_error)
    } else {
        Ok(tensor)
    }
}

#[derive(Clone)]
pub(super) struct TemporalOrder {
    nodes: PagedVec<[u32; WORDS]>,
    pub(super) packet: Option<Tensor>,
    host_root: u64,
}

pub(super) struct OrderAppend {
    order: TemporalOrder,
    updates: Vec<(usize, u32)>,
    previous_nodes: usize,
    host_pages: Vec<usize>,
    pub(super) staging_bytes: usize,
}

impl OrderAppend {
    pub(super) fn apply(mut self, device: &Device) -> Result<TemporalOrder> {
        if self.updates.is_empty() {
            return Ok(self.order);
        }
        let source = match &self.order.packet {
            Some(packet) => packet.clone(),
            None => metal_pages::zeros(DType::U32, 0, device)?,
        };
        self.order.packet = Some(metal_pages::extend(
            &source,
            &self.updates,
            self.order.nodes.len() * WORDS,
            device,
        )?);
        metal_pages::record_host_pages(
            self.order.host_root,
            self.previous_nodes,
            host_page_width(),
            self.host_pages.into_iter(),
            64 * 1024,
        );
        Ok(self.order)
    }
}

fn host_page_width() -> usize {
    (16 * 1024 / size_of::<[u32; WORDS]>()).clamp(1, 4096)
}

fn key(entity: u64, time: i64, sequence: u64, row: u64) -> [u32; 8] {
    let parts = [entity, (time as u64) ^ (1_u64 << 63), sequence, row];
    std::array::from_fn(|word| {
        let value = parts[word / 2];
        if word % 2 == 0 {
            (value >> 32) as u32
        } else {
            value as u32
        }
    })
}

fn bit(key: &[u32; 8], position: u32) -> usize {
    ((key[position as usize / 32] >> (31 - position % 32)) & 1) as usize
}

fn validate_column(column: &TemporalCanonicalColumn) -> Result<()> {
    let len = column.entity_ids.len();
    if column.event_times_nanos.len() != len
        || column.sequence_indexes.len() != len
        || column.values.rows() != len
        || len > (u32::MAX as usize).div_ceil(2)
    {
        return Err(Error::internal(
            "temporal order column shape exceeds its node domain",
        ));
    }
    Ok(())
}

impl TemporalOrder {
    pub(super) fn build(
        column: &TemporalCanonicalColumn,
        upload: &mut TensorUpload<'_>,
    ) -> Result<Self> {
        validate_column(column)?;
        let host_root = metal_pages::new_host_root()?;
        let mut order = Self {
            nodes: PagedVec::default(),
            packet: None,
            host_root,
        };
        let mut dirty = BTreeSet::new();
        for row in 0..column.entity_ids.len() {
            order.insert(row, column, &mut dirty)?;
        }
        let words = order
            .nodes
            .iter()
            .flat_map(|node| node.iter().copied())
            .collect::<Vec<_>>();
        let previous = upload.immutable_properties;
        upload.immutable_properties = true;
        let packet = upload.optional(&words);
        upload.immutable_properties = previous;
        order.packet = packet?;
        Ok(order)
    }

    fn count(&self) -> usize {
        self.nodes.get(0).map_or(0, |root| root[2] as usize)
    }

    fn insert(
        &mut self,
        row: usize,
        column: &TemporalCanonicalColumn,
        dirty: &mut BTreeSet<usize>,
    ) -> Result<()> {
        let value = key(
            column.entity_ids[row],
            column.event_times_nanos[row],
            column.sequence_indexes[row],
            row as u64,
        );
        let mut leaf = [0; WORDS];
        leaf[2] = 1;
        leaf[3] = LEAF;
        leaf[4] = checked_u32(row, "temporal physical row")?;
        leaf[5..].copy_from_slice(&value);
        if self.nodes.is_empty() {
            self.nodes.push(leaf);
            dirty.insert(0);
            return Ok(());
        }
        let mut position = 0;
        while self.nodes[position][3] != LEAF {
            let node = self.nodes[position];
            position = node[bit(&value, node[3])] as usize;
        }
        let previous = self.nodes[position];
        let differing = value
            .iter()
            .zip(&previous[5..])
            .enumerate()
            .find_map(|(word, (a, b))| {
                (a != b).then(|| word * 32 + (a ^ b).leading_zeros() as usize)
            })
            .ok_or_else(|| Error::internal("temporal physical row was indexed twice"))?
            as u32;
        debug_assert!((differing as usize) < KEY_BITS);
        position = 0;
        while self.nodes[position][3] < differing {
            let mut node = self.nodes[position];
            node[2] = node[2]
                .checked_add(1)
                .ok_or_else(|| Error::internal("temporal order count overflow"))?;
            self.nodes
                .replace(position, node)
                .map_err(Error::internal)?;
            dirty.insert(position);
            position = node[bit(&value, node[3])] as usize;
        }
        let previous = self.nodes[position];
        let old_index = checked_u32(self.nodes.len(), "temporal order node")?;
        let new_index = old_index
            .checked_add(1)
            .ok_or_else(|| Error::internal("temporal order node overflow"))?;
        self.nodes.push(previous);
        self.nodes.push(leaf);
        dirty.insert(old_index as usize);
        dirty.insert(new_index as usize);
        let mut branch = leaf;
        branch[bit(&value, differing)] = new_index;
        branch[1 - bit(&value, differing)] = old_index;
        branch[2] = previous[2]
            .checked_add(1)
            .ok_or_else(|| Error::internal("temporal order count overflow"))?;
        branch[3] = differing;
        self.nodes
            .replace(position, branch)
            .map_err(Error::internal)?;
        dirty.insert(position);
        Ok(())
    }

    pub(super) fn prepare_append(
        &self,
        before: usize,
        column: &TemporalCanonicalColumn,
    ) -> Result<OrderAppend> {
        validate_column(column)?;
        if before != self.count() || before > column.entity_ids.len() {
            return Err(Error::internal(
                "temporal order append does not match its physical prefix",
            ));
        }
        let mut order = self.clone();
        let mut dirty = BTreeSet::new();
        for row in before..column.entity_ids.len() {
            order.insert(row, column, &mut dirty)?;
        }
        let updates = dirty
            .iter()
            .flat_map(|position| {
                order.nodes[*position]
                    .into_iter()
                    .enumerate()
                    .map(move |(lane, value)| (position * WORDS + lane, value))
            })
            .collect::<Vec<_>>();
        let native_pages = updates
            .iter()
            .map(|(word, _)| word * 4 / metal_pages::page_bytes())
            .collect::<BTreeSet<_>>()
            .len();
        let host_pages = dirty
            .iter()
            .map(|row| row / host_page_width())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let staging_bytes = if updates.is_empty() {
            0
        } else {
            (native_pages + 8)
                .saturating_mul(metal_pages::page_bytes())
                .saturating_add(host_pages.len().saturating_mul(64 * 1024))
                .saturating_add(dirty.len().saturating_mul(4096))
                .saturating_add(updates.len().saturating_mul(size_of::<(usize, u32)>()))
        };
        Ok(OrderAppend {
            order,
            updates,
            previous_nodes: self.nodes.len(),
            host_pages,
            staging_bytes,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.packet.as_ref().map_or(0, metal_pages::buffer_bytes)
    }

    pub(super) fn bounds(
        &self,
        entity_order_keys: &Tensor,
        from: i64,
        to: i64,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        if from > to
            || entity_order_keys.dtype() != DType::I64
            || entity_order_keys.dims().len() != 1
        {
            return Err(Error::internal("temporal order bounds input is invalid"));
        }
        let rows = entity_order_keys.elem_count();
        let Some(packet) = &self.packet else {
            let empty = zero_vector(rows, DType::I64, device)?;
            return Ok((empty.clone(), empty));
        };
        if rows == 0 {
            let empty = zero_vector(0, DType::I64, device)?;
            return Ok((empty.clone(), empty));
        }
        let combined = entity_order_keys
            .apply_op1_no_bwd(&OrderQuery {
                packet: packet.clone(),
                nodes: self.nodes.len(),
                count: self.count(),
                mode: QueryMode::Bounds { from, to },
            })
            .map_err(candle_error)?;
        Ok((
            combined.narrow(0, 0, rows).map_err(candle_error)?,
            combined.narrow(0, rows, rows).map_err(candle_error)?,
        ))
    }

    pub(super) fn select(&self, ranks: &Tensor) -> Result<Tensor> {
        if ranks.dtype() != DType::I64 || ranks.dims().len() != 1 {
            return Err(Error::internal(
                "temporal order ranks must be a flat I64 tensor",
            ));
        }
        if ranks.elem_count() == 0 {
            return zero_vector(0, DType::U32, ranks.device());
        }
        let Some(packet) = &self.packet else {
            return Tensor::full(u32::MAX, ranks.shape(), ranks.device()).map_err(candle_error);
        };
        ranks
            .apply_op1_no_bwd(&OrderQuery {
                packet: packet.clone(),
                nodes: self.nodes.len(),
                count: self.count(),
                mode: QueryMode::Select,
            })
            .map_err(candle_error)
    }
}

#[derive(Clone, Copy)]
enum QueryMode {
    Bounds { from: i64, to: i64 },
    Select,
}
struct OrderQuery {
    packet: Tensor,
    nodes: usize,
    count: usize,
    mode: QueryMode,
}

impl CustomOp1 for OrderQuery {
    fn name(&self) -> &'static str {
        "irongraph-temporal-order"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "temporal order requires Metal".into(),
        ))
    }
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let (storage, packet_layout) = self.packet.storage_and_layout();
        let Storage::Metal(packet) = &*storage else {
            return Err(candle_core::Error::Msg(
                "temporal order packet moved off Metal".into(),
            ));
        };
        if input.dtype() != DType::I64
            || !layout.is_contiguous()
            || !packet_layout.is_contiguous()
            || packet.dtype() != DType::U32
        {
            return Err(candle_core::Error::Msg(
                "temporal order tensor layout is invalid".into(),
            ));
        }
        let rows = layout.shape().elem_count();
        let device = input.device();
        let pipelines = pipelines(device)?;
        let (pipeline, elements, dtype, from, to) = match self.mode {
            QueryMode::Bounds { from, to } => (
                &pipelines.bounds,
                rows.checked_mul(2).ok_or_else(|| {
                    candle_core::Error::Msg("temporal bound output overflow".into())
                })?,
                DType::I64,
                from as u64,
                to as u64,
            ),
            QueryMode::Select => (&pipelines.select, rows, DType::U32, 0, 0),
        };
        let bytes = elements
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| candle_core::Error::Msg("temporal order output overflow".into()))?;
        let output = device.new_buffer_builder().with_size(bytes).build()?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_input_buffer(0, Some(input.buffer()), layout.start_offset() * 8);
        encoder.set_input_buffer(1, Some(packet.buffer()), packet_layout.start_offset() * 4);
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(
            3,
            &[rows as u64, self.nodes as u64, self.count as u64, from, to],
        );
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: rows.div_ceil(256),
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, dtype),
            Shape::from(elements),
        ))
    }
}

#[derive(Clone)]
struct Pipelines {
    bounds: ComputePipeline,
    select: ComputePipeline,
}

fn pipelines(device: &candle_core::MetalDevice) -> candle_core::Result<Pipelines> {
    static PIPELINES: OnceLock<Pipelines> = OnceLock::new();
    if let Some(pipelines) = PIPELINES.get() {
        return Ok(pipelines.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(SOURCE, None)
        .map_err(|error| candle_core::Error::Msg(format!("compiling temporal order: {error}")))?;
    let load = |name| -> candle_core::Result<ComputePipeline> {
        let function = library
            .get_function(name, None)
            .map_err(|error| candle_core::Error::Msg(format!("loading temporal order: {error}")))?;
        let raw = device
            .metal_device()
            .as_ref()
            .newComputePipelineStateWithFunction_error(function.as_ref())
            .map_err(|error| {
                candle_core::Error::Msg(format!("creating temporal order pipeline: {error:?}"))
            })?;
        Ok(ComputePipeline::new(raw))
    };
    let pipelines = Pipelines {
        bounds: load("temporal_order_bounds")?,
        select: load("temporal_order_select")?,
    };
    let _ = PIPELINES.set(pipelines.clone());
    Ok(pipelines)
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
uint temporal_leaf(device const uint* nodes, uint node_count, uint rank) {
    uint node = 0;
    for (uint depth = 0; depth <= 256; ++depth) {
        if (node >= node_count) return 0xffffffffu;
        if (nodes[ulong(node)*13+3] == 0xffffffffu) return node;
        uint left = nodes[ulong(node)*13], right = nodes[ulong(node)*13+1];
        if (left >= node_count || right >= node_count) return 0xffffffffu;
        uint count = nodes[ulong(left)*13+2];
        if (rank < count) node = left;
        else { rank -= count; node = right; }
    }
    return 0xffffffffu;
}
bool temporal_key_less(device const uint* nodes, uint leaf, thread const uint* key) {
    for (uint word = 0; word < 8; ++word) {
        uint value = nodes[ulong(leaf)*13+5+word];
        if (value != key[word]) return value < key[word];
    }
    return false;
}
uint temporal_lower(device const uint* nodes, uint node_count, uint count, ulong entity, ulong time) {
    uint key[8] = {uint(entity>>32),uint(entity),uint(time>>32),uint(time),0,0,0,0};
    uint first = 0, last = count;
    while (first < last) {
        uint middle = first + (last-first)/2;
        uint leaf = temporal_leaf(nodes, node_count, middle);
        if (leaf == 0xffffffffu) return count;
        if (temporal_key_less(nodes,leaf,key)) first = middle+1;
        else last = middle;
    }
    return first;
}
kernel void temporal_order_bounds(device const long* entities [[buffer(0)]], device const uint* nodes [[buffer(1)]], device long* out [[buffer(2)]], constant ulong* args [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= args[0]) return;
    ulong entity = ulong(entities[row]) ^ (1ul<<63);
    out[row] = long(temporal_lower(nodes,uint(args[1]),uint(args[2]),entity,args[3]^(1ul<<63)));
    out[args[0]+row] = long(temporal_lower(nodes,uint(args[1]),uint(args[2]),entity,args[4]^(1ul<<63)));
}
kernel void temporal_order_select(device const long* ranks [[buffer(0)]], device const uint* nodes [[buffer(1)]], device uint* out [[buffer(2)]], constant ulong* args [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= args[0]) return;
    long rank = ranks[row];
    if (rank < 0 || ulong(rank) >= args[2]) { out[row] = 0xffffffffu; return; }
    uint leaf = temporal_leaf(nodes,uint(args[1]),uint(rank));
    out[row] = leaf == 0xffffffffu ? 0xffffffffu : nodes[ulong(leaf)*13+4];
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_column() -> TemporalCanonicalColumn {
        TemporalCanonicalColumn {
            entity_kind: crate::types::EntityKind::Node,
            target: 1,
            property: PropertyId(0),
            value_type: crate::graph::TemporalType::Integer,
            entity_ids: PagedVec::default(),
            event_times_nanos: PagedVec::default(),
            sequence_indexes: PagedVec::default(),
            values: PropertyColumns::default(),
        }
    }

    fn sample(
        entity_id: u64,
        event_time_nanos: i64,
        sequence_index: u64,
    ) -> crate::graph::TemporalSample {
        crate::graph::TemporalSample {
            entity_id,
            property: PropertyId(0),
            event_time_nanos,
            sequence_index,
            value: ScalarValue::Integer(
                (entity_id as i64)
                    .wrapping_mul(7919)
                    .wrapping_add(event_time_nanos),
            ),
        }
    }

    fn oracle(column: &TemporalCanonicalColumn) -> Vec<u32> {
        let mut rows = (0..column.entity_ids.len() as u32).collect::<Vec<_>>();
        rows.sort_by_key(|row| {
            let row = *row as usize;
            (
                column.entity_ids[row],
                column.event_times_nanos[row],
                column.sequence_indexes[row],
                row,
            )
        });
        rows
    }

    fn verify(
        order: &TemporalOrder,
        column: &TemporalCanonicalColumn,
        device: &Device,
    ) -> Result<()> {
        let expected = oracle(column);
        let ranks = if expected.is_empty() {
            zero_vector(0, DType::I64, device)?
        } else {
            Tensor::from_vec(
                (0..expected.len() as i64).collect::<Vec<_>>(),
                expected.len(),
                device,
            )
            .map_err(candle_error)?
        };
        assert_eq!(
            order
                .select(&ranks)?
                .to_vec1::<u32>()
                .map_err(candle_error)?,
            expected
        );
        let entities = [0_u64, 1, 7, 42, 1_u64 << 63, u64::MAX];
        let entity_keys = entities.map(|entity| (entity ^ (1_u64 << 63)) as i64);
        let entity_keys =
            Tensor::from_slice(&entity_keys, entity_keys.len(), device).map_err(candle_error)?;
        for (from, to) in [
            (i64::MIN, i64::MAX),
            (-7, 0),
            (0, 7),
            (7, 7),
            (i64::MIN, -1),
            (i64::MAX - 1, i64::MAX),
        ] {
            let (first, last) = order.bounds(&entity_keys, from, to, device)?;
            let first = first.to_vec1::<i64>().map_err(candle_error)?;
            let last = last.to_vec1::<i64>().map_err(candle_error)?;
            for (index, entity) in entities.iter().enumerate() {
                let lower = |time| {
                    expected.partition_point(|row| {
                        let row = *row as usize;
                        (
                            column.entity_ids[row],
                            column.event_times_nanos[row],
                            column.sequence_indexes[row],
                            row,
                        ) < (*entity, time, 0, 0)
                    }) as i64
                };
                assert_eq!(
                    first[index],
                    lower(from),
                    "lower bound for unsigned entity {entity}"
                );
                assert_eq!(
                    last[index],
                    lower(to),
                    "upper bound for unsigned entity {entity}"
                );
            }
        }
        let invalid = Tensor::from_slice(
            &[-1_i64, expected.len() as i64, expected.len() as i64 + 1],
            3,
            device,
        )
        .map_err(candle_error)?;
        assert_eq!(
            order
                .select(&invalid)?
                .to_vec1::<u32>()
                .map_err(candle_error)?,
            vec![u32::MAX; 3]
        );
        Ok(())
    }

    #[test]
    fn surgical_temporal_order_native_bounds_select_and_pins_match_unsigned_oracle() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        for rows in [4096, 32768] {
            let mut column = empty_column();
            let entities = [0_u64, 7, 42, 1_u64 << 63, u64::MAX];
            let samples = (0..rows)
                .map(|row| {
                    sample(
                        entities[row % entities.len()],
                        ((row * 3571 % 257) as i64) - 128,
                        (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    )
                })
                .collect::<Vec<_>>();
            column.append_samples(&samples)?;
            let mut upload = TensorUpload::new(&device);
            let original = TemporalOrder::build(&column, &mut upload)?;
            let old_column = column.clone();
            verify(&original, &old_column, &device)?;
            let mut current = original.clone();
            let additions = [
                sample(0, i64::MIN, u64::MAX),
                sample(u64::MAX, i64::MIN, u64::MAX),
                sample(u64::MAX, i64::MAX, 0),
                sample(7, 0, u64::MAX),
                sample(7, 0, 0),
                sample(7, 0, 0),
                sample(1_u64 << 63, -7, 1_u64 << 63),
            ];
            let mut largest_words = 0;
            let mut largest_retired = 0;
            let mut footprint = metal_pages::BranchPages::default();
            for addition in additions {
                let before = column.entity_ids.len();
                column.append_samples(&[addition])?;
                let append = current.prepare_append(before, &column)?;
                largest_words = largest_words.max(append.updates.len());
                assert!(append.updates.len() <= (KEY_BITS + 3) * WORDS);
                assert!(append.staging_bytes <= 16 * 1024 * 1024);
                assert!(
                    append.order.nodes.detached_page_bytes_from(&current.nodes)
                        <= (KEY_BITS + 3) * 16 * 1024
                );
                let changed_nodes = append.updates.len() / WORDS;
                let retired_host = append
                    .host_pages
                    .iter()
                    .filter(|page| **page * host_page_width() < append.previous_nodes)
                    .count()
                    * 64
                    * 1024;
                let capture = metal_pages::Capture::begin()?;
                current = append.apply(&device)?;
                let writes = capture.finish()?;
                assert!(writes.retired_bytes() >= retired_host);
                let written = writes.retired_bytes() - retired_host;
                largest_retired = largest_retired.max(written);
                assert!(written <= changed_nodes * metal_pages::page_bytes());
                assert!(
                    written < original.bytes(),
                    "one append replaced the entire dirty index"
                );
                let prior_footprint = footprint.clone();
                let prior_bytes = footprint.bytes();
                footprint.apply(&writes);
                let retained = footprint.bytes();
                footprint.apply(&writes);
                assert_eq!(
                    footprint.bytes(),
                    retained,
                    "host/native page identities must deduplicate"
                );
                assert_eq!(
                    prior_footprint.bytes(),
                    prior_bytes,
                    "an independent pinned branch changed accounting"
                );
                assert!(retained >= retired_host);
                verify(&current, &column, &device)?;
                verify(&original, &old_column, &device)?;
            }
            assert!(current.prepare_append(0, &column).is_err());
            let unchanged = current.prepare_append(column.entity_ids.len(), &column)?;
            assert_eq!(unchanged.staging_bytes, 0);
            assert!(unchanged.updates.is_empty());
            assert_eq!(
                unchanged.apply(&device)?.packet.as_ref().map(Tensor::id),
                current.packet.as_ref().map(Tensor::id)
            );
            eprintln!(
                "surgical temporal order: rows={rows}, maximum_patched_words={largest_words}, maximum_retired_bytes={largest_retired}, retained_native_bytes={}",
                current.bytes()
            );
        }
        Ok(())
    }

    #[test]
    fn surgical_temporal_order_empty_append_and_invalid_shapes() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        let mut column = empty_column();
        let mut upload = TensorUpload::new(&device);
        let empty = TemporalOrder::build(&column, &mut upload)?;
        assert_eq!(empty.bytes(), 0);
        verify(&empty, &column, &device)?;
        column.append_samples(&[sample(u64::MAX, i64::MIN, u64::MAX)])?;
        let current = empty.prepare_append(0, &column)?.apply(&device)?;
        verify(&current, &column, &device)?;
        let empty_entities = zero_vector(0, DType::I64, &device)?;
        assert_eq!(
            current
                .bounds(&empty_entities, 0, 0, &device)?
                .0
                .elem_count(),
            0
        );
        assert_eq!(current.select(&empty_entities)?.elem_count(), 0);
        assert!(current.bounds(&empty_entities, 1, 0, &device).is_err());
        let wrong_type = Tensor::zeros(1, DType::U32, &device).map_err(candle_error)?;
        assert!(current.select(&wrong_type).is_err());
        column.sequence_indexes.push(1);
        assert!(current.prepare_append(1, &column).is_err());
        verify(&empty, &empty_column(), &device)?;
        Ok(())
    }
}
