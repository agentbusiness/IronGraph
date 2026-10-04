//! Incremental exact lexical order. Patricia nodes have stable addresses; inserting a string
//! changes only its search path. Dense ranks are produced for requested IDs by the GPU.
use super::*;
use candle_metal_kernels::metal::ComputePipeline;

const LEAF: u32 = u32::MAX;
const WORDS: usize = 5;

#[derive(Clone)]
pub(super) struct LexicalOrder {
    // left, right, subtree count, discriminating bit, representative dictionary ID
    nodes: PagedVec<[u32; WORDS]>,
    pub(super) packet: Option<Tensor>,
    host_root: u64,
}

pub(super) struct OrderAppend {
    order: LexicalOrder,
    updates: Vec<(usize, u32)>,
    previous_nodes: usize,
    host_pages: Vec<usize>,
    pub(super) staging_bytes: usize,
}

impl OrderAppend {
    pub(super) fn apply(mut self, device: &Device) -> Result<LexicalOrder> {
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

fn bit(bytes: &[u8], position: u32) -> bool {
    let index = position as usize / 9;
    let lane = position as usize % 9;
    bytes
        .get(index)
        .is_some_and(|byte| lane == 0 || byte & (1 << (8 - lane)) != 0)
}

impl LexicalOrder {
    pub(super) fn build(columns: &PropertyColumns, upload: &mut TensorUpload<'_>) -> Result<Self> {
        let mut order = Self {
            nodes: PagedVec::default(),
            packet: None,
            host_root: metal_pages::new_host_root()?,
        };
        let mut dirty = BTreeSet::new();
        let dictionary = columns.string_dictionary();
        for id in 0..dictionary.len() {
            order.insert(
                checked_u32(id, "dictionary order ID")?,
                dictionary,
                &mut dirty,
            )?;
        }
        let words = order
            .nodes
            .iter()
            .flat_map(|node| node.iter().copied())
            .collect::<Vec<_>>();
        order.packet = upload.optional(&words)?;
        Ok(order)
    }

    fn insert(
        &mut self,
        id: u32,
        dictionary: &crate::graph::Dictionary,
        dirty: &mut BTreeSet<usize>,
    ) -> Result<()> {
        let value = dictionary
            .resolve(id)
            .ok_or_else(|| Error::internal("dictionary order lost its entry"))?
            .as_bytes();
        let leaf = [0, 0, 1, LEAF, id];
        if self.nodes.is_empty() {
            self.nodes.push(leaf);
            dirty.insert(0);
            return Ok(());
        }
        let mut position = 0;
        while self.nodes[position][3] != LEAF {
            let node = self.nodes[position];
            position = node[usize::from(bit(value, node[3]))] as usize;
        }
        let previous = dictionary
            .resolve(self.nodes[position][4])
            .ok_or_else(|| Error::internal("dictionary order lost its representative"))?
            .as_bytes();
        let common = value
            .iter()
            .zip(previous)
            .take_while(|(a, b)| a == b)
            .count();
        let suffix = match (value.get(common), previous.get(common)) {
            (Some(a), Some(b)) => 1 + (a ^ b).leading_zeros() as usize,
            (None, None) => {
                return Err(Error::internal(
                    "dictionary order contains a duplicate string",
                ));
            }
            _ => 0,
        };
        let differing = checked_u32(
            common
                .checked_mul(9)
                .and_then(|v| v.checked_add(suffix))
                .ok_or_else(|| Error::internal("dictionary order key exceeds address space"))?,
            "dictionary order key",
        )?;
        position = 0;
        while self.nodes[position][3] < differing {
            let mut node = self.nodes[position];
            node[2] = node[2]
                .checked_add(1)
                .ok_or_else(|| Error::internal("dictionary order count overflow"))?;
            self.nodes
                .replace(position, node)
                .map_err(Error::internal)?;
            dirty.insert(position);
            position = node[usize::from(bit(value, node[3]))] as usize;
        }
        let old = self.nodes[position];
        let old_index = checked_u32(self.nodes.len(), "dictionary order node")?;
        let new_index = old_index
            .checked_add(1)
            .ok_or_else(|| Error::internal("dictionary order node overflow"))?;
        self.nodes.push(old);
        self.nodes.push(leaf);
        dirty.insert(old_index as usize);
        dirty.insert(new_index as usize);
        let children = if bit(value, differing) {
            [old_index, new_index]
        } else {
            [new_index, old_index]
        };
        self.nodes
            .replace(
                position,
                [children[0], children[1], old[2] + 1, differing, id],
            )
            .map_err(Error::internal)?;
        dirty.insert(position);
        Ok(())
    }

    pub(super) fn prepare_append(
        &self,
        before: usize,
        columns: &PropertyColumns,
    ) -> Result<OrderAppend> {
        let mut order = self.clone();
        let dictionary = columns.string_dictionary();
        let mut dirty = BTreeSet::new();
        for id in before..dictionary.len() {
            order.insert(
                checked_u32(id, "dictionary order ID")?,
                dictionary,
                &mut dirty,
            )?;
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
        let pages = updates
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
            (pages + 4) * metal_pages::page_bytes()
                + host_pages.len() * 64 * 1024
                + updates.len() * size_of::<(usize, u32)>()
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
}

pub(super) fn ranks(
    dictionary: &StringDictionary,
    ids: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    if ids.elem_count() == 0 {
        return Tensor::zeros(0, DType::U32, device).map_err(candle_error);
    }
    if let Some(order) = &dictionary.order {
        if let Some(packet) = &order.packet {
            let bytes = match &dictionary.bytes {
                Some(bytes) => bytes.clone(),
                None => Tensor::zeros(1, DType::U8, device).map_err(candle_error)?,
            };
            return ids
                .apply_op1_no_bwd(&Rank {
                    packet: packet.clone(),
                    offsets: dictionary.offsets.clone(),
                    bytes,
                })
                .map_err(candle_error);
        }
        return Tensor::zeros(ids.shape(), DType::U32, device).map_err(candle_error);
    }
    match &dictionary.ranks {
        Some(ranks) => ranks.index_select(ids, 0).map_err(candle_error),
        None => Err(Error::internal(
            "resident dictionary has no lexical order index",
        )),
    }
}

struct Rank {
    packet: Tensor,
    offsets: Tensor,
    bytes: Tensor,
}

impl CustomOp1 for Rank {
    fn name(&self) -> &'static str {
        "irongraph-dictionary-order"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "dictionary order requires Metal".into(),
        ))
    }
    fn metal_fwd(
        &self,
        ids: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let device = ids.device();
        let rows = layout.shape().elem_count();
        let inputs = [&self.packet, &self.offsets, &self.bytes];
        let guards = inputs.map(Tensor::storage_and_layout);
        if ids.dtype() != DType::U32
            || !layout.is_contiguous()
            || guards.iter().any(|(_, layout)| !layout.is_contiguous())
        {
            return Err(candle_core::Error::Msg(
                "dictionary order tensor layout is invalid".into(),
            ));
        }
        let pipeline = pipeline(device)?;
        let output = device.new_buffer_builder().with_size(rows * 4).build()?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(0, Some(ids.buffer()), layout.start_offset() * 4);
        for (index, (guard, layout)) in guards.iter().enumerate() {
            let Storage::Metal(storage) = &**guard else {
                return Err(candle_core::Error::Msg(
                    "dictionary order input moved off Metal".into(),
                ));
            };
            encoder.set_input_buffer(
                index + 1,
                Some(storage.buffer()),
                layout.start_offset() * inputs[index].dtype().size_in_bytes(),
            );
        }
        encoder.set_output_buffer(4, Some(&output), 0);
        encoder.set_bytes(
            5,
            &[
                rows as u64,
                self.offsets.elem_count().saturating_sub(1) as u64,
            ],
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
            MetalStorage::new(output, device.clone(), rows, DType::U32),
            Shape::from(rows),
        ))
    }
}

fn pipeline(device: &candle_core::MetalDevice) -> candle_core::Result<ComputePipeline> {
    static PIPELINE: OnceLock<ComputePipeline> = OnceLock::new();
    if let Some(pipeline) = PIPELINE.get() {
        return Ok(pipeline.clone());
    }
    let library = device.metal_device().new_library_with_source(r#"
#include <metal_stdlib>
using namespace metal;
kernel void dictionary_rank(device const uint* ids [[buffer(0)]], device const uint* nodes [[buffer(1)]], device const uint* offsets [[buffer(2)]], device const uchar* bytes [[buffer(3)]], device uint* out [[buffer(4)]], constant ulong* args [[buffer(5)]], uint row [[thread_position_in_grid]]) {
    if (row >= args[0]) return;
    uint id = ids[row];
    if (id >= args[1]) { out[row] = 0; return; }
    uint start = offsets[id], length = offsets[id+1] - start;
    uint position = 0, rank = 0;
    while (nodes[position*5+3] != 0xffffffffu) {
        uint bit = nodes[position*5+3], index = bit / 9, lane = bit % 9;
        bool right = index < length && (lane == 0 || (bytes[start+index] & (1u << (8-lane))) != 0);
        if (right) rank += nodes[nodes[position*5]*5+2];
        position = nodes[position*5 + uint(right)];
    }
    out[row] = rank;
}

"#, None).map_err(|error| candle_core::Error::Msg(format!("compiling dictionary order: {error}")))?;
    let function = library
        .get_function("dictionary_rank", None)
        .map_err(|error| candle_core::Error::Msg(format!("loading dictionary order: {error}")))?;
    let raw = device
        .metal_device()
        .as_ref()
        .newComputePipelineStateWithFunction_error(function.as_ref())
        .map_err(|error| {
            candle_core::Error::Msg(format!("creating dictionary order pipeline: {error:?}"))
        })?;
    let pipeline = ComputePipeline::new(raw);
    let _ = PIPELINE.set(pipeline.clone());
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surgical_dictionary_insert_preserves_exact_order_and_pinned_ranks() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        for size in [4_096, 32_768] {
            let property = PropertyId(0);
            let mut columns = PropertyColumns::default();
            for id in 0..size {
                let text = format!("dirty-{id:08x}-{}", "retained-payload".repeat(4));
                columns.push_row(&[(property, ScalarValue::String(text.into()))])?;
            }
            let mut upload = TensorUpload::new(&device);
            let dictionary = upload_properties(&mut upload, &mut columns, false)?
                .dictionary
                .ok_or_else(|| Error::internal("test dictionary is absent"))?;
            let old_ids = Tensor::from_slice(&[0_u32, (size - 1) as u32], 2, &device)
                .map_err(candle_error)?;
            let old_ranks = ranks(&dictionary, &old_ids, &device)?
                .to_vec1::<u32>()
                .map_err(candle_error)?;
            let mut current = dictionary.clone();
            let mut branch = metal_pages::BranchPages::default();
            for text in [
                "",
                "dirty-",
                "dirty-00000000",
                "\0",
                "é",
                "😀",
                "dirty-00000000-longer",
            ] {
                columns.set(0, property, &ScalarValue::String(text.into()))?;
                let before = current.offsets.elem_count() - 1;
                let append = current
                    .order
                    .as_ref()
                    .ok_or_else(|| Error::internal("test lexical tree is absent"))?
                    .prepare_append(before, &columns)?;
                assert!(
                    append.updates.len() < 300,
                    "one string patched {} words for {size} entries",
                    append.updates.len()
                );
                let plan = property_delta::DictionaryAppend::plan(&current, &columns)?
                    .ok_or_else(|| Error::internal("test dictionary append is absent"))?;
                let staging_bytes = plan.staging_bytes;
                let capture = metal_pages::Capture::begin()?;
                plan.apply(&mut current, &mut columns, &device)?;
                let writes = capture.finish()?;
                assert!(writes.retired_bytes() > writes.native_retired_bytes());
                assert!(writes.retired_bytes() <= staging_bytes);
                branch.apply(&writes);
                let retained = branch.bytes();
                branch.apply(&writes);
                assert_eq!(branch.bytes(), retained);
                let entries = columns
                    .string_dictionary()
                    .values()
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                let ids = Tensor::from_slice(
                    &(0..entries.len() as u32).collect::<Vec<_>>(),
                    entries.len(),
                    &device,
                )
                .map_err(candle_error)?;
                assert_eq!(
                    ranks(&current, &ids, &device)?
                        .to_vec1::<u32>()
                        .map_err(candle_error)?,
                    resident_string_dictionary_ranks(&entries)?
                );
                assert_eq!(
                    ranks(&dictionary, &old_ids, &device)?
                        .to_vec1::<u32>()
                        .map_err(candle_error)?,
                    old_ranks
                );
            }
        }
        Ok(())
    }
}
