//! Row-local structural publication. The radix packet's positions never shift when rows change.

use super::*;
use std::sync::Arc;

#[derive(Clone, Default)]
pub(super) struct RowMap(PersistentMap<Arc<(Vec<u32>, Vec<u32>)>>);

impl RowMap {
    pub(super) fn get(&self, row: &u32) -> Option<&(Vec<u32>, Vec<u32>)> {
        self.0.get(u128::from(*row)).map(Arc::as_ref)
    }

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.len() == 0
    }
}

#[derive(Clone, Copy, Default)]
struct PayloadSpan {
    begin: u32,
    capacity: u32,
}

#[derive(Clone)]
struct FreeSpan {
    begin: u32,
    next: Option<Arc<Self>>,
}

impl Drop for FreeSpan {
    fn drop(&mut self) {
        // Preserve any shared suffix and avoid recursive destruction of a large free list.
        let mut tail = self.next.take();
        while let Some(next) = tail {
            match Arc::try_unwrap(next) {
                Ok(mut unique) => tail = unique.next.take(),
                Err(_) => break,
            }
        }
    }
}

/// Pair 0 is `[pair_count, root]`; branch pairs are `[zero, one]`, leaves are
/// `[payload_begin, payload_end]`, and payload pairs are `[neighbor, edge]`.
/// A 32-bit row lookup follows exactly 32 branch links. Zero means absent, including
/// a missing subtree; an existing leaf with equal bounds is an explicit empty row.
#[derive(Clone, Default)]
pub(super) struct ResidentCsrOverlay {
    pub(super) rows: RowMap,
    pub(super) packet: Option<Tensor>,
    spans: PersistentMap<PayloadSpan>,
    free: [Option<Arc<FreeSpan>>; 32],
    pending: Vec<AdjacencyRowDeviceDelta>,
    #[cfg(test)]
    last_patched_words: usize,
}

impl ResidentCsrOverlay {
    pub(super) fn merge_rows(
        &mut self,
        replacements: &[AdjacencyRowDeviceDelta],
        node_count: usize,
    ) -> Result<()> {
        for row in replacements {
            if row.dense as usize >= node_count
                || row.neighbors.len() != row.edges.len()
                || row
                    .neighbors
                    .iter()
                    .any(|neighbor| *neighbor as usize >= node_count)
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident CSR row is invalid",
                ));
            }
            if self
                .rows
                .get(&row.dense)
                .is_some_and(|old| old.0 == row.neighbors && old.1 == row.edges)
            {
                continue;
            }
            self.rows.0.insert_cow(
                u128::from(row.dense),
                Arc::new((row.neighbors.clone(), row.edges.clone())),
            );
            self.pending.push(row.clone());
        }
        Ok(())
    }

    pub(super) fn tensor_bytes(&self) -> usize {
        self.packet.as_ref().map_or(0, |packet| {
            #[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
            if matches!(packet.device(), Device::Metal(_)) {
                return metal_pages::buffer_bytes(packet);
            }
            packet.elem_count() * size_of::<u32>()
        })
    }

    #[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
    fn prepare_updates(&mut self) -> Result<(Vec<(usize, u32)>, usize)> {
        let old = self
            .packet
            .as_ref()
            .map(|packet| metal_shared_tensor_prefix::<u32>(packet, packet.elem_count()))
            .transpose()?;
        let words = old
            .as_ref()
            .map_or(&[][..], crate::graph::SharedFlat::as_slice);
        let mut patches = BTreeMap::<usize, u32>::new();
        let mut pairs = words.first().copied().unwrap_or(1);
        if words.is_empty() {
            patches.insert(0, 1);
            patches.insert(1, 0);
        }
        fn read(words: &[u32], patches: &BTreeMap<usize, u32>, index: usize) -> u32 {
            patches
                .get(&index)
                .copied()
                .unwrap_or_else(|| words.get(index).copied().unwrap_or(0))
        }
        fn allocate(pairs: &mut u32, count: u32) -> Result<u32> {
            let start = *pairs;
            *pairs = pairs
                .checked_add(count)
                .filter(|end| *end <= u32::MAX / 2)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "CSR arena exceeds u32 words",
                    )
                })?;
            Ok(start)
        }
        for row in &self.pending {
            let mut link = 1_usize;
            for shift in (0..32).rev() {
                let mut branch = read(words, &patches, link);
                if branch == 0 {
                    branch = allocate(&mut pairs, 1)?;
                    patches.insert(link, branch);
                    patches.insert(branch as usize * 2, 0);
                    patches.insert(branch as usize * 2 + 1, 0);
                }
                link = branch as usize * 2 + ((row.dense >> shift) & 1) as usize;
            }
            let mut leaf = read(words, &patches, link);
            if leaf == 0 {
                leaf = allocate(&mut pairs, 1)?;
                patches.insert(link, leaf);
            }
            let required = checked_u32(row.neighbors.len(), "CSR replacement degree")?;
            let old_span = self
                .spans
                .get(u128::from(row.dense))
                .copied()
                .unwrap_or_default();
            let span = if required <= old_span.capacity {
                old_span
            } else {
                if old_span.capacity != 0 {
                    let class = old_span.capacity.trailing_zeros() as usize;
                    self.free[class] = Some(Arc::new(FreeSpan {
                        begin: old_span.begin,
                        next: self.free[class].take(),
                    }));
                }
                let capacity = required.checked_next_power_of_two().ok_or_else(|| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "CSR degree capacity exhausted",
                    )
                })?;
                let class = capacity.trailing_zeros() as usize;
                let begin = if let Some(free) = self.free[class].take() {
                    self.free[class] = free.next.clone();
                    free.begin
                } else {
                    allocate(&mut pairs, capacity)?
                };
                let span = PayloadSpan { begin, capacity };
                self.spans.insert_cow(u128::from(row.dense), span);
                span
            };
            patches.insert(leaf as usize * 2, span.begin);
            patches.insert(leaf as usize * 2 + 1, span.begin + required);
            for (offset, (neighbor, edge)) in row.neighbors.iter().zip(&row.edges).enumerate() {
                let position = (span.begin as usize + offset) * 2;
                patches.insert(position, *neighbor);
                patches.insert(position + 1, *edge);
            }
        }
        patches.insert(0, pairs);
        Ok((patches.into_iter().collect(), pairs as usize * 2))
    }

    #[cfg(all(
        test,
        feature = "accelerator",
        any(target_os = "macos", target_os = "ios")
    ))]
    pub(super) fn rebuild_tensors(&mut self, upload: &mut TensorUpload<'_>) -> Result<()> {
        let prepared = if self.pending.is_empty() {
            None
        } else {
            Some(self.prepare_updates()?)
        };
        self.rebuild_prepared(upload, prepared)
    }

    #[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
    fn rebuild_prepared(
        &mut self,
        upload: &mut TensorUpload<'_>,
        prepared: Option<(Vec<(usize, u32)>, usize)>,
    ) -> Result<()> {
        let Some((updates, words)) = prepared else {
            upload.allocated_bytes = upload.allocated_bytes.saturating_add(self.tensor_bytes());
            return Ok(());
        };
        #[cfg(test)]
        {
            self.last_patched_words = updates.len();
        }
        let source = match self.packet.as_ref() {
            Some(packet) => packet.clone(),
            None => metal_pages::zeros(DType::U32, 2, upload.device)?,
        };
        self.packet = Some(metal_pages::extend(
            &source,
            &updates,
            words,
            upload.device,
        )?);
        upload.allocated_bytes = upload.allocated_bytes.saturating_add(self.tensor_bytes());
        self.pending.clear();
        Ok(())
    }
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
fn merge_changed_rows(
    overlay: &mut ResidentCsrOverlay,
    cold: &crate::graph::Csr,
    rows: &[AdjacencyRowDeviceDelta],
    node_count: usize,
) -> Result<()> {
    for row in rows {
        if overlay.rows.get(&row.dense).is_none()
            && row.neighbors.len() == row.edges.len()
            && cold.row(row.dense).is_some_and(|entries| {
                entries.eq(row.neighbors.iter().copied().zip(row.edges.iter().copied()))
            })
        {
            continue;
        }
        overlay.merge_rows(std::slice::from_ref(row), node_count)?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
pub(super) struct Plan {
    pub(super) bytes: usize,
    overlays: Vec<(ResidentCsrOverlay, Option<(Vec<(usize, u32)>, usize)>)>,
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
pub(super) fn plan(resident: &CandleResident, delta: &ResidentProjectDelta) -> Result<Plan> {
    let backing = resident
        .shared_graph
        .as_ref()
        .ok_or_else(|| Error::internal("shared graph missing"))?;
    let page = metal_pages::page_bytes();
    let mut bytes = 0_usize;
    let mut overlays = Vec::with_capacity(2);
    for (original, cold, rows) in [
        (
            &resident.outgoing_overlay,
            &backing.outgoing,
            &delta.graph.outgoing,
        ),
        (
            &resident.incoming_overlay,
            &backing.incoming,
            &delta.graph.incoming,
        ),
    ] {
        let mut overlay = original.clone();
        merge_changed_rows(&mut overlay, cold, rows, delta.graph.node_capacity)?;
        if overlay.pending.is_empty() {
            overlays.push((overlay, None));
            continue;
        }
        let (updates, words) = overlay.prepare_updates()?;
        let old_bytes = original.tensor_bytes();
        let changed_pages = updates
            .iter()
            .map(|(word, _)| word * 4 / page)
            .filter(|index| index * page < old_bytes)
            .collect::<BTreeSet<_>>()
            .len();
        let growth = (words * 4)
            .div_ceil(page)
            .saturating_mul(page)
            .saturating_sub(old_bytes);
        // Both directions share radix paths within a batch. Count each native page once,
        // plus bounded row-map paths, payload clones and temporary patch-map entries.
        bytes = bytes
            .saturating_add(changed_pages * page)
            .saturating_add(growth)
            .saturating_add(overlay.pending.len() * 16 * 1024)
            .saturating_add(updates.len() * 96);
        overlays.push((overlay, Some((updates, words))));
    }
    // Appended cold CSR offsets touch only their old terminal page and newly exposed suffix.
    let target_bytes = delta
        .graph
        .node_capacity
        .saturating_add(1)
        .saturating_mul(4);
    for offsets in [&resident.outgoing_offsets, &resident.incoming_offsets] {
        if offsets.elem_count() < delta.graph.node_capacity.saturating_add(1) {
            bytes = bytes
                .saturating_add(page)
                .saturating_add(
                    target_bytes
                        .div_ceil(page)
                        .saturating_mul(page)
                        .saturating_sub(metal_pages::buffer_bytes(offsets)),
                )
                .saturating_add(
                    (delta.graph.node_capacity + 1 - offsets.elem_count())
                        .saturating_mul(size_of::<(usize, u32)>()),
                );
        }
    }
    Ok(Plan { bytes, overlays })
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
fn patch<T: Copy + WithDType + PartialEq + Send + Sync + 'static>(
    tensor: &mut Option<Tensor>,
    rows: usize,
    mut updates: Vec<(usize, T)>,
    device: &Device,
) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    if let Some(tensor) = tensor.as_ref() {
        // Appended cells can already match the owned buffer's padding. Compare the
        // actual bytes: a narrowed view can retain nonzero cells past its shape.
        let compare_rows = rows.min(metal_pages::buffer_bytes(tensor) / size_of::<T>());
        let previous_view = if compare_rows > tensor.elem_count() {
            metal_pages::extend::<T>(tensor, &[], compare_rows, device)?
        } else {
            tensor.clone()
        };
        let previous = metal_shared_tensor_prefix::<T>(&previous_view, previous_view.elem_count())?;
        updates.retain(|(row, value)| previous.as_slice().get(*row) != Some(value));
    }
    if updates.is_empty()
        && tensor
            .as_ref()
            .is_some_and(|tensor| tensor.elem_count() == rows)
    {
        return Ok(());
    }
    let source = match tensor.as_ref() {
        Some(tensor) => tensor.clone(),
        None => metal_pages::zeros(T::DTYPE, rows, device)?,
    };
    *tensor = Some(metal_pages::extend(&source, &updates, rows, device)?);
    Ok(())
}

/// Call before applying the canonical delta, while old labels are still available.
#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
pub(super) fn patch_fixed_and_labels(
    resident: &mut CandleResident,
    delta: &ResidentProjectDelta,
    device: &Device,
) -> Result<()> {
    let graph = &delta.graph;
    macro_rules! nodes {
        ($field:ident, $value:expr) => {
            patch(
                &mut resident.$field,
                graph.node_capacity,
                graph
                    .nodes
                    .iter()
                    .map(|row| (row.dense as usize, ($value)(row)))
                    .collect(),
                device,
            )?;
        };
    }
    macro_rules! edges {
        ($field:ident, $value:expr) => {
            patch(
                &mut resident.$field,
                graph.edge_capacity,
                graph
                    .edges
                    .iter()
                    .map(|row| (row.dense as usize, ($value)(row)))
                    .collect(),
                device,
            )?;
        };
    }
    nodes!(
        node_entity_ids,
        |row: &crate::graph::NodeDeviceDelta| row.id.0 as i64
    );
    nodes!(node_id_order_keys, |row: &crate::graph::NodeDeviceDelta| {
        u64_order_key(row.id.0)
    });
    nodes!(
        node_revisions,
        |row: &crate::graph::NodeDeviceDelta| row.revision as i64
    );
    nodes!(node_active, |row: &crate::graph::NodeDeviceDelta| u8::from(
        row.active
    ));
    nodes!(node_layers, |row: &crate::graph::NodeDeviceDelta| row.layer
        as u8);
    edges!(
        edge_entity_ids,
        |row: &crate::graph::EdgeDeviceDelta| row.id.0 as i64
    );
    edges!(edge_sources, |row: &crate::graph::EdgeDeviceDelta| row
        .source);
    edges!(edge_targets, |row: &crate::graph::EdgeDeviceDelta| row
        .target);
    edges!(
        edge_types,
        |row: &crate::graph::EdgeDeviceDelta| row.relationship_type.0 as i64
    );
    edges!(edge_layers, |row: &crate::graph::EdgeDeviceDelta| row.layer
        as u8);
    edges!(edge_active, |row: &crate::graph::EdgeDeviceDelta| u8::from(
        row.active
    ));
    let old = resident
        .shared_graph
        .as_ref()
        .ok_or_else(|| Error::internal("shared graph missing"))?;
    let mut labels = BTreeMap::<LabelId, Vec<(usize, u8)>>::new();
    for node in &graph.nodes {
        let previous = old.node_labels.get(node.dense).unwrap_or(&[]);
        for label in previous
            .iter()
            .chain(&node.labels)
            .copied()
            .collect::<BTreeSet<_>>()
        {
            let present = node.labels.contains(&label);
            if previous.contains(&label) != present || !resident.node_labels.contains_key(&label) {
                labels
                    .entry(label)
                    .or_default()
                    .push((node.dense as usize, u8::from(present)));
            }
        }
        match stable_id_row(&resident.node_id_rows, node.id.0) {
            Some(existing) if *existing != node.dense => {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "stable node ID changed dense row",
                ));
            }
            None => {
                resident
                    .node_id_rows
                    .insert_cow(stable_id_key(node.id.0), node.dense);
            }
            Some(_) => {}
        }
    }
    for edge in &graph.edges {
        match stable_id_row(&resident.edge_id_rows, edge.id.0) {
            Some(existing) if *existing != edge.dense => {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "stable relationship ID changed dense row",
                ));
            }
            None => {
                resident
                    .edge_id_rows
                    .insert_cow(stable_id_key(edge.id.0), edge.dense);
            }
            Some(_) => {}
        }
    }
    if old.node_ids.len() != graph.node_capacity {
        for (label, bitmap) in &mut resident.node_labels {
            let updates = labels.remove(label).unwrap_or_default();
            *bitmap = metal_pages::extend(bitmap, &updates, graph.node_capacity, device)?;
        }
    }
    for (label, updates) in labels {
        if let Some(bitmap) = resident.node_labels.get_mut(&label) {
            *bitmap = metal_pages::patch(bitmap, &updates, device)?;
        } else {
            let bitmap = metal_pages::zeros(DType::U8, graph.node_capacity, device)?;
            resident
                .node_labels
                .insert(label, metal_pages::patch(&bitmap, &updates, device)?);
        }
    }
    Ok(())
}

/// Extends only the cold offset domain and updates only changed overlay rows. The backing must
/// already contain this delta's fixed/property rows, with CSR extension deferred.
#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
pub(super) fn patch_adjacency(
    resident: &mut CandleResident,
    backing: &mut GraphSharedBacking,
    delta: &ResidentProjectDelta,
    device: &Device,
    plan: Plan,
) -> Result<()> {
    let target = delta.graph.node_capacity.saturating_add(1);
    for (tensor, csr) in [
        (&mut resident.outgoing_offsets, &mut backing.outgoing),
        (&mut resident.incoming_offsets, &mut backing.incoming),
    ] {
        if tensor.elem_count() < target {
            let terminal = csr
                .offsets()
                .last()
                .copied()
                .ok_or_else(|| Error::internal("CSR offsets absent"))?;
            let compare_rows = target.min(metal_pages::buffer_bytes(tensor) / size_of::<u32>());
            let view = metal_pages::extend::<u32>(tensor, &[], compare_rows, device)?;
            let previous = metal_shared_tensor_prefix::<u32>(&view, compare_rows)?;
            let updates = (tensor.elem_count()..target)
                .filter(|row| previous.as_slice().get(*row) != Some(&terminal))
                .map(|row| (row, terminal))
                .collect::<Vec<_>>();
            *tensor = metal_pages::extend(tensor, &updates, target, device)?;
            csr.rebase_cow_offsets_extension(metal_shared_tensor_prefix(tensor, target)?)?;
        }
    }
    let previous =
        resident.outgoing_overlay.tensor_bytes() + resident.incoming_overlay.tensor_bytes();
    let mut upload = TensorUpload::new(device);
    for (target, (mut overlay, prepared)) in [
        &mut resident.outgoing_overlay,
        &mut resident.incoming_overlay,
    ]
    .into_iter()
    .zip(plan.overlays)
    {
        overlay.rebuild_prepared(&mut upload, prepared)?;
        *target = overlay;
    }
    let current =
        resident.outgoing_overlay.tensor_bytes() + resident.incoming_overlay.tensor_bytes();
    resident.allocated_bytes = resident
        .allocated_bytes
        .saturating_sub(previous)
        .saturating_add(current);
    resident.outgoing_rows = PagedVec::default();
    resident.incoming_rows = PagedVec::default();
    // Canonical fixed columns already hold persistent, bounded row updates. Rebinding every
    // column here would rebuild its page directory across unrelated rows on every publication.
    Ok(())
}

#[cfg(all(test, feature = "accelerator", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::graph::{EdgeInput, GraphStore, LayerMask, NodeInput};
    use crate::{EdgeId, Layer, NodeId, ScalarValue};

    fn device() -> Result<Device> {
        Device::new_metal(0).map_err(candle_error)
    }

    #[test]
    fn fixed_column_append_reuses_equal_padding_and_isolates_changed_siblings() -> Result<()> {
        let device = device()?;
        let source = metal_pages::zeros(candle_core::DType::U32, 4_097, &device)?;
        let dirty = metal_pages::patch(&source, &[(4_096, 77_u32)], &device)?;
        let root = dirty.narrow(0, 0, 4_096).map_err(candle_error)?;
        let address = |tensor: &Tensor| {
            let (storage, _) = tensor.storage_and_layout();
            let Storage::Metal(storage) = &*storage else {
                panic!("expected Metal storage")
            };
            storage.buffer().contents() as usize
        };
        let mut same = Some(root.clone());
        patch(&mut same, 4_097, vec![(4_096, 77_u32)], &device)?;
        assert_eq!(address(same.as_ref().unwrap()), address(&root));
        let mut first = Some(root.clone());
        patch(&mut first, 4_097, vec![(4_096, 11_u32)], &device)?;
        let mut second = Some(root.clone());
        patch(&mut second, 4_097, vec![(4_096, 22_u32)], &device)?;
        assert_ne!(address(first.as_ref().unwrap()), address(&root));
        assert_ne!(address(second.as_ref().unwrap()), address(&root));
        for (tensor, expected) in [
            (same.as_ref().unwrap(), 77),
            (first.as_ref().unwrap(), 11),
            (second.as_ref().unwrap(), 22),
        ] {
            let values = metal_shared_tensor_prefix::<u32>(tensor, 4_097)?;
            assert_eq!(values.as_slice()[4_096], expected);
            assert!(values.as_slice()[..4_096].iter().all(|value| *value == 0));
        }
        let mut zero = Some(source.narrow(0, 0, 4_096).map_err(candle_error)?);
        patch(&mut zero, 4_097, vec![(4_096, 0_u32)], &device)?;
        assert_eq!(address(zero.as_ref().unwrap()), address(&source));
        Ok(())
    }

    #[test]
    fn surgical_csr_free_list_drop_preserves_shared_suffix_without_recursion() {
        let pinned = Arc::new(FreeSpan {
            begin: 7,
            next: None,
        });
        let mut tail = Some(Arc::clone(&pinned));
        for begin in 0..100_000 {
            tail = Some(Arc::new(FreeSpan { begin, next: tail }));
        }
        drop(tail);
        assert_eq!(pinned.begin, 7);
        assert_eq!(Arc::strong_count(&pinned), 1);
    }

    fn row(dense: u32, entries: usize, seed: u32) -> AdjacencyRowDeviceDelta {
        AdjacencyRowDeviceDelta {
            dense,
            neighbors: (0..entries)
                .map(|offset| (seed + offset as u32) % 8_192)
                .collect(),
            edges: (0..entries).map(|offset| seed + offset as u32).collect(),
        }
    }

    fn decoded(packet: &Tensor, dense: u32) -> Result<Option<Vec<(u32, u32)>>> {
        let shared = metal_shared_tensor_prefix::<u32>(packet, packet.elem_count())?;
        let words = shared.as_slice();
        assert_eq!(words[0] as usize * 2, words.len());
        let mut link = words[1];
        for shift in (0..32).rev() {
            if link == 0 {
                return Ok(None);
            }
            assert!(link < words[0]);
            link = words[link as usize * 2 + ((dense >> shift) & 1) as usize];
        }
        if link == 0 {
            return Ok(None);
        }
        let begin = words[link as usize * 2];
        let end = words[link as usize * 2 + 1];
        Ok(Some(
            (begin..end)
                .map(|position| {
                    (
                        words[position as usize * 2],
                        words[position as usize * 2 + 1],
                    )
                })
                .collect(),
        ))
    }

    #[test]
    fn surgical_csr_edit_is_independent_of_prior_overlay_rows_and_preserves_readers() -> Result<()>
    {
        let device = device()?;
        let mut costs = Vec::new();
        for count in [64, 4_096] {
            let mut overlay = ResidentCsrOverlay::default();
            let dirty = (0..count)
                .map(|dense| row(dense, 2, dense))
                .collect::<Vec<_>>();
            overlay.merge_rows(&dirty, 8_192)?;
            overlay.rebuild_tensors(&mut TensorUpload::new(&device))?;
            let pinned = overlay.clone();
            let untouched = overlay.rows.0.get(7).cloned().unwrap();
            let bytes = overlay.tensor_bytes();
            overlay.merge_rows(&[row(3, 2, 7_000)], 8_192)?;
            overlay.rebuild_tensors(&mut TensorUpload::new(&device))?;
            costs.push(overlay.last_patched_words);
            assert_eq!(overlay.tensor_bytes(), bytes);
            assert!(Arc::ptr_eq(&untouched, overlay.rows.0.get(7).unwrap()));
            assert_eq!(
                decoded(pinned.packet.as_ref().unwrap(), 3)?,
                Some(vec![(3, 3), (4, 4)])
            );
            assert_eq!(
                decoded(overlay.packet.as_ref().unwrap(), 3)?,
                Some(vec![(7_000, 7_000), (7_001, 7_001)])
            );
            assert_eq!(decoded(overlay.packet.as_ref().unwrap(), 8_191)?, None);
            overlay.merge_rows(&[row(3, 0, 0)], 8_192)?;
            overlay.rebuild_tensors(&mut TensorUpload::new(&device))?;
            assert_eq!(
                decoded(overlay.packet.as_ref().unwrap(), 3)?,
                Some(Vec::new())
            );
            assert_eq!(overlay.tensor_bytes(), bytes);
        }
        assert_eq!(
            costs,
            vec![7, 7],
            "one dirty row must patch only its descriptor and payload"
        );
        Ok(())
    }

    #[test]
    fn surgical_csr_capacity_growth_reuses_spans_and_pinned_empty_rows() -> Result<()> {
        let device = device()?;
        let mut overlay = ResidentCsrOverlay::default();
        overlay.merge_rows(&[row(0, 0, 0)], 8_192)?;
        overlay.rebuild_tensors(&mut TensorUpload::new(&device))?;
        let pinned = overlay.clone();
        overlay.merge_rows(&[row(0, 6, 20)], 8_192)?;
        overlay.rebuild_tensors(&mut TensorUpload::new(&device))?;
        let high_water = overlay.tensor_bytes();
        for revision in 0..128 {
            overlay.merge_rows(&[row(0, 1 + revision % 6, revision as u32)], 8_192)?;
            overlay.rebuild_tensors(&mut TensorUpload::new(&device))?;
            assert_eq!(overlay.tensor_bytes(), high_water);
            assert!(overlay.last_patched_words <= 15);
        }
        assert_eq!(
            decoded(pinned.packet.as_ref().unwrap(), 0)?,
            Some(Vec::new())
        );
        assert_eq!(pinned.rows.get(&0).unwrap().0.len(), 0);
        Ok(())
    }

    #[test]
    fn surgical_overlay_only_paths_cover_grid_push_pull_and_persistent_transition() -> Result<()> {
        let device = device()?;
        let mut graph = GraphStore::default();
        let kind = graph.catalog_mut().intern_relationship_type("LINK")?;
        for id in 1..=128 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        let original = CandleResident::upload(
            ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?)),
            &device,
        )?;
        let endpoints = (2..=10)
            .map(|target| (1, target))
            .chain((2..=10).map(|source| (source, source + 9)))
            .chain((19..40).map(|source| (source, source + 1)));
        for (edge, (source, target)) in endpoints.enumerate() {
            graph.insert_edge(EdgeInput {
                id: EdgeId(edge as u64 + 1),
                source: NodeId(source),
                target: NodeId(target),
                relationship_type: kind,
                layer: Layer::Observed,
                revision: 2,
                properties: Vec::new(),
            })?;
        }
        let delta = ResidentProjectDelta {
            project: original.project,
            bookmark: Bookmark { term: 1, index: 2 },
            graph: graph.device_delta(2)?,
            temporal: Vec::new(),
            vectors: Vec::new(),
            invalidate_derived: false,
        };
        assert!(original.planned_delta_staging_bytes(&delta)? < 8 * 1024 * 1024);
        let staged = original.stage_delta(&delta, &device)?;
        assert!(staged.outgoing_neighbors.is_none());
        assert_eq!(staged.outgoing_overlay.rows.len(), 31);
        let rebuilt = CandleResident::upload(
            ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?)),
            &device,
        )?;
        let cancel = CancellationToken::new();
        for procedure in [
            crate::ResidentGraphProcedure::BreadthFirst { source_dense: 0 },
            crate::ResidentGraphProcedure::DepthFirst { source_dense: 0 },
            crate::ResidentGraphProcedure::DijkstraUnit { source_dense: 0 },
            crate::ResidentGraphProcedure::ShortestPath {
                source_dense: 0,
                target_dense: 39,
            },
        ] {
            let request = crate::ResidentGraphProcedureRequest {
                project: original.project,
                layers: LayerMask::ALL,
                procedure,
                max_output_rows: 128,
                deadline: None,
            };
            let actual = staged.execute_graph_procedure(&device, &request, &cancel)?;
            let expected = rebuilt.execute_graph_procedure(&device, &request, &cancel)?;
            assert_eq!(actual, expected, "overlay-only {procedure:?}");
        }
        assert!(original.expand_out(&device, &[0], &cancel)?.is_empty());
        graph.insert_node(NodeInput {
            id: NodeId(129),
            layer: Layer::Observed,
            revision: 3,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
        let appended = rebuilt.stage_delta(
            &ResidentProjectDelta {
                project: original.project,
                bookmark: Bookmark { term: 1, index: 3 },
                graph: graph.device_delta(3)?,
                temporal: Vec::new(),
                vectors: Vec::new(),
                invalidate_derived: false,
            },
            &device,
        )?;
        for (before, after) in [
            (&rebuilt.outgoing_offsets, &appended.outgoing_offsets),
            (&rebuilt.incoming_offsets, &appended.incoming_offsets),
        ] {
            let old = metal_shared_tensor_prefix::<u32>(before, before.elem_count())?;
            let new = metal_shared_tensor_prefix::<u32>(after, after.elem_count())?;
            assert_eq!(
                old.as_slice().as_ptr(),
                new.as_slice().as_ptr(),
                "prepared offsets remain shared"
            );
            assert_eq!(old.len(), 129);
            let gpu_values = after.to_vec1::<u32>().map_err(candle_error)?;
            assert_eq!(&gpu_values[..129], old.as_slice());
            assert_eq!(gpu_values[129], *old.as_slice().last().unwrap());
        }
        assert!(appended.expand_out(&device, &[128], &cancel)?.is_empty());
        assert_eq!(
            rebuilt.expand_out(&device, &[0], &cancel)?,
            appended.expand_out(&device, &[0], &cancel)?
        );
        Ok(())
    }

    #[test]
    fn surgical_structural_publication_keeps_labels_properties_and_pinned_topology() -> Result<()> {
        let device = device()?;
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("Vertex")?;
        let selected = graph.catalog_mut().intern_label("Selected")?;
        let body = graph.catalog_mut().intern_property("body")?;
        let kind = graph.catalog_mut().intern_relationship_type("LINK")?;
        let content =
            ScalarValue::String(Arc::from("complete unrelated source text ".repeat(2_048)));
        for id in 1..=4_096 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
                labels: vec![label],
                properties: vec![(body, content.clone())],
            })?;
        }
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: kind,
            layer: Layer::Observed,
            revision: 1,
            properties: Vec::new(),
        })?;
        let pinned = CandleResident::upload(
            ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?)),
            &device,
        )?;
        let property_id = pinned.string_nodes[&body].values.as_ref().unwrap().id();
        let old_label_id = pinned.node_labels[&label].id();
        graph.add_node_labels(NodeId(3), vec![selected], 2)?;
        graph.delete_edge(EdgeId(1), 2)?;
        graph.insert_edge(EdgeInput {
            id: EdgeId(2),
            source: NodeId(3),
            target: NodeId(4),
            relationship_type: kind,
            layer: Layer::Workspace,
            revision: 2,
            properties: Vec::new(),
        })?;
        let delta = ResidentProjectDelta {
            project: ProjectId(uuid::Uuid::nil()),
            bookmark: Bookmark { term: 1, index: 2 },
            graph: graph.device_delta(2)?,
            temporal: Vec::new(),
            vectors: Vec::new(),
            invalidate_derived: false,
        };
        let current = pinned.stage_delta(&delta, &device)?;
        device.synchronize().map_err(candle_error)?;
        assert_eq!(
            current.string_nodes[&body].values.as_ref().unwrap().id(),
            property_id,
            "structural changes must retain unrelated property tensors"
        );
        assert_eq!(
            current.node_labels[&label].id(),
            old_label_id,
            "unchanged label bitmaps must retain their tensor"
        );
        assert!(!pinned.node_labels.contains_key(&selected));
        assert_eq!(
            current.node_labels[&selected]
                .to_vec1::<u8>()
                .map_err(candle_error)?[2],
            1
        );
        let cancel = CancellationToken::new();
        assert_eq!(pinned.expand_out(&device, &[0], &cancel)?, vec![(0, 1, 0)]);
        assert!(current.expand_out(&device, &[0], &cancel)?.is_empty());
        assert_eq!(current.expand_out(&device, &[2], &cancel)?, vec![(2, 3, 1)]);
        assert_eq!(current.expand_in(&device, &[3], &cancel)?, vec![(2, 3, 1)]);
        let result = current.execute_graph_procedure(
            &device,
            &crate::ResidentGraphProcedureRequest {
                project: ProjectId(uuid::Uuid::nil()),
                layers: LayerMask::ALL,
                procedure: crate::ResidentGraphProcedure::BreadthFirst { source_dense: 2 },
                max_output_rows: 4_096,
                deadline: None,
            },
            &cancel,
        )?;
        let crate::ResidentGraphProcedureResult::BreadthFirst {
            node_rows,
            distance,
        } = result
        else {
            return Err(Error::internal("BFS returned another result shape"));
        };
        assert_eq!(node_rows, vec![2, 3]);
        assert_eq!(distance, vec![0, 1]);
        Ok(())
    }
}
