//! Immutable CSR/CSC adjacency with append-only deltas and tombstones.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock},
};

use serde::{Deserialize, Serialize};

use crate::{Error, ErrorCode, Result};

use super::{
    persistent::{PagedVec, PersistentMap},
    shared::{FlatColumn, SharedFlat},
};

/// Compact sparse row adjacency carrying edge ordinals and neighbor ordinals.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Csr {
    offsets: FlatColumn<u32>,
    neighbors: FlatColumn<u32>,
    edges: FlatColumn<u32>,
}

impl Csr {
    pub fn build(node_count: usize, triples: &[(u32, u32, u32)]) -> Result<Self> {
        Self::build_owned(node_count, triples.to_vec())
    }

    /// Builds the transposed adjacency without first materializing a reversed input and then
    /// cloning it again inside [`Self::build`]. Full graph procedures construct outgoing and
    /// incoming CSR together, so avoiding that redundant edge-sized allocation materially lowers
    /// their peak setup memory.
    pub fn build_transposed(node_count: usize, triples: &[(u32, u32, u32)]) -> Result<Self> {
        Self::build_owned(
            node_count,
            triples
                .iter()
                .map(|(source, target, edge)| (*target, *source, *edge))
                .collect(),
        )
    }

    fn build_owned(node_count: usize, mut sorted: Vec<(u32, u32, u32)>) -> Result<Self> {
        sorted.sort_unstable_by_key(|(source, target, edge)| (*source, *target, *edge));
        let mut offsets = vec![0_u32; node_count.saturating_add(1)];
        for (source, _, _) in &sorted {
            let source = *source as usize;
            if source >= node_count {
                return Err(Error::invalid_data("adjacency source is out of bounds"));
            }
            offsets[source + 1] = offsets[source + 1].checked_add(1).ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "adjacency degree exhausted",
                )
            })?;
        }
        for row in 1..offsets.len() {
            offsets[row] = offsets[row].checked_add(offsets[row - 1]).ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "adjacency offset exhausted",
                )
            })?;
        }
        let mut neighbors = Vec::with_capacity(sorted.len());
        let mut edges = Vec::with_capacity(sorted.len());
        for (_, target, edge) in sorted {
            if target as usize >= node_count {
                return Err(Error::invalid_data("adjacency target is out of bounds"));
            }
            neighbors.push(target);
            edges.push(edge);
        }
        Ok(Self {
            offsets: FlatColumn::from_vec(offsets),
            neighbors: FlatColumn::from_vec(neighbors),
            edges: FlatColumn::from_vec(edges),
        })
    }

    /// Grows the row count without touching the neighbor or edge arrays.
    ///
    /// Appended rows have no adjacency yet, so every new offset repeats the current terminal
    /// offset. Rebuilding instead — which is what happens when a delta only changes the node
    /// count — materializes every edge in the project into a triple vector, sorts it twice, and
    /// allocates two fresh CSRs, so inserting one edgeless node cost time and memory proportional
    /// to the whole graph. This is the same result in time proportional to the rows added.
    pub fn extend_rows(&mut self, node_count: usize) -> Result<()> {
        let current = self.offsets.as_slice();
        let Some(terminal) = current.last().copied() else {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "adjacency offsets are empty",
            ));
        };
        let required = node_count.saturating_add(1);
        if current.len() > required {
            return Err(Error::invalid_data(
                "adjacency row count cannot be reduced by extension",
            ));
        }
        if current.len() == required {
            return Ok(());
        }
        let mut offsets = current.to_vec();
        offsets.resize(required, terminal);
        self.offsets = FlatColumn::from_vec(offsets);
        Ok(())
    }

    /// Replaces complete named rows in one ordered pass without materializing or sorting triples.
    /// Delta producers already provide rows in canonical neighbor/edge order, so rebuilding a
    /// `(source, target, edge)` list and sorting it repeats work and doubles peak staging memory.
    pub fn replace_rows(
        &self,
        node_count: usize,
        replacements: &BTreeMap<u32, (&[u32], &[u32])>,
    ) -> Result<Self> {
        let current_rows = self.offsets.as_slice().len().saturating_sub(1);
        if current_rows > node_count {
            return Err(Error::invalid_data(
                "adjacency row count cannot move backwards",
            ));
        }
        let current_offsets = self.offsets.as_slice();
        let current_neighbors = self.neighbors.as_slice();
        let current_edges = self.edges.as_slice();
        if current_neighbors.len() != current_edges.len()
            || current_offsets.last().copied().map(|value| value as usize)
                != Some(current_neighbors.len())
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "existing adjacency columns are structurally inconsistent",
            ));
        }
        for (row, (neighbors, edges)) in replacements {
            if *row as usize >= node_count
                || neighbors.len() != edges.len()
                || neighbors
                    .iter()
                    .any(|neighbor| *neighbor as usize >= node_count)
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "replacement adjacency row is structurally invalid",
                ));
            }
        }

        let mut offsets = Vec::with_capacity(node_count.saturating_add(1));
        offsets.push(0);
        let mut replacement = replacements.iter().peekable();
        let mut total = 0_usize;
        for row in 0..node_count {
            let dense = u32::try_from(row)
                .map_err(|_| Error::new(ErrorCode::ResultBudgetExceeded, "node row exceeds u32"))?;
            let width = if replacement
                .peek()
                .is_some_and(|(candidate, _)| **candidate == dense)
            {
                replacement
                    .next()
                    .map_or(0, |(_, (neighbors, _))| neighbors.len())
            } else if row < current_rows {
                (current_offsets[row + 1] - current_offsets[row]) as usize
            } else {
                0
            };
            total = total.checked_add(width).ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "adjacency entries exceed u32",
                )
            })?;
            offsets.push(u32::try_from(total).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "adjacency entries exceed u32",
                )
            })?);
        }

        let mut neighbors = Vec::with_capacity(total);
        let mut edges = Vec::with_capacity(total);
        let mut copied_through = 0_usize;
        for (row, (replacement_neighbors, replacement_edges)) in replacements {
            let row = *row as usize;
            let start = if row < current_rows {
                current_offsets[row] as usize
            } else {
                current_neighbors.len()
            };
            let end = if row < current_rows {
                current_offsets[row + 1] as usize
            } else {
                current_neighbors.len()
            };
            neighbors.extend_from_slice(&current_neighbors[copied_through..start]);
            edges.extend_from_slice(&current_edges[copied_through..start]);
            neighbors.extend_from_slice(replacement_neighbors);
            edges.extend_from_slice(replacement_edges);
            copied_through = end;
        }
        neighbors.extend_from_slice(&current_neighbors[copied_through..]);
        edges.extend_from_slice(&current_edges[copied_through..]);
        if neighbors.len() != total || edges.len() != total {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "replacement adjacency cardinality disagrees with offsets",
            ));
        }
        Ok(Self {
            offsets: FlatColumn::from_vec(offsets),
            neighbors: FlatColumn::from_vec(neighbors),
            edges: FlatColumn::from_vec(edges),
        })
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn from_shared(
        offsets: SharedFlat<u32>,
        neighbors: SharedFlat<u32>,
        edges: SharedFlat<u32>,
    ) -> Result<Self> {
        let offsets_slice = offsets.as_slice();
        if offsets_slice.first().copied() != Some(0)
            || offsets_slice.windows(2).any(|pair| pair[0] > pair[1])
            || offsets_slice.last().copied().map(|value| value as usize) != Some(neighbors.len())
            || neighbors.len() != edges.len()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "shared adjacency columns are structurally inconsistent",
            ));
        }
        Ok(Self {
            offsets: FlatColumn::from_shared(offsets),
            neighbors: FlatColumn::from_shared(neighbors),
            edges: FlatColumn::from_shared(edges),
        })
    }

    /// Binds a CSR written by the selected device in the current unpublished command stream.
    /// Structural values cannot be read safely until that stream reaches its publication fence;
    /// lengths are checked here and the producer's kernels prove monotonic offsets and exact row
    /// copies. The ordinary [`Self::from_shared`] path remains the validating constructor for
    /// host-populated allocations.
    pub fn from_device_generated_shared(
        offsets: SharedFlat<u32>,
        neighbors: SharedFlat<u32>,
        edges: SharedFlat<u32>,
    ) -> Result<Self> {
        if offsets.len() == 0 || neighbors.len() != edges.len() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "device-generated adjacency columns have invalid lengths",
            ));
        }
        Ok(Self {
            offsets: FlatColumn::from_shared(offsets),
            neighbors: FlatColumn::from_shared(neighbors),
            edges: FlatColumn::from_shared(edges),
        })
    }

    /// Rebinds only the row-offset allocation after empty rows are appended. The neighbor and
    /// edge generations remain byte-identical because an edgeless node cannot change either
    /// payload column.
    pub fn rebase_shared_offsets(&mut self, offsets: SharedFlat<u32>) -> Result<()> {
        let values = offsets.as_slice();
        if values.first().copied() != Some(0)
            || values.windows(2).any(|pair| pair[0] > pair[1])
            || values.last().copied().map(|value| value as usize)
                != Some(self.neighbors.as_slice().len())
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "shared adjacency offsets are structurally inconsistent",
            ));
        }
        self.offsets = FlatColumn::from_shared(offsets);
        Ok(())
    }

    /// Rebinds an appended offset prefix from the same allocation, validating only the new tail.
    /// The immutable prefix was validated when this CSR was admitted; the allocation-identity
    /// check proves it cannot have been substituted while a pinned generation retains it.
    pub fn rebase_shared_offsets_extension(&mut self, offsets: SharedFlat<u32>) -> Result<()> {
        let FlatColumn::Shared(previous) = &self.offsets else {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "adjacency offset extension has no shared prefix",
            ));
        };
        let previous_len = previous.len();
        let values = offsets.as_slice();
        let previous_terminal = previous.as_slice().last().copied();
        let extended_prefix = previous.is_prefix_of(&offsets);
        let boundary = previous_len
            .checked_sub(1)
            .and_then(|index| values.get(index).copied());
        let tail_monotonic = previous_len != 0
            && values
                .get(previous_len - 1..)
                .is_some_and(|tail| tail.windows(2).all(|pair| pair[0] <= pair[1]));
        let terminal = values.last().copied().map(|value| value as usize);
        let neighbor_len = self.neighbors.as_slice().len();
        if previous_len == 0
            || values.len() <= previous_len
            || !extended_prefix
            || boundary != previous_terminal
            || !tail_monotonic
            || terminal != Some(neighbor_len)
        {
            // Full-prefix diagnostics are failure-only. The successful delta path must inspect
            // only the appended tail, never scale with unrelated resident rows.
            let nonzero_offsets = values.iter().filter(|value| **value != 0).count();
            let first_nonzero = values
                .iter()
                .enumerate()
                .find(|(_, value)| **value != 0)
                .map(|(index, value)| (index, *value));
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "shared adjacency offset extension is structurally inconsistent: previous_len={previous_len} next_len={} same_allocation={extended_prefix} previous_terminal={previous_terminal:?} boundary={boundary:?} next_terminal={terminal:?} neighbors={neighbor_len} tail_monotonic={tail_monotonic} nonzero_offsets={nonzero_offsets} first_nonzero={first_nonzero:?}",
                    values.len(),
                ),
            ));
        }
        let retained = previous.extended_prefix(values.len()).map_err(|message| {
            Error::new(
                ErrorCode::CorruptStorage,
                format!("shared adjacency prefix extension failed: {message}"),
            )
        })?;
        self.offsets = FlatColumn::from_shared(retained);
        Ok(())
    }

    #[must_use]
    pub fn row(
        &self,
        node: u32,
    ) -> Option<impl DoubleEndedIterator<Item = (u32, u32)> + ExactSizeIterator + '_> {
        let row = node as usize;
        let start = *self.offsets.as_slice().get(row)? as usize;
        let end = *self.offsets.as_slice().get(row + 1)? as usize;
        Some(
            self.neighbors
                .as_slice()
                .get(start..end)?
                .iter()
                .copied()
                .zip(self.edges.as_slice().get(start..end)?.iter().copied()),
        )
    }

    /// Multi-source frontier reachability via iterated boolean sparse matrix–vector products.
    ///
    /// Treats this CSR as a boolean adjacency matrix `A` and repeatedly forms the next frontier as
    /// the boolean product `Aᵀ · f`: the union of the out-neighbourhoods of the current frontier,
    /// minus already-visited nodes. Starting from `seeds` (hop 0), it returns, for every node, the
    /// smallest hop count at which it is reached, or `None` if it is never reached within
    /// `max_hops`. `max_hops = None` runs to the transitive-closure fixpoint. `check` is polled once
    /// per hop for bounded-latency cancellation.
    ///
    /// This is the O(hops × touched-edges) formulation — one whole frontier level advances per
    /// iteration rather than one node per queue pop — which is exactly the shape a GPU expands with
    /// one dispatch per hop, and it seeds from a whole set at once. Because it advances level by
    /// level and marks each node at its first (shortest) hop, the distances are shortest-hop
    /// distances; for unbounded transitive reachability the reachable *set* it yields also coincides
    /// with trail-semantics reachability, which is what makes it a safe accelerator for
    /// `(a)-[:R*]->(b)`-style distinct-endpoint reachability.
    pub fn multi_source_frontier_distances(
        &self,
        seeds: impl IntoIterator<Item = u32>,
        max_hops: Option<u32>,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Vec<Option<u32>>> {
        let node_count = self.offsets().len().saturating_sub(1);
        let mut distance = vec![None; node_count];
        let mut frontier: Vec<u32> = Vec::new();
        for seed in seeds {
            let index = seed as usize;
            if index < node_count && distance[index].is_none() {
                distance[index] = Some(0);
                frontier.push(seed);
            }
        }
        let mut hop = 0_u32;
        let mut since_check = 0_usize;
        while !frontier.is_empty() {
            if max_hops.is_some_and(|max| hop >= max) {
                break;
            }
            check()?;
            let next_hop = hop.saturating_add(1);
            let mut next: Vec<u32> = Vec::new();
            for &node in &frontier {
                let Some(row) = self.row(node) else {
                    continue;
                };
                for (neighbor, _) in row {
                    // A single hop can touch millions of edges on a wide graph; keep cancellation
                    // latency bounded within the frontier expansion, not just once per hop.
                    since_check = since_check.saturating_add(1);
                    if since_check >= 4096 {
                        since_check = 0;
                        check()?;
                    }
                    let index = neighbor as usize;
                    if distance[index].is_none() {
                        distance[index] = Some(next_hop);
                        next.push(neighbor);
                    }
                }
            }
            frontier = next;
            hop = next_hop;
        }
        Ok(distance)
    }

    /// Distinct nodes reachable from `seeds` in a number of hops within `[min_hops, max_hops]`, using
    /// [`Self::multi_source_frontier_distances`]. `max_hops = None` is unbounded. Only correct as a
    /// substitute for trail-semantics variable-length reachability when `min_hops <= 1` (for larger
    /// minimums, shortest-hop distance is not the same predicate as "exists a walk of length ≥ min").
    pub fn multi_source_reachable(
        &self,
        seeds: impl IntoIterator<Item = u32>,
        min_hops: u32,
        max_hops: Option<u32>,
        check: impl FnMut() -> Result<()>,
    ) -> Result<Vec<u32>> {
        let distance = self.multi_source_frontier_distances(seeds, max_hops, check)?;
        Ok(distance
            .into_iter()
            .enumerate()
            .filter_map(|(node, hop)| hop.filter(|hop| *hop >= min_hops).map(|_| node as u32))
            .collect())
    }

    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        self.offsets.as_slice()
    }

    #[must_use]
    pub fn neighbors(&self) -> &[u32] {
        self.neighbors.as_slice()
    }

    #[must_use]
    pub fn edges(&self) -> &[u32] {
        self.edges.as_slice()
    }
}

/// Recent adjacency change merged with the immutable base during reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdjacencyDelta {
    Insert { source: u32, target: u32, edge: u32 },
    Delete { source: u32, target: u32, edge: u32 },
}

/// Bidirectional immutable adjacency plus a bounded append-only change buffer.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Adjacency {
    outgoing: Csr,
    incoming: Csr,
    deltas: PagedVec<AdjacencyDelta>,
    #[serde(skip)]
    delta_rows: OnceLock<Arc<DeltaRows>>,
}

#[derive(Clone, Debug, Default)]
struct DeltaRows {
    outgoing: PersistentMap<PagedVec<AdjacencyDelta>>,
    incoming: PersistentMap<PagedVec<AdjacencyDelta>>,
}

impl DeltaRows {
    fn append(&mut self, delta: AdjacencyDelta) {
        let (source, target) = match delta {
            AdjacencyDelta::Insert { source, target, .. }
            | AdjacencyDelta::Delete { source, target, .. } => (source, target),
        };
        let mut outgoing = self
            .outgoing
            .get(u128::from(source))
            .cloned()
            .unwrap_or_default();
        outgoing.push(delta);
        self.outgoing.insert(u128::from(source), outgoing);
        let mut incoming = self
            .incoming
            .get(u128::from(target))
            .cloned()
            .unwrap_or_default();
        incoming.push(delta);
        self.incoming.insert(u128::from(target), incoming);
    }
}

impl Adjacency {
    pub fn from_csr(outgoing: Csr, incoming: Csr) -> Self {
        Self {
            outgoing,
            incoming,
            deltas: PagedVec::default(),
            delta_rows: OnceLock::new(),
        }
    }

    #[must_use]
    pub fn outgoing(&self) -> &Csr {
        &self.outgoing
    }

    #[must_use]
    pub fn incoming(&self) -> &Csr {
        &self.incoming
    }

    #[must_use]
    pub fn deltas(&self) -> Vec<AdjacencyDelta> {
        self.deltas.to_vec()
    }

    #[must_use]
    pub fn delta_len(&self) -> usize {
        self.deltas.len()
    }

    pub fn insert(&mut self, source: u32, target: u32, edge: u32) {
        let delta = AdjacencyDelta::Insert {
            source,
            target,
            edge,
        };
        self.append_delta(delta);
    }

    pub fn delete(&mut self, source: u32, target: u32, edge: u32) {
        let delta = AdjacencyDelta::Delete {
            source,
            target,
            edge,
        };
        self.append_delta(delta);
    }

    fn append_delta(&mut self, delta: AdjacencyDelta) {
        let existing = self.delta_rows.get().cloned();
        self.deltas.push(delta);
        let mut rows = existing
            .as_deref()
            .cloned()
            .unwrap_or_else(|| Self::build_delta_rows(&self.deltas));
        if existing.is_some() {
            rows.append(delta);
        }
        self.delta_rows = OnceLock::new();
        let _ = self.delta_rows.set(Arc::new(rows));
    }

    fn build_delta_rows(deltas: &PagedVec<AdjacencyDelta>) -> DeltaRows {
        let mut rows = DeltaRows::default();
        for delta in deltas.iter().copied() {
            rows.append(delta);
        }
        rows
    }

    fn delta_rows(&self) -> &DeltaRows {
        self.delta_rows
            .get_or_init(|| Arc::new(Self::build_delta_rows(&self.deltas)))
    }

    /// Returns a deterministic merged outgoing row without rebuilding the base.
    pub fn expand_out(&self, source: u32, output: &mut Vec<(u32, u32)>) {
        output.clear();
        let mut deleted = BTreeSet::new();
        let mut inserted = BTreeSet::new();
        for delta in self
            .delta_rows()
            .outgoing
            .get(u128::from(source))
            .map(PagedVec::iter)
            .into_iter()
            .flatten()
        {
            match *delta {
                AdjacencyDelta::Insert {
                    source: row,
                    target,
                    edge,
                } if row == source => {
                    inserted.insert((target, edge));
                    deleted.remove(&(target, edge));
                }
                AdjacencyDelta::Delete {
                    source: row,
                    target,
                    edge,
                } if row == source => {
                    inserted.remove(&(target, edge));
                    deleted.insert((target, edge));
                }
                _ => {}
            }
        }
        if inserted.is_empty() && deleted.is_empty() {
            // No pending deltas for this row: the base CSR row is already emitted in sorted,
            // de-duplicated (target, edge) order by `Csr::build`, so copy it directly and skip the
            // O(d log d) sort+dedup. This is the hot path once adjacency is compacted.
            if let Some(base) = self.outgoing.row(source) {
                output.extend(base);
            }
            return;
        }
        if let Some(base) = self.outgoing.row(source) {
            output.extend(base.filter(|entry| !deleted.contains(entry)));
        }
        output.extend(inserted);
        output.sort_unstable();
        output.dedup();
    }

    /// Returns at most `limit` entries from one deterministically merged outgoing row.
    pub fn expand_out_bounded(&self, source: u32, limit: usize, output: &mut Vec<(u32, u32)>) {
        output.clear();
        if limit == 0 {
            return;
        }
        let mut deleted = BTreeSet::new();
        let mut inserted = BTreeSet::new();
        for delta in self
            .delta_rows()
            .outgoing
            .get(u128::from(source))
            .map(PagedVec::iter)
            .into_iter()
            .flatten()
        {
            match *delta {
                AdjacencyDelta::Insert {
                    source: row,
                    target,
                    edge,
                } if row == source => {
                    inserted.insert((target, edge));
                    deleted.remove(&(target, edge));
                }
                AdjacencyDelta::Delete {
                    source: row,
                    target,
                    edge,
                } if row == source => {
                    inserted.remove(&(target, edge));
                    deleted.insert((target, edge));
                }
                _ => {}
            }
        }
        merge_bounded_row(
            self.outgoing
                .row(source)
                .into_iter()
                .flatten()
                .filter(|entry| !deleted.contains(entry)),
            inserted.into_iter(),
            limit,
            output,
        );
    }

    /// Returns a deterministic merged incoming row without rebuilding the base.
    pub fn expand_in(&self, target: u32, output: &mut Vec<(u32, u32)>) {
        output.clear();
        let mut deleted = BTreeSet::new();
        let mut inserted = BTreeSet::new();
        for delta in self
            .delta_rows()
            .incoming
            .get(u128::from(target))
            .map(PagedVec::iter)
            .into_iter()
            .flatten()
        {
            match *delta {
                AdjacencyDelta::Insert {
                    source,
                    target: row,
                    edge,
                } if row == target => {
                    inserted.insert((source, edge));
                    deleted.remove(&(source, edge));
                }
                AdjacencyDelta::Delete {
                    source,
                    target: row,
                    edge,
                } if row == target => {
                    inserted.remove(&(source, edge));
                    deleted.insert((source, edge));
                }
                _ => {}
            }
        }
        if inserted.is_empty() && deleted.is_empty() {
            // No pending deltas: the base CSC row is already sorted and de-duplicated; copy directly.
            if let Some(base) = self.incoming.row(target) {
                output.extend(base);
            }
            return;
        }
        if let Some(base) = self.incoming.row(target) {
            output.extend(base.filter(|entry| !deleted.contains(entry)));
        }
        output.extend(inserted);
        output.sort_unstable();
        output.dedup();
    }

    /// Returns at most `limit` entries from one deterministically merged incoming row.
    pub fn expand_in_bounded(&self, target: u32, limit: usize, output: &mut Vec<(u32, u32)>) {
        output.clear();
        if limit == 0 {
            return;
        }
        let mut deleted = BTreeSet::new();
        let mut inserted = BTreeSet::new();
        for delta in self
            .delta_rows()
            .incoming
            .get(u128::from(target))
            .map(PagedVec::iter)
            .into_iter()
            .flatten()
        {
            match *delta {
                AdjacencyDelta::Insert {
                    source,
                    target: row,
                    edge,
                } if row == target => {
                    inserted.insert((source, edge));
                    deleted.remove(&(source, edge));
                }
                AdjacencyDelta::Delete {
                    source,
                    target: row,
                    edge,
                } if row == target => {
                    inserted.remove(&(source, edge));
                    deleted.insert((source, edge));
                }
                _ => {}
            }
        }
        merge_bounded_row(
            self.incoming
                .row(target)
                .into_iter()
                .flatten()
                .filter(|entry| !deleted.contains(entry)),
            inserted.into_iter(),
            limit,
            output,
        );
    }

    /// Rebuilds CSR and CSC at a snapshot boundary and clears applied deltas.
    pub fn compact(&mut self, node_count: usize, visible_edges: &[(u32, u32, u32)]) -> Result<()> {
        self.outgoing = Csr::build(node_count, visible_edges)?;
        self.incoming = Csr::build_transposed(node_count, visible_edges)?;
        self.deltas = PagedVec::default();
        self.delta_rows = OnceLock::new();
        Ok(())
    }
}

fn merge_bounded_row(
    base: impl Iterator<Item = (u32, u32)>,
    inserted: impl Iterator<Item = (u32, u32)>,
    limit: usize,
    output: &mut Vec<(u32, u32)>,
) {
    let mut base = base.peekable();
    let mut inserted = inserted.peekable();
    while output.len() < limit {
        let next = match (base.peek().copied(), inserted.peek().copied()) {
            (Some(left), Some(right)) if left < right => base.next(),
            (Some(left), Some(right)) if right < left => inserted.next(),
            (Some(_), Some(_)) => {
                let value = base.next();
                inserted.next();
                value
            }
            (Some(_), None) => base.next(),
            (None, Some(_)) => inserted.next(),
            (None, None) => None,
        };
        let Some(next) = next else {
            break;
        };
        if output.last().copied() != Some(next) {
            output.push(next);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn shared_offset_extension_validates_only_the_same_allocation_tail() -> Result<()> {
        let old_rows = 1_000_000_usize;
        let appended = 256_usize;
        let offsets: Arc<[u32]> = (0..=old_rows + appended)
            .map(|row| u32::try_from(row).expect("test row fits u32"))
            .collect::<Vec<_>>()
            .into();
        let full_offsets =
            SharedFlat::from_arc_slice(Arc::clone(&offsets)).expect("shared offsets");
        let old_offsets = full_offsets
            .slice(0, old_rows + 1)
            .expect("shared offset prefix");
        let neighbors: Arc<[u32]> = (0..old_rows + appended)
            .map(|row| u32::try_from(row).expect("test row fits u32"))
            .collect::<Vec<_>>()
            .into();
        let shared_neighbors =
            SharedFlat::from_arc_slice(Arc::clone(&neighbors)).expect("shared neighbors");
        let shared_edges = SharedFlat::from_arc_slice(neighbors).expect("shared edges");
        let mut csr = Csr::from_shared(
            old_offsets,
            shared_neighbors
                .slice(0, old_rows)
                .expect("shared neighbor prefix"),
            shared_edges.slice(0, old_rows).expect("shared edge prefix"),
        )?;

        // Keep payload cardinality fixed for this synthetic extension; only empty rows are legal.
        let empty_tail: Arc<[u32]> = std::iter::once(0)
            .chain(std::iter::repeat_n(
                u32::try_from(old_rows).expect("test row fits u32"),
                old_rows + appended,
            ))
            .collect::<Vec<_>>()
            .into();
        let full_empty = SharedFlat::from_arc_slice(empty_tail).expect("shared empty offsets");
        let old_empty = full_empty
            .slice(0, old_rows + 1)
            .expect("shared empty offset prefix");
        csr.offsets = FlatColumn::from_shared(old_empty);
        csr.rebase_shared_offsets_extension(full_empty)?;

        assert_eq!(csr.offsets().len(), old_rows + appended + 1);
        assert_eq!(csr.offsets()[old_rows], old_rows as u32);
        assert_eq!(csr.offsets().last().copied(), Some(old_rows as u32));

        // A byte-identical but separately allocated prefix is not trusted as an extension.
        let replacement: Arc<[u32]> = csr.offsets().to_vec().into();
        assert!(
            csr.rebase_shared_offsets_extension(
                SharedFlat::from_arc_slice(replacement).expect("replacement offsets")
            )
            .is_err()
        );
        let _ = offsets;
        Ok(())
    }

    #[test]
    fn extending_rows_matches_a_full_rebuild_without_touching_the_edge_arrays() -> Result<()> {
        // Appending nodes with no adjacency is the common shape of a `CREATE (n)` commit. It used
        // to fall through to a rebuild that re-materialized and re-sorted every edge in the
        // project, so a single edgeless insert cost time and memory proportional to the whole
        // graph. Extending must produce exactly what that rebuild produced.
        let triples = (0..64_u32)
            .flat_map(|source| {
                (0..4_u32).map(move |step| (source, (source + step) % 64, source * 4 + step))
            })
            .collect::<Vec<_>>();
        let mut extended = Csr::build(64, &triples)?;
        let neighbors_before = extended.neighbors().to_vec();
        let edges_before = extended.edges().to_vec();

        extended.extend_rows(96)?;
        let rebuilt = Csr::build(96, &triples)?;

        assert_eq!(extended.offsets(), rebuilt.offsets());
        assert_eq!(extended.neighbors(), rebuilt.neighbors());
        assert_eq!(extended.edges(), rebuilt.edges());
        // The edge arrays are shared, not rewritten: only the offset array grows.
        assert_eq!(extended.neighbors(), neighbors_before.as_slice());
        assert_eq!(extended.edges(), edges_before.as_slice());
        assert_eq!(extended.offsets().len(), 97);

        // Every appended row is empty, and every original row still reads back unchanged.
        for node in 64..96_u32 {
            assert_eq!(extended.row(node).map(Iterator::count), Some(0));
        }
        for node in 0..64_u32 {
            let original = Csr::build(64, &triples)?;
            assert_eq!(
                extended.row(node).map(Iterator::collect::<Vec<_>>),
                original.row(node).map(Iterator::collect::<Vec<_>>)
            );
        }

        // Extending to the size it already has is a no-op, and shrinking is refused rather than
        // silently discarding rows.
        extended.extend_rows(96)?;
        assert_eq!(extended.offsets().len(), 97);
        assert!(extended.extend_rows(32).is_err());
        Ok(())
    }

    #[test]
    fn replacing_sparse_rows_matches_a_canonical_full_rebuild() -> Result<()> {
        let original = vec![
            (0, 1, 0),
            (0, 2, 1),
            (1, 2, 2),
            (2, 3, 3),
            (3, 4, 4),
            (4, 0, 5),
        ];
        let csr = Csr::build(5, &original)?;
        let row_zero_neighbors = [3_u32];
        let row_zero_edges = [6_u32];
        let row_three_neighbors = [0_u32, 2_u32];
        let row_three_edges = [7_u32, 8_u32];
        let appended_neighbors = [1_u32];
        let appended_edges = [9_u32];
        let replacements = BTreeMap::from([
            (
                0,
                (row_zero_neighbors.as_slice(), row_zero_edges.as_slice()),
            ),
            (
                3,
                (row_three_neighbors.as_slice(), row_three_edges.as_slice()),
            ),
            (
                5,
                (appended_neighbors.as_slice(), appended_edges.as_slice()),
            ),
        ]);

        let replaced = csr.replace_rows(6, &replacements)?;
        let expected = Csr::build(
            6,
            &[
                (0, 3, 6),
                (1, 2, 2),
                (2, 3, 3),
                (3, 0, 7),
                (3, 2, 8),
                (4, 0, 5),
                (5, 1, 9),
            ],
        )?;

        assert_eq!(replaced.offsets(), expected.offsets());
        assert_eq!(replaced.neighbors(), expected.neighbors());
        assert_eq!(replaced.edges(), expected.edges());
        Ok(())
    }

    #[test]
    fn bounded_expansion_stops_at_requested_high_degree_prefix() -> Result<()> {
        let triples = (1..=10_000_u32)
            .map(|target| (0, target, target - 1))
            .collect::<Vec<_>>();
        let mut adjacency = Adjacency::default();
        adjacency.compact(10_001, &triples)?;
        adjacency.delete(0, 1, 0);
        adjacency.insert(0, 1, 10_001);
        let mut output = Vec::new();
        adjacency.expand_out_bounded(0, 7, &mut output);
        assert_eq!(output.len(), 7);
        assert_eq!(output[0], (1, 10_001));
        assert_eq!(output[6], (7, 6));
        Ok(())
    }

    #[test]
    fn frontier_distances_match_bfs_levels_and_respect_hop_bounds() -> Result<()> {
        // Chain 0->1->2->3->4 with a 0->2 shortcut and a 4->4 self-loop.
        let triples = vec![
            (0, 1, 0),
            (1, 2, 1),
            (2, 3, 2),
            (3, 4, 3),
            (0, 2, 4),
            (4, 4, 5),
        ];
        let csr = Csr::build(5, &triples)?;

        // Single-source shortest-hop distances (the shortcut puts node 2 at hop 1).
        let distance = csr.multi_source_frontier_distances([0], None, || Ok(()))?;
        assert_eq!(distance, vec![Some(0), Some(1), Some(1), Some(2), Some(3)]);

        // A hop bound stops the frontier early.
        let bounded = csr.multi_source_frontier_distances([0], Some(1), || Ok(()))?;
        assert_eq!(bounded, vec![Some(0), Some(1), Some(1), None, None]);

        // Multi-source seeds advance every seed's frontier simultaneously.
        let multi = csr.multi_source_frontier_distances([0, 3], None, || Ok(()))?;
        assert_eq!(multi[0], Some(0));
        assert_eq!(multi[3], Some(0));
        assert_eq!(multi[4], Some(1));

        // Reachable set filtered by minimum hops.
        let mut reachable = csr.multi_source_reachable([0], 1, None, || Ok(()))?;
        reachable.sort_unstable();
        assert_eq!(reachable, vec![1, 2, 3, 4]);
        let mut deep = csr.multi_source_reachable([0], 2, None, || Ok(()))?;
        deep.sort_unstable();
        assert_eq!(deep, vec![3, 4]);
        Ok(())
    }

    #[test]
    fn frontier_reachability_terminates_on_cycles() -> Result<()> {
        // A 3-cycle plus a tail: 0->1->2->0 and 2->3.
        let csr = Csr::build(4, &[(0, 1, 0), (1, 2, 1), (2, 0, 2), (2, 3, 3)])?;
        let distance = csr.multi_source_frontier_distances([0], None, || Ok(()))?;
        // Every node is reached exactly once at its shortest hop; the cycle does not loop forever.
        assert_eq!(distance, vec![Some(0), Some(1), Some(2), Some(3)]);
        Ok(())
    }
}
