//! Compact, canonical, device-addressable labeled-property graph representation.

#![allow(
    clippy::collapsible_if,
    clippy::double_must_use,
    clippy::implied_bounds_in_impls,
    clippy::large_enum_variant,
    clippy::len_without_is_empty,
    clippy::manual_ignore_case_cmp,
    clippy::needless_range_loop,
    clippy::obfuscated_if_else,
    clippy::too_many_arguments,
    clippy::wrong_self_convention
)]

pub use irongraph_types::*;

mod adjacency;
mod algorithms;
mod columns;
mod index;
pub mod knowledge;
mod persistent;
mod shared;
mod statistics;
mod store;
mod temporal;

pub use adjacency::{Adjacency, AdjacencyDelta, Csr};
pub use algorithms::{
    Components, DijkstraResult, PageRankConfig, bfs, bfs_cancellable, clustering_coefficients,
    clustering_coefficients_cancellable, dfs, dfs_cancellable, dijkstra, dijkstra_cancellable,
    k_core, k_core_cancellable, louvain_communities, louvain_communities_cancellable, page_rank,
    page_rank_cancellable, shortest_path, shortest_path_cancellable, strongly_connected_components,
    strongly_connected_components_cancellable, triangle_count, triangle_count_cancellable,
    weakly_connected_components, weakly_connected_components_cancellable,
};
pub use columns::MIXED_STRING_TAG;
pub use columns::{ByteValues, Dictionary, PackedLists, PropertyColumns, TypedColumn, Validity};
pub use index::SharedVectorBacking;
pub use index::{
    AnnDeviceImage, DerivedIndexDeviceImage, DerivedIndexState, EmbeddingDType,
    EmbeddingIndexDefinition, EmbeddingProfile, EqualityIndex, GraphIndexDefinition,
    GraphIndexKind, IVF_PQ_ASSIGNMENT_TILE_BYTES, IVF_PQ_BUILD_BATCH_ROWS,
    IVF_PQ_MIN_RECALL_BASIS_POINTS, IVF_PQ_SIZE_CLASS_VERSION, IndexCatalog, IndexDeviceImage,
    IndexKey, IndexStatus, IvfPqBuildKernel, IvfPqBuildPlan, IvfPqConfig, IvfPqIndex,
    OptimizerIndexStatistics, PostingDeviceImage, RangeIndex, ResolvedVectorMutation,
    ScalarIndexCandidate, Similarity, TextDeviceImage, TextIndex, VectorDeviceImage, VectorHit,
    VectorIndex,
};
pub use persistent::PagedVec;
pub use persistent::PersistentMap;
pub use persistent::{stable_id_key, stable_id_row};
#[allow(unused_imports)]
pub use shared::{SharedAllocation, SharedFlat};
pub use statistics::{FanoutStatistics, NumericHistogram, PropertyStatistics, StatisticsSnapshot};
pub use store::GraphSharedBacking;
pub use store::{
    AdjacencyRowDeviceDelta, CompactionMap, EdgeDeviceDelta, EdgeInput, EdgeView, GraphChangeIds,
    GraphDeviceDelta, GraphMutation, GraphSnapshot, GraphStore, NameCatalog, NodeDeviceDelta,
    NodeInput, NodeView,
};
pub use temporal::TemporalCanonicalColumn;
pub use temporal::{
    AggregateSet, RollupBucket, RollupDeviceColumn, TemporalDeclaration, TemporalDeviceColumn,
    TemporalDeviceImage, TemporalDeviceValues, TemporalOptimizerColumnStatistics,
    TemporalRollupDefinition, TemporalSample, TemporalStore, TemporalType, WindowKind, WindowSpec,
};

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

bitflags! {
    /// Query-visible physical graph layers.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct LayerMask: u8 {
        const OBSERVED = 1 << (Layer::Observed as u8);
        const KNOWLEDGE = 1 << (Layer::Knowledge as u8);
        const WORKSPACE = 1 << (Layer::Workspace as u8);
        /// The default authority layers. Deliberately excludes WORKSPACE.
        const AUTHORITY = Self::OBSERVED.bits() | Self::KNOWLEDGE.bits();
        const ALL = Self::AUTHORITY.bits() | Self::WORKSPACE.bits();
    }
}

impl Default for LayerMask {
    fn default() -> Self {
        Self::AUTHORITY
    }
}

impl LayerMask {
    /// Returns whether the mask contains a record's single physical layer.
    #[must_use]
    pub const fn contains_layer(self, layer: Layer) -> bool {
        self.bits() & (1_u8 << (layer as u8)) != 0
    }
}

#[cfg(test)]
mod layer_mask_tests {
    use super::LayerMask;
    use crate::Layer;

    #[test]
    fn authority_excludes_workspace_but_all_includes_it() {
        // The isolation invariant: AUTHORITY must never see Workspace.
        assert!(LayerMask::AUTHORITY.contains_layer(Layer::Observed));
        assert!(LayerMask::AUTHORITY.contains_layer(Layer::Knowledge));
        assert!(!LayerMask::AUTHORITY.contains_layer(Layer::Workspace));
        assert!(LayerMask::ALL.contains_layer(Layer::Workspace));
        assert_eq!(LayerMask::default(), LayerMask::AUTHORITY);
    }

    #[test]
    fn workspace_layer_byte_round_trips() {
        assert_eq!(
            Layer::try_from(2).expect("byte 2 is Workspace"),
            Layer::Workspace
        );
        assert_eq!(Layer::Workspace as u8, 2);
    }
}
