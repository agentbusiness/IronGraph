//! Standalone IronGraph database, protocols, broker, and automatic text embedding.

#![allow(
    clippy::await_holding_lock,
    clippy::cloned_ref_to_slice_refs,
    clippy::collapsible_if,
    clippy::collapsible_match,
    clippy::double_must_use,
    clippy::drop_non_drop,
    clippy::empty_line_after_doc_comments,
    clippy::explicit_auto_deref,
    clippy::items_after_test_module,
    clippy::large_enum_variant,
    clippy::manual_ignore_case_cmp,
    clippy::manual_slice_fill,
    clippy::needless_borrow,
    clippy::result_large_err,
    clippy::should_implement_trait,
    clippy::single_match,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_get_then_check,
    clippy::unwrap_or_default
)]

pub mod broker;
pub mod config;
pub mod embeddings;
#[allow(dead_code, unused_imports)]
pub mod engine;
pub mod protocol;
pub mod server;

pub mod cypher {
    pub use irongraph_cypher::*;
}
pub mod error {
    pub use irongraph_types::error::*;
}
pub mod execution {
    pub use irongraph_execution::*;
}
pub mod gpu {
    pub use irongraph_gpu::*;

    #[cfg(test)]
    pub fn metal_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static METAL_DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());
        match METAL_DEVICE.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}
pub mod graph {
    pub use irongraph_graph::*;
}
pub mod storage {
    pub use irongraph_storage::*;
}
pub mod types {
    pub use irongraph_types::types::*;
}

pub use irongraph_types::{
    Bookmark, CommitAcknowledgement, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result,
    ScalarValue,
};
