//! Automatic local text embedding and verified encoder artifacts.

#![allow(
    clippy::collapsible_if,
    clippy::double_ended_iterator_last,
    clippy::iter_skip_next,
    clippy::large_enum_variant,
    clippy::manual_clamp,
    clippy::manual_is_multiple_of,
    clippy::needless_question_mark,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

mod device;
mod embedding;
#[cfg(feature = "accelerator")]
mod embedding_cache;
#[cfg(feature = "accelerator")]
mod encoder;
mod install;

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

    #[cfg(all(
        test,
        feature = "accelerator",
        any(target_os = "macos", target_os = "ios")
    ))]
    pub fn metal_test_device() -> Option<candle_core::Device> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            candle_core::Device::new_metal(0)
        }))
        .ok()
        .and_then(std::result::Result::ok)
    }
}
pub mod graph {
    pub use irongraph_graph::*;
}

pub use device::EmbeddingDevice;
pub use embedding::{EmbeddingModelArtifacts, LocalEmbeddingModel};
pub use install::{
    DEFAULT_EMBEDDING_CONFIG_BYTES, DEFAULT_EMBEDDING_CONFIG_FILE, DEFAULT_EMBEDDING_CONFIG_SHA256,
    DEFAULT_EMBEDDING_MODEL_BYTES, DEFAULT_EMBEDDING_MODEL_FILE, DEFAULT_EMBEDDING_MODEL_SHA256,
    DEFAULT_EMBEDDING_REVISION, DEFAULT_EMBEDDING_TOKENIZER_BYTES,
    DEFAULT_EMBEDDING_TOKENIZER_FILE, DEFAULT_EMBEDDING_TOKENIZER_SHA256,
    ensure_default_embedding_model,
};
pub use irongraph_types::{EdgeId, Error, ErrorCode, NodeId, ProjectId, Result};
