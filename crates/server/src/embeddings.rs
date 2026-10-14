//! Automatic local text-embedding integration.

pub use irongraph_embedding::{
    EmbeddingDevice, EmbeddingStorage, LocalEmbeddingModel, ModelInstallProgress,
    ensure_default_embedding_model, ensure_default_embedding_model_with_storage,
};
