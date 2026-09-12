//! Backend-neutral text-embedding contract.

use irongraph_graph::EmbeddingProfile;
use irongraph_types::Result;

pub trait TextEmbedding: Send + Sync {
    fn profile(&self) -> &EmbeddingProfile;
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text)
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }

    /// Produce overlapping, rebuildable index windows for one canonical value.
    ///
    /// The returned strings are never stored as graph records. Every vector made from them keeps
    /// the canonical node/edge as its owner, so retrieval resolves a window hit back to the one
    /// complete original object. Backends that do not need a shorter accelerated input retain the
    /// whole value.
    fn index_windows(&self, text: &str) -> Result<Vec<String>> {
        Ok(vec![text.to_owned()])
    }

    /// Produce overlapping retrieval windows for one user/context query. The full message remains
    /// untouched in the conversation; these windows seed semantic search independently.
    fn query_windows(&self, text: &str) -> Result<Vec<String>> {
        Ok(vec![text.to_owned()])
    }

    /// Encode a batch while allowing a caller-owned background operation to yield to interactive
    /// work. Implementations without a cancellable encoder retain the normal batch semantics.
    fn embed_batch_cancellable(
        &self,
        texts: &[String],
        _is_cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch(texts)
    }
}
