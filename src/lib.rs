//! Thin composition facade for the standalone `IronGraph` binary.

pub use irongraph_client as client;
pub use irongraph_embedded as embedded;
pub use irongraph_server::*;

pub use irongraph_types::{
    Bookmark, CommitAcknowledgement, DocumentItem, DocumentList, DocumentMap, EdgeId, Error,
    ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
};
