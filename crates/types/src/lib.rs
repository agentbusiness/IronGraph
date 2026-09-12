//! Foundational IronGraph identities, values, layers, bookmarks, and errors.

pub mod document;
pub mod error;
pub mod types;

pub use document::DocumentItem;
pub use error::{Error, ErrorCode, Result};
pub use types::{
    Bookmark, CommitAcknowledgement, CredentialId, DocumentList, DocumentMap, EdgeId, EntityKind,
    LabelId, Layer, MessageId, NodeId, ProjectId, PropertyId, RelationshipTypeId, ScalarValue,
    StreamId,
};
