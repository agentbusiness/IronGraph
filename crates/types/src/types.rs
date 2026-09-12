//! Compact types shared across storage, execution, and protocols.

use std::{fmt, sync::Arc};

use ordered_float::OrderedFloat;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer, de::Error as _, ser::SerializeSeq as _,
};
use uuid::Uuid;

macro_rules! u64_id {
    ($name:ident) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Default,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            Serialize,
            Deserialize,
        )]
        #[serde(transparent)]
        #[repr(transparent)]
        pub struct $name(pub u64);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

u64_id!(NodeId);
u64_id!(EdgeId);
u64_id!(PropertyId);
u64_id!(LabelId);
u64_id!(RelationshipTypeId);
u64_id!(CredentialId);
u64_id!(MessageId);
u64_id!(StreamId);

/// Immutable project identity. Display names are mutable catalog data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProjectId(pub Uuid);

impl ProjectId {
    /// Creates a cryptographically random project identifier.
    #[must_use]
    pub fn random() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for ProjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Position through which the standalone state machine has applied mutations.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Bookmark {
    /// Storage epoch containing the applied entry.
    pub term: u64,
    /// Monotonic local write index.
    pub index: u64,
}

impl fmt::Display for Bookmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.term, self.index)
    }
}

/// Wire-protocol acknowledgement shape. A standalone process applies every accepted write locally.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CommitAcknowledgement {
    #[default]
    Published,
}

/// Physical graph namespace encoded in exactly one byte.
///
/// `Workspace` is an isolated graph namespace. It is excluded from authority-layer reads unless a
/// query explicitly selects it with `USE LAYER WORKSPACE`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "UPPERCASE")]
pub enum Layer {
    #[default]
    Observed = 0,
    Knowledge = 1,
    Workspace = 2,
}

impl Layer {
    /// Every physical layer, in byte order. This is the one place the set is enumerated: a scan
    /// mask, a per-layer counter or a device tensor loop iterates this rather than listing layers
    /// itself, so adding a layer cannot leave one subsystem quietly seeing two.
    pub const ALL: [Self; 3] = [Self::Observed, Self::Knowledge, Self::Workspace];
}

impl TryFrom<u8> for Layer {
    type Error = crate::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Observed),
            1 => Ok(Self::Knowledge),
            2 => Ok(Self::Workspace),
            _ => Err(crate::Error::invalid_data("invalid layer byte")),
        }
    }
}

/// Node versus relationship target, generally carried by an enclosing segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EntityKind {
    Node = 0,
    Relationship = 1,
}

/// Canonical property value. LIST and MAP are flat non-temporal documents; every other variant
/// is a scalar accepted by the temporal engine when its declaration type matches.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ScalarValue {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(OrderedFloat<f64>),
    String(Arc<str>),
    Bytes(Arc<[u8]>),
    Date(i64),
    LocalTime(i64),
    ZonedTime {
        nanos: i64,
        offset_seconds: i32,
    },
    LocalDateTime {
        seconds: i64,
        nanos: u32,
    },
    ZonedDateTime {
        seconds: i64,
        nanos: u32,
        timezone: Arc<str>,
    },
    Duration {
        months: i64,
        days: i64,
        seconds: i64,
        nanos: i32,
    },
    /// Canonically encoded non-temporal Cypher list document.
    List(DocumentList),
    /// Canonically encoded non-temporal Cypher map document.
    Map(DocumentMap),
}

impl ScalarValue {
    /// Returns a stable type label used in query diagnostics and result schemas.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "NULL",
            Self::Boolean(_) => "BOOLEAN",
            Self::Integer(_) => "INTEGER",
            Self::Float(_) => "FLOAT",
            Self::String(_) => "STRING",
            Self::Bytes(_) => "BYTES",
            Self::Date(_) => "DATE",
            Self::LocalTime(_) => "LOCAL TIME",
            Self::ZonedTime { .. } => "ZONED TIME",
            Self::LocalDateTime { .. } => "LOCAL DATETIME",
            Self::ZonedDateTime { .. } => "ZONED DATETIME",
            Self::Duration { .. } => "DURATION",
            Self::List(_) => "LIST",
            Self::Map(_) => "MAP",
        }
    }

    /// Documents are legal graph properties but are never legal temporal samples.
    #[must_use]
    pub const fn is_document(&self) -> bool {
        matches!(self, Self::List(_) | Self::Map(_))
    }
}

/// Flat canonical bytes for one Cypher list document.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DocumentList(pub(crate) Arc<[u8]>);

/// Flat canonical bytes for one Cypher map document.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DocumentMap(pub(crate) Arc<[u8]>);

/// Reads canonical document bytes from either a sequence of byte elements or a byte string.
struct CanonicalDocumentBytes;

impl<'de> serde::de::Visitor<'de> for CanonicalDocumentBytes {
    type Value = Vec<u8>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("canonical document bytes")
    }

    fn visit_bytes<E: serde::de::Error>(self, value: &[u8]) -> std::result::Result<Self::Value, E> {
        Ok(value.to_vec())
    }

    fn visit_byte_buf<E: serde::de::Error>(
        self,
        value: Vec<u8>,
    ) -> std::result::Result<Self::Value, E> {
        Ok(value)
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut bytes = Vec::with_capacity(sequence.size_hint().unwrap_or_default());
        while let Some(byte) = sequence.next_element::<u8>()? {
            bytes.push(byte);
        }
        Ok(bytes)
    }
}

macro_rules! document_serde {
    ($name:ident, $root:expr) => {
        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                // A byte sequence, not a byte string. A length-prefixed encoding renders the two
                // identically, so every log record and snapshot already on disk keeps its exact
                // bytes. A self-describing encoding does not: written as a byte string, a document
                // is unreadable to the buffered read that an enum with a separate tag and content
                // field performs, and unreadable to a borrowed byte read once it outgrows the
                // decoder's fixed scratch space. Both applied here — the first rejected every list
                // and map property at commit, the second made any snapshot holding an embedding
                // vector unrecoverable. A sequence has neither limit at any length.
                let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
                for byte in self.0.iter() {
                    sequence.serialize_element(byte)?;
                }
                sequence.end()
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                // Reads the sequence written above. The visitor still accepts a byte string, so a
                // record from an encoding that folds the two shapes together decodes unchanged.
                let bytes = deserializer.deserialize_seq(CanonicalDocumentBytes)?;
                crate::document::validate_canonical(&bytes, $root).map_err(D::Error::custom)?;
                Ok(Self(bytes.into()))
            }
        }
    };
}

document_serde!(DocumentList, crate::document::DocumentRoot::List);
document_serde!(DocumentMap, crate::document::DocumentRoot::Map);
