//! Backend-neutral typed execution values and columnar result batches.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use irongraph_types::{
    Bookmark, DocumentItem, DocumentList, DocumentMap, EdgeId, Error, ErrorCode, Layer, NodeId,
    Result, ScalarValue,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnType {
    Any,
    Null,
    Boolean,
    Integer,
    Float,
    String,
    Bytes,
    Temporal,
    Duration,
    Node,
    Relationship,
    Path,
    Vector,
    List,
    Map,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultNode {
    pub id: NodeId,
    pub layer: Layer,
    pub revision: u64,
    pub labels: Vec<String>,
    pub properties: BTreeMap<String, ScalarValue>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultEdge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub relationship_type: String,
    pub layer: Layer,
    pub revision: u64,
    pub properties: BTreeMap<String, ScalarValue>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ResultValue {
    Scalar(ScalarValue),
    Node(ResultNode),
    Relationship(ResultEdge),
    Path {
        nodes: Vec<ResultNode>,
        relationships: Vec<ResultEdge>,
    },
    Vector(Vec<f32>),
    List(Vec<ResultValue>),
    Map(BTreeMap<String, ResultValue>),
}

impl ResultValue {
    #[must_use]
    pub fn column_type(&self) -> ColumnType {
        match self {
            Self::Scalar(value) => match value {
                ScalarValue::Null => ColumnType::Null,
                ScalarValue::Boolean(_) => ColumnType::Boolean,
                ScalarValue::Integer(_) => ColumnType::Integer,
                ScalarValue::Float(_) => ColumnType::Float,
                ScalarValue::String(_) => ColumnType::String,
                ScalarValue::Bytes(_) => ColumnType::Bytes,
                ScalarValue::Date(_)
                | ScalarValue::LocalTime(_)
                | ScalarValue::ZonedTime { .. }
                | ScalarValue::LocalDateTime { .. }
                | ScalarValue::ZonedDateTime { .. } => ColumnType::Temporal,
                ScalarValue::Duration { .. } => ColumnType::Duration,
                ScalarValue::List(_) => ColumnType::List,
                ScalarValue::Map(_) => ColumnType::Map,
            },
            Self::Node(_) => ColumnType::Node,
            Self::Relationship(_) => ColumnType::Relationship,
            Self::Path { .. } => ColumnType::Path,
            Self::Vector(_) => ColumnType::Vector,
            Self::List(_) => ColumnType::List,
            Self::Map(_) => ColumnType::Map,
        }
    }

    /// Materializes a canonical property document only at the query result boundary.
    pub fn from_property(value: ScalarValue) -> Result<Self> {
        match value {
            ScalarValue::List(value) => Ok(Self::List(
                value
                    .items()?
                    .into_iter()
                    .map(document_item_to_result)
                    .collect::<Result<_>>()?,
            )),
            ScalarValue::Map(value) => Ok(Self::Map(
                value
                    .entries()?
                    .into_iter()
                    .map(|(name, value)| Ok((name.to_string(), document_item_to_result(value)?)))
                    .collect::<Result<_>>()?,
            )),
            value => Ok(Self::Scalar(value)),
        }
    }

    /// Converts a Cypher list/map mutation value into one canonical flat property document.
    pub fn into_property(self) -> Result<ScalarValue> {
        match self {
            Self::Scalar(value) => Ok(value),
            Self::List(values) => Ok(ScalarValue::List(DocumentList::new(
                values
                    .into_iter()
                    .map(result_to_document_item)
                    .collect::<Result<_>>()?,
            )?)),
            Self::Map(values) => Ok(ScalarValue::Map(DocumentMap::new(
                values
                    .into_iter()
                    .map(|(name, value)| Ok((name.into(), result_to_document_item(value)?)))
                    .collect::<Result<_>>()?,
            )?)),
            Self::Node(_) | Self::Relationship(_) | Self::Path { .. } | Self::Vector(_) => {
                Err(Error::new(
                    ErrorCode::QueryType,
                    "property documents may contain only scalar, list, and map values",
                ))
            }
        }
    }
}

fn document_item_to_result(value: DocumentItem) -> Result<ResultValue> {
    match value {
        DocumentItem::Scalar(value) => ResultValue::from_property(value),
        DocumentItem::List(values) => Ok(ResultValue::List(
            values
                .into_iter()
                .map(document_item_to_result)
                .collect::<Result<_>>()?,
        )),
        DocumentItem::Map(values) => Ok(ResultValue::Map(
            values
                .into_iter()
                .map(|(name, value)| Ok((name.to_string(), document_item_to_result(value)?)))
                .collect::<Result<_>>()?,
        )),
    }
}

fn result_to_document_item(value: ResultValue) -> Result<DocumentItem> {
    match value {
        ResultValue::Scalar(value) if value.is_document() => {
            match ResultValue::from_property(value)? {
                ResultValue::List(values) => Ok(DocumentItem::List(
                    values
                        .into_iter()
                        .map(result_to_document_item)
                        .collect::<Result<_>>()?,
                )),
                ResultValue::Map(values) => Ok(DocumentItem::Map(
                    values
                        .into_iter()
                        .map(|(name, value)| Ok((name.into(), result_to_document_item(value)?)))
                        .collect::<Result<_>>()?,
                )),
                _ => Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "document property decoded to a scalar root",
                )),
            }
        }
        ResultValue::Scalar(value) => Ok(DocumentItem::Scalar(value)),
        ResultValue::List(values) => Ok(DocumentItem::List(
            values
                .into_iter()
                .map(result_to_document_item)
                .collect::<Result<_>>()?,
        )),
        ResultValue::Map(values) => Ok(DocumentItem::Map(
            values
                .into_iter()
                .map(|(name, value)| Ok((name.into(), result_to_document_item(value)?)))
                .collect::<Result<_>>()?,
        )),
        ResultValue::Node(_)
        | ResultValue::Relationship(_)
        | ResultValue::Path { .. }
        | ResultValue::Vector(_) => Err(Error::new(
            ErrorCode::QueryType,
            "property documents may contain only scalar, list, and map values",
        )),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultColumn {
    pub name: String,
    pub value_type: ColumnType,
    pub values: Vec<ResultValue>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultBatch {
    pub row_count: usize,
    pub columns: Vec<ResultColumn>,
}

/// Incremental execution output. A synchronous consumer provides natural transport
/// backpressure; batches are not retained by the query engine after the callback returns.
#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionStreamItem {
    Schema(Vec<(String, ColumnType)>),
    Batch(ResultBatch),
}

impl ResultBatch {
    #[must_use]
    pub fn validate(&self) -> bool {
        self.columns
            .iter()
            .all(|column| column.values.len() == self.row_count)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatementStats {
    pub nodes_created: u64,
    pub nodes_deleted: u64,
    pub relationships_created: u64,
    pub relationships_deleted: u64,
    pub properties_set: u64,
    pub labels_added: u64,
    pub labels_removed: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QueryResult {
    pub schema: Vec<(String, ColumnType)>,
    pub batches: Vec<ResultBatch>,
    pub bookmark: Bookmark,
    pub statistics: StatementStats,
    pub truncated: bool,
}
