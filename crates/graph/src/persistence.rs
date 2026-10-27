//! Encoded persistence records, independent of the canonical in-memory representation.

use crate::{Error, ErrorCode, Result};
use serde::{Serialize, de::DeserializeOwned};

/// One current-state record update. Keys are relative to a project namespace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersistenceOp {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    DeletePrefix { prefix: Vec<u8> },
}

pub(crate) fn encode(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(value, &mut bytes).map_err(|error| {
        Error::invalid_data(format!("persistence record encoding failed: {error}"))
    })?;
    Ok(bytes)
}

pub(crate) fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let mut input = bytes;
    let value = ciborium::de::from_reader(&mut input)
        .map_err(|error| corrupt(format!("invalid persistence record: {error}")))?;
    if !input.is_empty() {
        return Err(corrupt("persistence record has trailing bytes"));
    }
    Ok(value)
}

pub(crate) fn put(key: Vec<u8>, value: &impl Serialize) -> Result<PersistenceOp> {
    Ok(PersistenceOp::Put {
        key,
        value: encode(value)?,
    })
}

pub(crate) fn corrupt(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::CorruptStorage, message.into())
}

pub(crate) fn key(prefix: &[u8], ids: &[u64]) -> Vec<u8> {
    let mut key = prefix.to_vec();
    for id in ids {
        key.extend_from_slice(&id.to_be_bytes());
    }
    key
}

pub(crate) fn ids(bytes: &[u8], count: usize) -> Result<Vec<u64>> {
    if bytes.len() != count * 8 {
        return Err(corrupt("invalid persistence record key"));
    }
    bytes
        .chunks_exact(8)
        .map(|part| {
            Ok(u64::from_be_bytes(
                part.try_into()
                    .map_err(|_| corrupt("invalid record identifier"))?,
            ))
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn apply_records(
    records: &mut std::collections::BTreeMap<Vec<u8>, Vec<u8>>,
    operations: Vec<PersistenceOp>,
) {
    for operation in operations {
        match operation {
            PersistenceOp::Put { key, value } => {
                records.insert(key, value);
            }
            PersistenceOp::Delete { key } => {
                records.remove(&key);
            }
            PersistenceOp::DeletePrefix { prefix } => {
                records.retain(|key, _| !key.starts_with(&prefix))
            }
        }
    }
}
