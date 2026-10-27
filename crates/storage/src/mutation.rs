use std::{
    io::Write,
    sync::{Arc, OnceLock},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use irongraph_types::{Bookmark, Error, ErrorCode, ProjectId, Result};

/// Resolved in-flight command category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum MutationKind {
    Graph = 0,
    Security = 2,
    Policy = 3,
    Broker = 4,
    Project = 5,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct MutationBody {
    term: u64,
    index: u64,
    kind: MutationKind,
    project_id: Option<ProjectId>,
    request_id: Option<Uuid>,
    commit_time_millis: i64,
    payload: Vec<u8>,
}

/// Resolved, backend-neutral in-flight command. It is never a durable journal record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationEntry {
    body: Arc<MutationBody>,
    checksum: [u8; 32],
    #[serde(skip)]
    checksum_verified: OnceLock<()>,
}

impl MutationEntry {
    pub fn new(
        term: u64,
        index: u64,
        kind: MutationKind,
        project_id: Option<ProjectId>,
        request_id: Option<Uuid>,
        commit_time_millis: i64,
        payload: Vec<u8>,
    ) -> Result<Self> {
        if term == 0 || index == 0 || payload.is_empty() {
            return Err(Error::invalid_data(
                "mutation term, index, and payload must be non-zero",
            ));
        }
        if matches!(
            kind,
            MutationKind::Graph | MutationKind::Broker | MutationKind::Project
        ) && project_id.is_none()
        {
            return Err(Error::invalid_data(
                "project-scoped mutation is missing project ID",
            ));
        }
        if matches!(kind, MutationKind::Security | MutationKind::Policy) && project_id.is_some() {
            return Err(Error::invalid_data(
                "store-wide mutation must not carry a project ID",
            ));
        }
        let body = MutationBody {
            term,
            index,
            kind,
            project_id,
            request_id,
            commit_time_millis,
            payload,
        };
        let checksum = body_checksum(&body)?;
        let checksum_verified = OnceLock::new();
        let _ = checksum_verified.set(());
        Ok(Self {
            body: Arc::new(body),
            checksum,
            checksum_verified,
        })
    }

    #[must_use]
    pub fn bookmark(&self) -> Bookmark {
        Bookmark {
            term: self.body.term,
            index: self.body.index,
        }
    }

    #[must_use]
    pub fn term(&self) -> u64 {
        self.body.term
    }

    #[must_use]
    pub fn index(&self) -> u64 {
        self.body.index
    }

    #[must_use]
    pub fn kind(&self) -> MutationKind {
        self.body.kind
    }

    #[must_use]
    pub fn project_id(&self) -> Option<ProjectId> {
        self.body.project_id
    }

    #[must_use]
    pub fn request_id(&self) -> Option<Uuid> {
        self.body.request_id
    }

    #[must_use]
    pub fn commit_time_millis(&self) -> i64 {
        self.body.commit_time_millis
    }

    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.body.payload
    }

    #[must_use]
    pub const fn checksum(&self) -> [u8; 32] {
        self.checksum
    }

    pub fn verify(&self) -> Result<()> {
        if self.checksum_verified.get().is_some() {
            return Ok(());
        }
        let calculated = body_checksum(&self.body)?;
        if calculated != self.checksum {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "mutation checksum mismatch",
            ));
        }
        let _ = self.checksum_verified.set(());
        Ok(())
    }
}

fn body_checksum(body: &MutationBody) -> Result<[u8; 32]> {
    struct HashWriter(blake3::Hasher);

    impl Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let writer = postcard::to_io(body, HashWriter(blake3::Hasher::new()))
        .map_err(|error| Error::new(ErrorCode::InvalidData, error.to_string()))?;
    Ok(*writer.0.finalize().as_bytes())
}
