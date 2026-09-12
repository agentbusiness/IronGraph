use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    Bookmark, Error, ProjectId, Result,
    storage::{AdmissionClass, ConnectionId, MutationKind},
};

use super::{
    ApplicationWait, ExecutionClass, NodeIdentityPublic, ProcessId, StoreFingerprint, StoreId,
};

/// Local process identity and selected execution class.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StandaloneNode {
    pub identity: NodeIdentityPublic,
    pub execution_class: ExecutionClass,
}

impl StandaloneNode {
    pub fn validate_for(&self, node_id: ProcessId) -> Result<()> {
        self.identity.validate()?;
        if self.identity.node_id != node_id {
            return Err(Error::invalid_data(
                "standalone process identity does not match",
            ));
        }
        Ok(())
    }
}

impl Default for StandaloneNode {
    fn default() -> Self {
        Self {
            identity: NodeIdentityPublic {
                store_id: StoreId(uuid::Uuid::nil()),
                node_id: ProcessId(uuid::Uuid::nil()),
                signing_key: [0_u8; 32],
                store_fingerprint: StoreFingerprint([0_u8; 32]),
            },
            execution_class: ExecutionClass::Cpu,
        }
    }
}

/// Deterministic command submitted to the local write sequencer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteCommand {
    pub kind: MutationKind,
    pub project_id: Option<ProjectId>,
    pub request_id: Option<Uuid>,
    pub commit_time_millis: i64,
    pub payload: Vec<u8>,
}

impl WriteCommand {
    pub fn validate(&self) -> Result<()> {
        if self.payload.is_empty() {
            return Err(Error::invalid_data("write mutation payload is empty"));
        }
        if matches!(
            self.kind,
            MutationKind::Graph | MutationKind::Broker | MutationKind::Project
        ) && self.project_id.is_none()
        {
            return Err(Error::invalid_data(
                "project-scoped mutation is missing project ID",
            ));
        }
        if matches!(self.kind, MutationKind::Security | MutationKind::Policy)
            && self.project_id.is_some()
        {
            return Err(Error::invalid_data(
                "store-wide mutation carries a project ID",
            ));
        }
        Ok(())
    }
}

/// Application response returned only after the committed entry is durably applied locally.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteResponse {
    pub bookmark: Bookmark,
    pub request_id: Option<Uuid>,
    pub payload: Vec<u8>,
    pub duplicate: bool,
}

/// A write accepted by the standalone process and ordered by its local sequencer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteRequest {
    pub command: WriteCommand,
    pub timeout_millis: u64,
    /// Ephemeral origin used for fair in-memory admission at the receiving node and sequencer.
    /// It is never copied into the durable command.
    pub connection_id: ConnectionId,
    pub admission_class: AdmissionClass,
    /// Explicit transactions are bound to the sequencer term and applied snapshot observed at
    /// BEGIN. Autocommit and control writes leave this absent.
    pub transaction_fence: Option<TransactionFence>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionFence {
    pub sequencer: ProcessId,
    pub term: u64,
    pub snapshot: Bookmark,
}

/// Committed local write response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedWrite {
    pub response: WriteResponse,
    pub application: ApplicationWait,
}
