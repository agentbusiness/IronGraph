//! Standalone state-backend contract: the `MutationStateBackend` trait the write engine applies
//! mutations through, and its supporting types.

use std::path::Path;

use async_trait::async_trait;
use uuid::Uuid;

use crate::storage::MutationEntry;
use crate::{Bookmark, Error, Result};

use super::{BackendSnapshot, WriteCommand};

/// Hard resource limits for the durable write layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteStorageLimits {
    pub max_log_record_bytes: usize,
    pub max_log_entries_per_read: usize,
    pub max_snapshot_bytes: usize,
}

/// Verified export of the canonical standalone snapshot. The checksum covers the complete framed
/// snapshot file, including its embedded application checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotBackup {
    pub bookmark: Bookmark,
    pub bytes: u64,
    pub checksum: [u8; 32],
}

impl WriteStorageLimits {
    pub(crate) fn validate(self) -> Result<Self> {
        if self.max_log_record_bytes == 0
            || self.max_log_entries_per_read == 0
            || self.max_snapshot_bytes == 0
        {
            return Err(Error::invalid_data("write storage limits must be non-zero"));
        }
        Ok(self)
    }
}

/// Result of applying a deterministic mutation to the canonical local backend.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MutationApplyResult {
    pub response: Vec<u8>,
    pub duplicate: bool,
}

/// Ephemeral local-sequencer ownership for one state-dependent command admission. A pipelined
/// reservation represents the command's effect in the backend's ordered pending-state overlay;
/// it is never serialized into the WAL or a snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandReservation {
    token: Option<Uuid>,
    position: Bookmark,
    pipelined: bool,
}

impl CommandReservation {
    #[must_use]
    pub const fn serialized(position: Bookmark) -> Self {
        Self {
            token: None,
            position,
            pipelined: false,
        }
    }

    pub fn owned_serialized(position: Bookmark) -> Result<Self> {
        if position.term == 0 {
            return Err(Error::invalid_data(
                "owned command reservation has no write term",
            ));
        }
        Ok(Self {
            token: Some(Uuid::new_v4()),
            position,
            pipelined: false,
        })
    }

    pub fn pipelined(position: Bookmark) -> Result<Self> {
        if position.term == 0 {
            return Err(Error::invalid_data(
                "pipelined command reservation has no write term",
            ));
        }
        Ok(Self {
            token: Some(Uuid::new_v4()),
            position,
            pipelined: true,
        })
    }

    #[must_use]
    pub const fn position(self) -> Bookmark {
        self.position
    }

    #[must_use]
    pub const fn may_release_after_append(self) -> bool {
        self.pipelined
    }

    #[must_use]
    pub(crate) const fn token(self) -> Option<Uuid> {
        self.token
    }
}

/// Terminal disposition of one local command reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandReservationOutcome {
    Applied,
    Rejected,
}

/// Production state backend contract used by the local ordered write path.
///
/// `apply_mutation` atomically publishes one ordered mutation and is idempotent for an already
/// applied bookmark or request ID. The WAL and its snapshots are the sole recovery source;
/// implementations must not introduce a second durable mutation log.
#[async_trait]
pub trait MutationStateBackend: Send + Sync + 'static {
    /// Resolve sequencer-owned nondeterminism before state-dependent ordered reservation. A
    /// retried command is prepared again by the local sequencer that can
    /// actually append it; only the returned command is committed.
    async fn prepare_command(
        &self,
        mut command: WriteCommand,
        sequencer_time_millis: i64,
    ) -> Result<WriteCommand> {
        command.commit_time_millis = sequencer_time_millis;
        command.validate()?;
        Ok(command)
    }

    /// Validate a resolved command against the sequencer's current applied state before append.
    /// This is the serialized fallback for backends without an ordered pending-state overlay.
    async fn validate_command(&self, command: &WriteCommand) -> Result<()> {
        command.validate()
    }

    /// Atomically validates a prepared command against the latest ordered state and reserves its
    /// exact next log position. Backends without an ordered pending-state overlay return a
    /// serialized reservation, which keeps the runtime gate through local application.
    async fn reserve_command(
        &self,
        command: &WriteCommand,
        position: Bookmark,
    ) -> Result<CommandReservation> {
        self.validate_command(command).await?;
        Ok(CommandReservation::serialized(position))
    }

    /// Releases ephemeral state and resources owned by a reservation. `Applied` is delivered only
    /// after the local state machine returned the response at the reservation's exact position;
    /// `Rejected` means the local write path definitively rejected or discarded the command.
    async fn complete_command_reservation(
        &self,
        _reservation: CommandReservation,
        _outcome: CommandReservationOutcome,
    ) -> Result<()> {
        Ok(())
    }

    async fn apply_mutation(&self, mutation: &MutationEntry) -> Result<MutationApplyResult>;

    /// The bookmark (term, index) of the most recently applied mutation. The standalone write path
    /// reads this under the write gate to allocate the next ordered index directly — applying a
    /// mutation with `index = applied.index + 1`.
    async fn applied_bookmark(&self) -> Bookmark;

    /// Return an exact immutable snapshot through `bookmark`, never a later live view.
    async fn build_snapshot(
        &self,
        bookmark: Bookmark,
        destination: &Path,
    ) -> Result<BackendSnapshot>;

    /// Atomically install a verified snapshot and make repeated installation idempotent.
    async fn install_snapshot(&self, bookmark: Bookmark, snapshot: &BackendSnapshot) -> Result<()>;
}
