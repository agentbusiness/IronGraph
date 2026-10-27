//! Standalone state-backend contract: the `MutationStateBackend` trait the write engine applies
//! mutations through, and its supporting types.

use async_trait::async_trait;
use uuid::Uuid;

use crate::storage::{MutationEntry, RecordBatch};
use crate::{Bookmark, Error, Result};

use super::WriteCommand;

/// Result of applying a deterministic mutation to the canonical local backend.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MutationApplyResult {
    pub response: Vec<u8>,
    pub duplicate: bool,
    /// Current records captured at this mutation's exact applied position. The persistence
    /// worker owns these bytes after publication; it never reads a later live graph view.
    /// A newly applied position requires an explicit span, including for an empty metadata-only
    /// change. An already-applied entry may omit records when it advances no position.
    pub persistence: Option<RecordBatch>,
}

/// Ephemeral local-sequencer ownership for one state-dependent command admission. A pipelined
/// reservation represents the command's effect in the backend's ordered pending-state overlay;
/// it is never part of durable state.
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
    pub const fn may_pipeline(self) -> bool {
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
/// applied bookmark or request ID. Its captured record batches are the sole database recovery
/// source; implementations must not retain a durable mutation history alongside them.
#[async_trait]
pub trait MutationStateBackend: Send + Sync + 'static {
    /// Resolve sequencer-owned nondeterminism before state-dependent ordered reservation. A
    /// retried command is prepared again by the local sequencer; only the returned command is
    /// applied.
    async fn prepare_command(
        &self,
        mut command: WriteCommand,
        sequencer_time_millis: i64,
    ) -> Result<WriteCommand> {
        command.commit_time_millis = sequencer_time_millis;
        command.validate()?;
        Ok(command)
    }

    /// Validate a resolved command against the sequencer's current applied state before apply.
    /// This is the serialized fallback for backends without an ordered pending-state overlay.
    async fn validate_command(&self, command: &WriteCommand) -> Result<()> {
        command.validate()
    }

    /// Atomically validates a prepared command against the latest ordered state and reserves its
    /// exact next write position. Backends without an ordered pending-state overlay return a
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

    /// Commit captured current-record batches durably. Called only by the dedicated persistence
    /// thread, never by query readers or asynchronous runtime workers. Backends that return no
    /// persistence batches need no implementation (for example, in-memory test backends).
    fn persist(&self, _batches: &[RecordBatch]) -> Result<()> {
        Ok(())
    }
}
