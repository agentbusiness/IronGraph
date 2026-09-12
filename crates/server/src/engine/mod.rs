//! Standalone runtime: the direct-apply write engine, node identity, durable snapshot, and the
//! local stream and queue state.

mod activation;
mod backend;
mod bootstrap;
mod credential;
mod execution_class;
mod identity;
mod runtime;
mod security;
mod snapshot_data;
mod wait;
mod write_types;

pub use activation::{ActivationBarrier, ActivationSubject, EmbeddingProfileActivation};
pub use backend::{
    CommandReservation, CommandReservationOutcome, MutationApplyResult, MutationStateBackend,
    SnapshotBackup, WriteStorageLimits,
};
pub use bootstrap::{
    BootstrappedNode, SingleNodeBootstrapConfig, load_existing_node_identity,
    load_or_generate_genesis_identity, open_standalone,
};
pub use credential::{
    AuthorizedCredential, CredentialRecord, CredentialRegistry, LayerScope, OperationScope,
    ProtocolScope, certificate_ed25519_identity_key, certificate_public_key_fingerprint,
};
pub use execution_class::ExecutionClass;
pub use identity::{NodeIdentity, NodeIdentityPublic, ProcessId, StoreFingerprint, StoreId};
pub use runtime::{SequencerPosition, WriteRuntime};
pub use security::SecurityState;
pub use snapshot_data::{BackendSnapshot, SnapshotAttachment, SnapshotData};
pub use wait::ApplicationWait;
pub use write_types::{
    CommittedWrite, StandaloneNode, TransactionFence, WriteCommand, WriteRequest, WriteResponse,
};
