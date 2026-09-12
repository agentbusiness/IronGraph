//! Durable append-only storage, checkpoint publication, and bounded write admission.

#![allow(clippy::items_after_test_module)]

mod admission;
mod checkpoint;
mod cow;
mod frame;
mod fs;
mod log;
mod segment;

pub use admission::{
    AdmissionClass, AdmissionController, AdmissionLimits, AdmissionPermit, ConnectionId,
};
pub use checkpoint::{
    CheckpointManifest, CheckpointSource, CheckpointStore, ManifestId, RecoveredCheckpoint,
};
#[doc(hidden)]
pub use cow::CowArc;
pub use frame::{Durability, FramedFile, FramedRecord, RecoveryReport};
#[doc(hidden)]
pub use fs::{atomic_create, atomic_write, read_bounded, sync_durable};
pub use log::{DurableLog, MutationEntry, MutationKind};
pub use segment::{SegmentDescriptor, SegmentFamily, SegmentRecord, SegmentStore};
#[doc(hidden)]
pub use segment::{SegmentPin, SegmentRecordLocation};
