//! Transactional current-record persistence and bounded write scheduling.

#![allow(clippy::items_after_test_module)]

mod admission;
mod fs;
mod mutation;
mod records;
mod segment;

pub use admission::{
    AdmissionClass, AdmissionController, AdmissionLimits, AdmissionPermit, ConnectionId,
};
#[doc(hidden)]
pub use fs::{atomic_create, atomic_write, read_bounded, sync_durable};
pub use mutation::{MutationEntry, MutationKind};
pub use records::{RecordBatch, RecordOp, RecordStore};
pub use segment::{SegmentDescriptor, SegmentFamily, SegmentRecord, SegmentStore};
#[doc(hidden)]
pub use segment::{SegmentPin, SegmentRecordLocation};
