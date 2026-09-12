#![allow(clippy::all, clippy::nursery, clippy::pedantic)]

//! Consolidated external-protocol integration-test harness.
#[path = "../tests/bolt_driver_interop.rs"]
mod bolt_driver_interop;
#[path = "../tests/broker_harness/mod.rs"]
mod broker_harness;
#[path = "../tests/queue_interop.rs"]
mod queue_interop;
#[path = "../tests/stream_interop.rs"]
mod stream_interop;
