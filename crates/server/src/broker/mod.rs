//! Stream and Queue messaging over one local WAL-backed broker state machine.
//!
//! Both surfaces are wire-compatible with an existing protocol so off-the-shelf clients work
//! unmodified — Stream speaks Kafka, Queue speaks AMQP 0-9-1 — but they are one system, not two:
//! a Stream topic partition and a Queue queue are both a `StreamId` over the same log, and message
//! payloads are stored once in the same content-addressed table. The product names are used
//! everywhere except where a wire format dictates otherwise.

mod engine;
mod memory;
mod queue;
mod stream;

pub use engine::{
    AmqpBatchRecord, AmqpExchangeKind, BrokerCommand, BrokerCommit, BrokerCoordinator, BrokerReply,
    BrokerStateMachine, ConsumerLagMetrics, Delivery, ExchangeMetrics, IngressMetadata,
    KafkaBatchRecord, PartitionRead, PayloadRecord, QueueInfo, QueueKind, QueueMetrics,
    RetentionPolicy, StreamOffset, TopicMetrics,
};
pub use queue::QueueServer;
pub use stream::StreamServer;
