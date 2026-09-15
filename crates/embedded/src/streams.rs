use std::collections::BTreeMap;

use irongraph_server::broker::{BrokerCommand, BrokerCoordinator, BrokerReply, KafkaBatchRecord};
use irongraph_types::{Bookmark, CommitAcknowledgement, Error, ErrorCode, ProjectId};
use serde::{Deserialize, Serialize};

use crate::{EmbeddedDatabase, OperationOptions, Result};

/// One native stream record. Null values and empty values remain distinct.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamRecord {
    pub key: Option<Vec<u8>>,
    #[serde(default)]
    pub headers: BTreeMap<String, Vec<u8>>,
    pub value: Option<Vec<u8>>,
    pub create_time_ms: Option<i64>,
}

/// Atomic append to one explicitly selected project/topic/partition.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamAppend {
    pub project_id: ProjectId,
    pub topic: String,
    pub partition: i32,
    pub records: Vec<StreamRecord>,
}

/// Published acknowledgement uses the same asynchronous WAL durability as graph writes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamAcknowledgement {
    pub bookmark: Bookmark,
    pub first_offset: u64,
    pub record_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamFetch {
    pub project_id: ProjectId,
    pub topic: String,
    pub partition: i32,
    pub offset: u64,
    pub max_records: usize,
    pub max_bytes: usize,
}

/// Each record retains its canonical message ID and original ingress metadata.
#[derive(Clone, Debug, Serialize)]
pub struct StreamPage {
    pub records: Vec<(u64, std::sync::Arc<irongraph_server::broker::PayloadRecord>)>,
    pub high_watermark: u64,
    pub next_offset: u64,
    pub truncated: bool,
}

impl EmbeddedDatabase {
    pub fn stream_append(
        &self,
        request: StreamAppend,
        options: OperationOptions,
    ) -> Result<StreamAcknowledgement> {
        let operation = self.begin_operation(options)?;
        operation.check()?;
        if request.records.is_empty() {
            return Err(Error::invalid_data("stream append requires at least one record").into());
        }
        let bytes = request.records.iter().fold(0usize, |total, record| {
            record.headers.iter().fold(
                total
                    .saturating_add(64)
                    .saturating_add(record.key.as_ref().map_or(0, Vec::len))
                    .saturating_add(record.value.as_ref().map_or(0, Vec::len)),
                |bytes, (name, value)| {
                    bytes
                        .saturating_add(16)
                        .saturating_add(name.len())
                        .saturating_add(value.len())
                },
            )
        });
        if bytes > self.options.max_write_bytes {
            return Err(Error::new(
                ErrorCode::Backpressure,
                "stream append exceeds the configured write budget before dispatch",
            )
            .into());
        }
        let core = self
            .core
            .as_ref()
            .ok_or_else(|| Error::invalid_data("database is closed"))?;
        let command = BrokerCommand::PublishKafkaBatch {
            project: request.project_id,
            topic: request.topic,
            partition: request.partition,
            resolved_time_ms: 0,
            records: request
                .records
                .into_iter()
                .map(|record| KafkaBatchRecord {
                    create_time_ms: record.create_time_ms,
                    key: record.key,
                    headers: record.headers,
                    value_is_null: record.value.is_none(),
                    payload: record.value.unwrap_or_default(),
                })
                .collect(),
        };
        operation.check()?;
        // After dispatch, wait for the authoritative outcome. Cancellation must not turn an
        // acknowledged append into a retryable error or fabricate a rollback.
        let timeout = u32::try_from(
            operation
                .deadline
                .saturating_duration_since(std::time::Instant::now())
                .as_millis(),
        )
        .unwrap_or(u32::MAX)
        .max(1);
        let commit = core.database.submit_with_timeout(
            command,
            CommitAcknowledgement::Published,
            timeout,
        )?;
        match commit.reply {
            BrokerReply::KafkaBatchPublished { first_offset, record_count } => Ok(StreamAcknowledgement { bookmark: commit.bookmark, first_offset, record_count }),
            _ => Err(Error::internal("stream append committed but returned an unexpected reply; outcome is uncertain; do not retry automatically").into()),
        }
    }

    pub fn stream_fetch(
        &self,
        request: StreamFetch,
        options: OperationOptions,
    ) -> Result<StreamPage> {
        let operation = self.begin_operation(options)?;
        operation.check()?;
        if request.max_records == 0
            || request.max_bytes == 0
            || request.max_bytes > self.options.max_write_bytes
        {
            return Err(Error::invalid_data("stream fetch requires positive bounds and max_bytes no greater than max_write_bytes").into());
        }
        let core = self
            .core
            .as_ref()
            .ok_or_else(|| Error::invalid_data("database is closed"))?;
        let (high_watermark, records) =
            core.database
                .fetch_partition_bounded(&irongraph_server::broker::PartitionRead {
                    project: request.project_id,
                    topic: &request.topic,
                    partition: request.partition,
                    offset: request.offset,
                    maximum_bytes: request.max_bytes,
                    maximum_records: request.max_records,
                })?;
        let mut selected = Vec::new();
        let mut bytes = 0usize;
        let mut next_offset = request.offset;
        let mut truncated = false;
        for (offset, record) in records {
            operation.check()?;
            if offset >= high_watermark {
                break;
            }
            // Include source metadata and payload in the response byte budget. Kafka itself
            // permits an oversized first record for progress; native finite reads reject it.
            let size = serde_json::to_vec(&(offset, &record))
                .map_err(|error| Error::internal(error.to_string()))?
                .len();
            if selected.len() == request.max_records
                || bytes.saturating_add(size) > request.max_bytes
            {
                if selected.is_empty() {
                    return Err(Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "first stream record exceeds max_bytes; increase the explicit fetch bound",
                    )
                    .into());
                }
                truncated = true;
                break;
            }
            bytes += size;
            next_offset = offset.saturating_add(1);
            selected.push((offset, record));
        }
        truncated |= next_offset < high_watermark;
        operation.check()?;
        Ok(StreamPage {
            records: selected,
            high_watermark,
            next_offset,
            truncated,
        })
    }
}
