import { describe, expect, it } from 'vitest';
import {
  bindQueue,
  alterQueueRetention,
  alterTopicRetention,
  clearTopic,
  createExchange,
  createQueue,
  createTopic,
  cypherIdent,
  dropExchange,
  formatBytes,
  lagRows,
  queueSummaries,
  topicSummaries,
  unbindQueue,
} from './broker';

describe('cypherIdent', () => {
  it('passes plain identifiers through untouched', () => {
    expect(cypherIdent('events')).toBe('events');
    expect(cypherIdent('_audit_2')).toBe('_audit_2');
  });

  it('backtick-quotes anything the grammar would read as more than one token', () => {
    expect(cypherIdent('order.events')).toBe('`order.events`');
    expect(cypherIdent('2024-audit')).toBe('`2024-audit`');
    expect(cypherIdent('a b')).toBe('`a b`');
  });

  it('doubles embedded backticks so the quote cannot be escaped', () => {
    expect(cypherIdent('odd`name')).toBe('`odd``name`');
  });
});

describe('statement builders', () => {
  it('write the administration grammar exactly', () => {
    expect(createTopic('events', 12)).toBe('CREATE TOPIC events PARTITIONS 12');
    expect(createQueue('jobs', 'STREAM')).toBe('CREATE QUEUE jobs STREAM');
    expect(createTopic('audit', 3, 30)).toBe('CREATE TOPIC audit PARTITIONS 3 RETENTION 30 DAYS');
    expect(createQueue('archive', 'STREAM', 7)).toBe('CREATE QUEUE archive STREAM RETENTION 7 DAYS');
    expect(alterTopicRetention('audit', 90)).toBe('ALTER TOPIC audit RETENTION 90 DAYS');
    expect(alterQueueRetention('archive', 14)).toBe('ALTER QUEUE archive RETENTION 14 DAYS');
    expect(createExchange('routing', 'TOPIC')).toBe('CREATE EXCHANGE routing TYPE TOPIC');
    expect(bindQueue('jobs', 'routing', 'work')).toBe('BIND QUEUE jobs TO EXCHANGE routing KEY work');
    expect(unbindQueue('jobs', 'routing', 'work')).toBe('UNBIND QUEUE jobs TO EXCHANGE routing KEY work');
    expect(clearTopic('events')).toBe('CLEAR TOPIC events');
    expect(dropExchange('routing')).toBe('DROP EXCHANGE routing');
  });

  it('truncates a fractional partition count rather than emitting a float', () => {
    expect(createTopic('events', 3.7)).toBe('CREATE TOPIC events PARTITIONS 3');
  });
});

describe('topicSummaries', () => {
  it('folds the per-partition rows into one summary per topic', () => {
    const rows = [
      { name: 'events', partition: '1', base_offset: '0', next_offset: '5', record_count: '5', retained_bytes: '900' },
      { name: 'events', partition: '0', base_offset: '0', next_offset: '7', record_count: '7', retained_bytes: '1200' },
      { name: 'audit', partition: '0', base_offset: '2', next_offset: '2', record_count: '0', retained_bytes: '0' },
    ];
    const summaries = topicSummaries(rows);
    expect(summaries.map((summary) => summary.name)).toEqual(['audit', 'events']);
    const events = summaries[1]!;
    expect(events.records).toBe(12);
    expect(events.retainedBytes).toBe(2100);
    // Partition detail survives, ordered by partition number for the margin's breakdown.
    expect(events.partitions.map((partition) => partition.partition)).toEqual([0, 1]);
  });
});

describe('queueSummaries', () => {
  it('reads the SHOW QUEUES columns, however the wire typed them', () => {
    const summaries = queueSummaries([
      { name: 'jobs', kind: 'STREAM', message_count: '4', available_count: '3', retained_bytes: '512', retention_ms: '604800000' },
    ]);
    expect(summaries[0]).toEqual({ name: 'jobs', kind: 'STREAM', messages: 4, available: 3, retainedBytes: 512, retentionDays: 7 });
  });
});

describe('lagRows', () => {
  it('sorts the biggest lag to the top, then names the rest stably', () => {
    const rows = lagRows([
      { group: 'a', topic: 't', partition: '0', committed_offset: '5', next_offset: '5', lag: '0' },
      { group: 'b', topic: 't', partition: '1', committed_offset: '2', next_offset: '9', lag: '7' },
    ]);
    expect(rows[0]!.group).toBe('b');
    expect(rows[0]!.lag).toBe(7);
  });
});

describe('formatBytes', () => {
  it('writes magnitudes a reader can glance at', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(999)).toBe('999 B');
    expect(formatBytes(1_500)).toBe('1.5 kB');
    expect(formatBytes(2_400_000)).toBe('2.4 MB');
    expect(formatBytes(53_000_000_000)).toBe('53 GB');
  });
});
