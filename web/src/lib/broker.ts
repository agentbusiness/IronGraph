import { flag, num, text, type Row } from './cypher';

/**
 * The streams screen's side of the Cypher stream-administration grammar.
 *
 * Every operation the screen offers is one statement on the same `/api/query` channel the query
 * screen uses — `CREATE TOPIC`, `BIND QUEUE`, `PURGE QUEUE` and the rest. The builders live here,
 * beside the row readers for the four `SHOW` registers, so the screen composes statements it can
 * show the reader verbatim instead of hiding the language behind the controls.
 */

/**
 * A name, written so the parser reads it as one identifier.
 *
 * The grammar takes bare identifiers; anything else — a dot-routed key, a dashed name — must be
 * backtick-quoted, with embedded backticks doubled. Emitted here rather than at each call site so
 * no builder can forget and hand the parser two tokens where the reader typed one name.
 */
export function cypherIdent(name: string): string {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) ? name : `\`${name.replaceAll('`', '``')}\``;
}

export const SHOW_TOPICS = 'SHOW TOPICS';
export const SHOW_QUEUES = 'SHOW QUEUES';
export const SHOW_EXCHANGES = 'SHOW EXCHANGES';
export const SHOW_CONSUMER_LAG = 'SHOW CONSUMER LAG';

/** The partition count the engine accepts: a u16, and zero partitions is not a topic. */
export const PARTITION_LIMIT = 65_535;

export type QueueKind = 'CLASSIC' | 'STREAM';
export type ExchangeKind = 'DIRECT' | 'FANOUT' | 'TOPIC';

export const createTopic = (name: string, partitions: number, retentionDays?: number): string =>
  `CREATE TOPIC ${cypherIdent(name)} PARTITIONS ${Math.trunc(partitions)}${retentionDays === undefined ? '' : ` RETENTION ${Math.trunc(retentionDays)} DAYS`}`;
export const createQueue = (name: string, kind: QueueKind, retentionDays?: number): string =>
  `CREATE QUEUE ${cypherIdent(name)} ${kind}${retentionDays === undefined ? '' : ` RETENTION ${Math.trunc(retentionDays)} DAYS`}`;
export const alterTopicRetention = (name: string, retentionDays: number): string =>
  `ALTER TOPIC ${cypherIdent(name)} RETENTION ${Math.trunc(retentionDays)} DAYS`;
export const alterQueueRetention = (name: string, retentionDays: number): string =>
  `ALTER QUEUE ${cypherIdent(name)} RETENTION ${Math.trunc(retentionDays)} DAYS`;
export const createExchange = (name: string, kind: ExchangeKind): string =>
  `CREATE EXCHANGE ${cypherIdent(name)} TYPE ${kind}`;
export const bindQueue = (queue: string, exchange: string, key: string): string =>
  `BIND QUEUE ${cypherIdent(queue)} TO EXCHANGE ${cypherIdent(exchange)} KEY ${cypherIdent(key)}`;
export const unbindQueue = (queue: string, exchange: string, key: string): string =>
  `UNBIND QUEUE ${cypherIdent(queue)} TO EXCHANGE ${cypherIdent(exchange)} KEY ${cypherIdent(key)}`;
export const clearTopic = (name: string): string => `CLEAR TOPIC ${cypherIdent(name)}`;
export const purgeQueue = (name: string): string => `PURGE QUEUE ${cypherIdent(name)}`;
export const dropTopic = (name: string): string => `DROP TOPIC ${cypherIdent(name)}`;
export const dropQueue = (name: string): string => `DROP QUEUE ${cypherIdent(name)}`;
export const dropExchange = (name: string): string => `DROP EXCHANGE ${cypherIdent(name)}`;

export interface TopicPartition {
  partition: number;
  baseOffset: number;
  nextOffset: number;
  records: number;
  retainedBytes: number;
}

export interface TopicSummary {
  name: string;
  partitions: TopicPartition[];
  records: number;
  retainedBytes: number;
  retentionDays?: number;
}

/**
 * `SHOW TOPICS` answers one row per partition. A reader asking "what topics are there" is asking
 * about topics, so the rows are folded per name — the partition detail stays on the summary for
 * the margin to unfold when one topic is open.
 */
export function topicSummaries(rows: Row[]): TopicSummary[] {
  const byName = new Map<string, TopicSummary>();
  rows.forEach((row) => {
    const name = text(row.name);
    const partition: TopicPartition = {
      partition: num(row.partition),
      baseOffset: num(row.base_offset),
      nextOffset: num(row.next_offset),
      records: num(row.record_count),
      retainedBytes: num(row.retained_bytes),
    };
    const retentionMs = num(row.retention_ms);
    const summary = byName.get(name) ?? {
      name,
      partitions: [],
      records: 0,
      retainedBytes: 0,
      retentionDays: retentionMs > 0 ? retentionMs / 86_400_000 : undefined,
    };
    summary.partitions.push(partition);
    summary.records += partition.records;
    summary.retainedBytes += partition.retainedBytes;
    byName.set(name, summary);
  });
  return [...byName.values()]
    .map((summary) => ({
      ...summary,
      partitions: [...summary.partitions].sort((left, right) => left.partition - right.partition),
    }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

export interface QueueSummary {
  name: string;
  kind: string;
  messages: number;
  available: number;
  retainedBytes: number;
  retentionDays?: number;
}

export function queueSummaries(rows: Row[]): QueueSummary[] {
  return rows
    .map((row) => {
      const retentionMs = num(row.retention_ms);
      return {
        name: text(row.name),
        kind: text(row.kind),
        messages: num(row.message_count),
        available: num(row.available_count),
        retainedBytes: num(row.retained_bytes),
        retentionDays: retentionMs > 0 ? retentionMs / 86_400_000 : undefined,
      };
    })
    .sort((left, right) => left.name.localeCompare(right.name));
}

export interface ExchangeSummary {
  name: string;
  kind: string;
  durable: boolean;
  bindings: number;
}

export function exchangeSummaries(rows: Row[]): ExchangeSummary[] {
  return rows
    .map((row) => ({
      name: text(row.name),
      kind: text(row.kind),
      durable: flag(row.durable),
      bindings: num(row.binding_count),
    }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

export interface LagRow {
  group: string;
  topic: string;
  partition: number;
  committed: number;
  next: number;
  lag: number;
}

export function lagRows(rows: Row[]): LagRow[] {
  return rows
    .map((row) => ({
      group: text(row.group),
      topic: text(row.topic),
      partition: num(row.partition),
      committed: num(row.committed_offset),
      next: num(row.next_offset),
      lag: num(row.lag),
    }))
    .sort(
      (left, right) =>
        right.lag - left.lag ||
        left.group.localeCompare(right.group) ||
        left.topic.localeCompare(right.topic) ||
        left.partition - right.partition,
    );
}

/**
 * Bytes, written for a reader. Whole bytes below a kilobyte, then one decimal per thousand-step —
 * the broker's retained sizes are glanced at for magnitude, not audited to the byte.
 */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  if (bytes < 1_000) return `${Math.trunc(bytes).toLocaleString()} B`;
  const units = ['kB', 'MB', 'GB', 'TB'] as const;
  let value = bytes;
  let unit = -1;
  do {
    value /= 1_000;
    unit += 1;
  } while (value >= 1_000 && unit < units.length - 1);
  return `${value < 10 ? value.toFixed(1) : Math.round(value).toLocaleString()} ${units[unit]}`;
}

/** A name the grammar can hold at all: non-empty once trimmed. Quoting handles the rest. */
export function validName(name: string): boolean {
  return name.trim().length > 0;
}
