import type { GraphNode } from '../types';
import { isRecord, stringField } from './guards';

const NODE_TITLE_KEYS = ['name', 'title', 'username', 'handle'] as const;

/**
 * The human name for a node: a naming property when it has one, else its label or id. Lives here
 * rather than with the scene model so the 3D renderer can label a node without pulling Graphology in.
 */
export function nodeDisplayLabel(node: GraphNode | undefined): string {
  if (!node) return 'Unknown node';
  for (const key of NODE_TITLE_KEYS) {
    const value = node.properties[key];
    if (value !== null && value !== undefined && formatValue(value).length > 0) return formatValue(value, 180);
  }
  return node.labels[0] ?? node.id;
}

export function formatTime(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(timestamp);
}

/**
 * When something is, written the way a person says it.
 *
 * The year is stated only when it is not the current one. Omitting it always read a March 2027
 * appointment as "Mar 30" — the same string March just gone would print — so a date seven months
 * ahead was indistinguishable from an old one, and a calendar of upcoming things looked like a
 * calendar of ancient ones.
 */
export function whenLabel(instantMs: number, nowMs: number = Date.now()): string {
  const at = new Date(instantMs);
  const thisYear = at.getFullYear() === new Date(nowMs).getFullYear();
  return at.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    ...(thisYear ? {} : { year: 'numeric' }),
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** The same rule for a date with no clock time: a day, and the year when it is not this one. */
export function dayLabel(instantMs: number, nowMs: number = Date.now()): string {
  const at = new Date(instantMs);
  const thisYear = at.getFullYear() === new Date(nowMs).getFullYear();
  return at.toLocaleDateString(undefined, {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
    ...(thisYear ? {} : { year: 'numeric' }),
  });
}

export function formatRelativeTime(timestamp: number): string {
  const seconds = Math.round((timestamp - Date.now()) / 1000);
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
  if (Math.abs(seconds) < 60) return formatter.format(seconds, 'second');
  const minutes = Math.round(seconds / 60);
  if (Math.abs(minutes) < 60) return formatter.format(minutes, 'minute');
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return formatter.format(hours, 'hour');
  return formatter.format(Math.round(hours / 24), 'day');
}

export function formatValue(value: unknown, maxLength = 600): string {
  if (value === null) return 'null';
  if (value === undefined) return '';
  if (typeof value === 'string') return value.length <= maxLength ? value : `${value.slice(0, maxLength - 1)}…`;
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') return String(value);
  if (isRecord(value)) {
    const kind = stringField(value, '__kind');
    if (kind === 'node') {
      const labels = Array.isArray(value.labels) ? value.labels.filter((item): item is string => typeof item === 'string') : [];
      return `(${labels.map((label) => `:${label}`).join('')} #${stringField(value, 'id') ?? '?'})`;
    }
    if (kind === 'relationship') return `[:${stringField(value, 'relationshipType') ?? '?'} #${stringField(value, 'id') ?? '?'}]`;
    if (kind === 'path') {
      const nodes = Array.isArray(value.nodes) ? value.nodes.length : 0;
      const relationships = Array.isArray(value.relationships) ? value.relationships.length : 0;
      return `Path (${nodes} nodes, ${relationships} relationships)`;
    }
    if (kind === 'vector' && Array.isArray(value.value)) {
      const shown = value.value.slice(0, 8).map((item) => typeof item === 'number' ? item.toPrecision(5) : '?');
      return `⟨${shown.join(', ')}${value.value.length > shown.length ? ', …' : ''}⟩`;
    }
    if (kind === 'bytes') {
      const bytes = typeof value.value === 'string' ? Math.floor(value.value.length * 0.75) : 0;
      return `[${bytes.toLocaleString()} bytes]`;
    }
    if (kind === 'date' || kind === 'time' || kind === 'datetime' || kind === 'duration') {
      return typeof value.value === 'string' ? value.value : '[invalid temporal value]';
    }
  }
  try {
    const serialized: string | undefined = JSON.stringify(value, (_key: string, nested: unknown): unknown =>
      typeof nested === 'bigint' ? nested.toString() : nested,
    );
    if (!serialized) return '[value]';
    return serialized.length <= maxLength ? serialized : `${serialized.slice(0, maxLength - 1)}…`;
  } catch {
    return '[unserializable value]';
  }
}

/**
 * How long the engine took, at the resolution it took it in.
 *
 * The server reported whole milliseconds, and a graph engine that answers four hundred rows in
 * well under one of them reported `0 ms` — a number that reads as "broken" and throws away the one
 * measurement that says what this engine is. The unit follows the magnitude: microseconds below a
 * millisecond, milliseconds above it, seconds once milliseconds stop being readable. Two
 * significant figures under ten so 4.2 ms and 12 ms are both a glance, and never a bare zero —
 * a request that finished faster than the clock can name is reported as such, not as nothing.
 */
export function formatElapsed(microseconds: number | undefined): string | undefined {
  if (microseconds === undefined || !Number.isFinite(microseconds) || microseconds < 0) return undefined;
  if (microseconds === 0) return '<1 \u00b5s';
  if (microseconds < 1_000) return `${Math.round(microseconds).toLocaleString()} \u00b5s`;
  const milliseconds = microseconds / 1_000;
  if (milliseconds < 10) return `${milliseconds.toFixed(1)} ms`;
  if (milliseconds < 1_000) return `${Math.round(milliseconds).toLocaleString()} ms`;
  const seconds = milliseconds / 1_000;
  return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds).toLocaleString()} s`;
}

/**
 * What the summary measured, whichever field the server had to say it with. A server that predates
 * microsecond timing sends only whole milliseconds; reading its `0` as zero microseconds would
 * print "<1 µs" for a request that was never measured at all, so an absent microsecond field with
 * a zero millisecond field is treated as no measurement rather than as a very fast one.
 */
export function elapsedMicroseconds(statistics: Record<string, unknown> | undefined): number | undefined {
  if (!statistics) return undefined;
  const micros = statistics.elapsed_us;
  if (typeof micros === 'number' && Number.isFinite(micros)) return micros;
  const millis = statistics.elapsed_ms;
  if (typeof millis === 'number' && Number.isFinite(millis) && millis > 0) return millis * 1_000;
  return undefined;
}

export function errorMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === 'AbortError') return 'Cancelled.';
  if (error instanceof Error) return error.message;
  return 'An unexpected error occurred.';
}
