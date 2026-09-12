import type { QueryStreamEvent, ResultColumn } from '../types';
import { runQuery } from './api';

/**
 * The one way this client reads and writes the graph.
 *
 * The entire client-facing data surface is the `/api/query` Cypher stream.
 *
 * **The queries live beside the functions that send them**, rather than in one big catalogue, so a
 * screen's data requirements are readable where the screen's data is fetched. A query nobody can
 * find is a query nobody will fix.
 */

/** One row, already aligned to the names its query returned. */
export type Row = Record<string, unknown>;

/**
 * Run a query and collect its rows.
 *
 * `/api/query` streams natively; this helper collects the whole answer before rendering. A read that
 * genuinely wants to stream should
 * use `runQuery` directly rather than fighting this.
 */
export async function query(
  projectId: string,
  cypher: string,
  parameters: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<Row[]> {
  let columns: ResultColumn[] = [];
  const rows: Row[] = [];
  let completed = false;
  for await (const event of runQuery({ projectId, query: cypher, parameters, signal })) {
    if (event.type === 'error') throw new Error(event.message);
    if (event.type === 'schema') columns = event.columns;
    if (event.type === 'batch') rows.push(...batchRows(event, columns));
    if (event.type === 'summary') {
      completed = true;
      break;
    }
  }
  // A stream that stops without a summary is a refused query, not an empty result.
  //
  // The engine can decline a shape — `ORDER BY` over a label scan is the known one — and when it
  // does, the stream simply ends: no `error` event, no `summary`. Treating that as "nothing found"
  // otherwise makes a refused query indistinguishable from a real empty result.
  if (!completed) {
    throw new Error(
      'The query ended without completing. The engine refused this shape rather than returning nothing.',
    );
  }
  return rows;
}

/**
 * Run a query for its effect and wait for it to finish.
 *
 * Separate from [`query`] because a write that returns nothing is not a read that found nothing,
 * and a caller silently treating one as the other is how a failed save looks like an empty list.
 */
export async function execute(
  projectId: string,
  cypher: string,
  parameters: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<void> {
  for await (const event of runQuery({ projectId, query: cypher, parameters, signal })) {
    if (event.type === 'error') throw new Error(event.message);
    if (event.type === 'summary') return;
  }
  throw new Error('The query ended before it completed.');
}

/**
 * Turn one batch into named rows.
 *
 * The wire is column-oriented — every column carries its own array of values — so this transposes.
 * Column names arrive on the schema event and are keyed off exactly as the query wrote them, which
 * is why every query here returns `RETURN x.y AS name` rather than relying on a positional index:
 * a reordered `RETURN` must not silently reassign fields.
 */
function batchRows(event: Extract<QueryStreamEvent, { type: 'batch' }>, schema: ResultColumn[]): Row[] {
  const columns = event.columns ?? [];
  const names =
    columns.length > 0
      ? columns.map((column: ResultColumn) => column.name)
      : schema.map((column: ResultColumn) => column.name);
  const height = columns[0]?.values?.length ?? 0;
  const rows: Row[] = [];
  for (let index = 0; index < height; index += 1) {
    const row: Row = {};
    names.forEach((name: string, position: number) => {
      row[name] = unwrap(columns[position]?.values?.[index]);
    });
    rows.push(row);
  }
  return rows;
}

/** The wire wraps every scalar as `{type, value}`; callers want the value. */
function unwrap(cell: unknown): unknown {
  if (cell && typeof cell === 'object' && 'value' in cell) {
    return cell.value;
  }
  return cell ?? null;
}

/** A string, however the engine typed it. Absent and null both read as empty. */
export function text(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') return String(value);
  return JSON.stringify(value) ?? '';
}

/**
 * A number, however the engine typed it.
 *
 * Integers arrive as **strings** on this wire, because a 64-bit value does not survive JSON's
 * float. Parsing here rather than at each call site is what stops a timestamp silently becoming
 * `NaN` in one module and a string in another.
 */
export function num(value: unknown): number {
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}

/** A boolean, however the engine typed it. */
export function flag(value: unknown): boolean {
  if (typeof value === 'boolean') return value;
  return text(value).toLowerCase() === 'true';
}

/** The layer prefixes, so no query spells them by hand and gets one wrong. */
export const WORKSPACE = 'USE LAYER WORKSPACE ';
export const WORKSPACE_WRITE = 'USE LAYER WORKSPACE WRITE LAYER WORKSPACE ';
export const OBSERVED = 'USE LAYER OBSERVED ';
/**
 * Every layer, for a read that crosses them.
 *
 * A cross-layer edge is legal and carries its own layer, but the read must **name every layer it
 * touches** or the `MATCH` silently finds nothing — no error, no rows. That failure is why this is
 * a constant rather than something each query writes out.
 */
export const ALL_LAYERS = 'USE LAYER OBSERVED, KNOWLEDGE, WORKSPACE ';
export const ALL_LAYERS_WRITE_WORKSPACE = 'USE LAYER OBSERVED, KNOWLEDGE, WORKSPACE WRITE LAYER WORKSPACE ';
