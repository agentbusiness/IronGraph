import type { QueryResult } from '../types';
import { runQuery } from './api';
import { EMPTY_RESULT, applyQueryEvent } from './queryResult';

/** Run one bounded statement and fold its stream into the same result shape as Query. */
export async function executeQuery(
  query: string,
  projectId?: string,
  parameters: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<QueryResult> {
  let result = EMPTY_RESULT;
  for await (const event of runQuery({ query, projectId, parameters, signal })) {
    if (event.type === 'error') throw new Error(event.message);
    result = applyQueryEvent(result, event);
    if (event.type === 'summary') return result;
  }
  throw new Error('The query ended before IronGraph returned a summary.');
}

export function rowObjects(result: QueryResult): Record<string, unknown>[] {
  return result.rows.map((row) => Object.fromEntries(
    result.columns.map((column, index) => [column.name, row[index]]),
  ));
}
