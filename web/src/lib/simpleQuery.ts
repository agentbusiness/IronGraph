import type { QueryResult } from '../types';
import { runQuery } from './api';
import { QueryResultCollector } from './queryResult';

/** Run one statement and collect its complete stream into the same result shape as Query. */
export async function executeQuery(
  query: string,
  projectId?: string,
  parameters: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<QueryResult> {
  const result = new QueryResultCollector();
  for await (const event of runQuery({ query, projectId, parameters, signal })) {
    if (event.type === 'error') throw new Error(event.message);
    result.append(event);
    if (event.type === 'summary') return result.finish();
  }
  throw new Error('The query ended before IronGraph returned a summary.');
}

export function rowObjects(result: QueryResult): Record<string, unknown>[] {
  return result.rows.map((row) => Object.fromEntries(
    result.columns.map((column, index) => [column.name, row[index]]),
  ));
}
