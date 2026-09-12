import type {
  CompletionSchema,
  GraphEdge,
  GraphNode,
  LocalAiIntegration,
  Project,
  QueryStreamEvent,
  ResultColumn,
} from '../types';
import { booleanField, isRecord, stringArray, stringField } from './guards';
import { decodeResponse, envelopeType } from './stream';

export interface QueryRequest {
  requestId?: string;
  projectId?: string;
  query: string;
  parameters?: Record<string, unknown>;
  bookmark?: { term: number; index: number };
  signal?: AbortSignal;
}

function protocolError(message: string): Error {
  return new Error(`Invalid server stream: ${message}`);
}

function unwrapTypedValue(value: unknown): unknown {
  if (!isRecord(value) || typeof value.type !== 'string') return value;
  if (value.type === 'null') return null;
  if (!('value' in value)) return value;
  const payload = value.value;
  switch (value.type) {
    case 'integer': return payload;
    case 'list': return Array.isArray(payload) ? payload.map(unwrapTypedValue) : [];
    case 'map':
      return isRecord(payload)
        ? Object.fromEntries(Object.entries(payload).map(([key, nested]) => [key, unwrapTypedValue(nested)]))
        : {};
    case 'node': {
      if (!isRecord(payload)) return payload;
      const properties = isRecord(payload.properties)
        ? Object.fromEntries(Object.entries(payload.properties).map(([key, nested]) => [key, unwrapTypedValue(nested)]))
        : {};
      return {
        __kind: 'node',
        id: stringField(payload, 'id', 'stable_id') ?? '',
        labels: stringArray(payload.labels),
        properties,
      } satisfies GraphNode & { __kind: 'node' };
    }
    case 'relationship': {
      if (!isRecord(payload)) return payload;
      const properties = isRecord(payload.properties)
        ? Object.fromEntries(Object.entries(payload.properties).map(([key, nested]) => [key, unwrapTypedValue(nested)]))
        : {};
      return {
        __kind: 'relationship',
        id: stringField(payload, 'id', 'stable_id') ?? '',
        source: stringField(payload, 'source', 'start', 'source_id') ?? '',
        target: stringField(payload, 'target', 'end', 'target_id') ?? '',
        relationshipType: stringField(payload, 'relationship_type', 'type') ?? '',
        properties,
      } satisfies GraphEdge & { __kind: 'relationship' };
    }
    case 'path':
      return isRecord(payload)
        ? {
            __kind: 'path',
            nodes: Array.isArray(payload.nodes) ? payload.nodes.map(unwrapTypedValue) : [],
            relationships: Array.isArray(payload.relationships)
              ? payload.relationships.map(unwrapTypedValue)
              : Array.isArray(payload.edges) ? payload.edges.map(unwrapTypedValue) : [],
          }
        : payload;
    case 'bytes':
    case 'date':
    case 'time':
    case 'datetime':
    case 'duration':
    case 'vector':
      return { __kind: value.type, value: payload };
    default: return payload;
  }
}

function queryEvent(data: unknown, explicitType?: string): QueryStreamEvent | undefined {
  if (!isRecord(data)) throw protocolError('query event must be an object');
  const type = explicitType ?? stringField(data, 'type');
  if (!type) throw protocolError('query event has no type');
  switch (type) {
    case 'catalog':
      return {
        type: 'catalog',
        catalog: {
          labels: stringArray(data.labels),
          relationshipTypes: stringArray(data.relationship_types),
          properties: stringArray(data.properties),
          functions: stringArray(data.functions),
        },
      };
    case 'schema':
      return {
        type: 'schema',
        columns: (Array.isArray(data.columns) ? data.columns : []).filter(isRecord).map((column) => ({
          name: stringField(column, 'name') ?? '',
          valueType: stringField(column, 'value_type'),
        })),
      };
    case 'batch':
      if (!Array.isArray(data.columns)) throw protocolError('batch columns are missing');
      return {
        type: 'batch',
        columns: data.columns.filter(isRecord).map((column) => ({
          name: stringField(column, 'name') ?? '',
          valueType: stringField(column, 'value_type'),
          values: Array.isArray(column.values) ? column.values.map(unwrapTypedValue) : [],
        })),
      };
    case 'summary':
      return {
        type: 'summary',
        bookmark: isRecord(data.bookmark)
          ? `${stringField(data.bookmark, 'term') ?? '0'}:${stringField(data.bookmark, 'index') ?? '0'}`
          : stringField(data, 'bookmark'),
        statistics: isRecord(data.statistics) ? { ...data.statistics } : undefined,
        truncated: booleanField(data, 'truncated'),
        truncationReason: stringField(data, 'truncation_reason'),
      };
    case 'error':
      return {
        type: 'error',
        code: stringField(data, 'code') ?? 'QueryError',
        message: stringField(data, 'message') ?? 'Query failed.',
      };
    default: return undefined;
  }
}

export async function* runQuery(request: QueryRequest): AsyncGenerator<QueryStreamEvent> {
  const response = await fetch('/api/query', {
    method: 'POST',
    cache: 'no-store',
    credentials: 'same-origin',
    headers: { Accept: 'application/x-ndjson', 'Content-Type': 'application/json' },
    body: JSON.stringify({
      request_id: request.requestId ?? crypto.randomUUID(),
      project_id: request.projectId ?? null,
      query: request.query,
      parameters: request.parameters ?? {},
      ...(request.bookmark ? { bookmark: request.bookmark } : {}),
    }),
    signal: request.signal,
  });
  for await (const envelope of decodeResponse(response)) {
    const event = queryEvent(envelope.data, envelopeType(envelope));
    if (event) yield event;
  }
}

export function columnsToRows(columns: Array<ResultColumn & { values: unknown[] }>): unknown[][] {
  const rowCount = columns.reduce((maximum, column) => Math.max(maximum, column.values.length), 0);
  return Array.from({ length: rowCount }, (_, rowIndex) => columns.map((column) => column.values[rowIndex]));
}

function valueAt(row: unknown[], columns: ResultColumn[], names: string[]): unknown {
  const normalized = names.map((name) => name.toLowerCase());
  const index = columns.findIndex((column) => normalized.includes(column.name.toLowerCase()));
  return index >= 0 ? row[index] : undefined;
}

function scalarText(value: unknown): string {
  return typeof value === 'string' || typeof value === 'number' || typeof value === 'bigint' || typeof value === 'boolean'
    ? String(value)
    : '';
}

export async function fetchProjects(signal?: AbortSignal): Promise<Project[]> {
  let columns: ResultColumn[] = [];
  const rows: unknown[][] = [];
  for await (const event of runQuery({ query: 'SHOW PROJECTS', signal })) {
    if (event.type === 'error') throw new Error(event.message);
    if (event.type === 'schema') columns = event.columns;
    if (event.type === 'batch') rows.push(...columnsToRows(event.columns ?? []));
    if (event.type === 'summary') break;
  }
  return rows
    .map((row) => ({
      id: scalarText(valueAt(row, columns, ['id', 'project_id'])),
      name: scalarText(valueAt(row, columns, ['name', 'display_name'])),
    }))
    .filter(({ id, name }) => id.length > 0 && name.length > 0)
    .sort((a, b) => a.name.localeCompare(b.name));
}

export async function createProject(name: string, signal?: AbortSignal): Promise<void> {
  const trimmed = name.trim();
  if (!trimmed) throw new Error('Enter a project name.');
  const identifier = `\`${trimmed.replaceAll('`', '``')}\``;
  for await (const event of runQuery({ query: `CREATE PROJECT ${identifier}`, signal })) {
    if (event.type === 'error') throw new Error(event.message);
    if (event.type === 'summary') return;
  }
  throw new Error('Project creation ended before completion.');
}

async function integrationResponse(response: Response): Promise<LocalAiIntegration[]> {
  const value: unknown = await response.json();
  if (!response.ok) {
    const detail = isRecord(value) ? stringField(value, 'detail') : undefined;
    throw new Error(detail ?? `Integration request failed with HTTP ${response.status}.`);
  }
  if (!Array.isArray(value)) throw new Error('Integration response must be an array.');
  return value.filter(isRecord).map((entry) => ({
    host: stringField(entry, 'host') ?? '',
    display_name: stringField(entry, 'display_name') ?? '',
    integration: stringField(entry, 'integration') ?? '',
    detected: Boolean(entry.detected),
    state: (stringField(entry, 'state') ?? 'not-installed') as LocalAiIntegration['state'],
    installed_version: stringField(entry, 'installed_version'),
    available_version: stringField(entry, 'available_version') ?? '',
    activation_required: Boolean(entry.activation_required),
    activation_instruction: stringField(entry, 'activation_instruction'),
    install_command: stringField(entry, 'install_command') ?? '',
    update_command: stringField(entry, 'update_command') ?? '',
  })).filter((entry) => entry.host && entry.display_name);
}

export async function fetchLocalAiIntegrations(signal?: AbortSignal): Promise<LocalAiIntegration[]> {
  return integrationResponse(await fetch('/system/local-ai-integrations', {
    cache: 'no-store',
    credentials: 'same-origin',
    signal,
  }));
}

export async function changeLocalAiIntegration(
  host: string,
  action: 'install' | 'update' | 'repair',
): Promise<LocalAiIntegration[]> {
  await integrationResponse(await fetch('/system/local-ai-integrations/' + encodeURIComponent(host) + '/' + action, {
    method: 'POST',
    cache: 'no-store',
    credentials: 'same-origin',
  }));
  return fetchLocalAiIntegrations();
}

export async function fetchCompletionSchema(projectId: string, signal?: AbortSignal): Promise<CompletionSchema> {
  const empty: CompletionSchema = { labels: [], relationshipTypes: [], properties: [], functions: [] };
  for await (const event of runQuery({ projectId, query: 'MATCH (n) RETURN count(n)', signal })) {
    if (event.type === 'catalog') return event.catalog;
    if (event.type === 'error') throw new Error(event.message);
    if (event.type === 'summary') break;
  }
  return empty;
}

export type GraphLayer = 'OBSERVED' | 'KNOWLEDGE' | 'WORKSPACE';
export const GRAPH_LAYERS: readonly GraphLayer[] = ['OBSERVED', 'KNOWLEDGE', 'WORKSPACE'];
export const DEFAULT_READ_LAYERS: readonly GraphLayer[] = ['OBSERVED', 'KNOWLEDGE'];

export interface LayerCensus {
  layer: GraphLayer;
  nodes: number;
  edges: number;
  labels: Map<string, number>;
  relationshipTypes: Map<string, number>;
}

export interface SchemaCensus {
  nodes: number;
  edges: number;
  labels: Map<string, number>;
  relationshipTypes: Map<string, number>;
  layersOfLabel: Map<string, GraphLayer[]>;
  layersOfRelationshipType: Map<string, GraphLayer[]>;
  byLayer: LayerCensus[];
}

function countRow(row: unknown[]): [string, number] | undefined {
  const [name, count] = row;
  if (typeof name !== 'string' || name.length === 0) return undefined;
  const amount = typeof count === 'number' ? count : Number(scalarText(count));
  return Number.isFinite(amount) ? [name, amount] : undefined;
}

async function groupedCounts(projectId: string, statement: string, signal?: AbortSignal): Promise<Map<string, number>> {
  const counts = new Map<string, number>();
  for await (const event of runQuery({ projectId, query: statement, signal })) {
    if (event.type === 'error') throw new Error(event.message);
    if (event.type === 'batch') {
      columnsToRows(event.columns ?? []).forEach((row) => {
        const entry = countRow(row);
        if (entry) counts.set(entry[0], entry[1]);
      });
    }
    if (event.type === 'summary') break;
  }
  return counts;
}

async function singleCount(projectId: string, statement: string, signal?: AbortSignal): Promise<number> {
  for await (const event of runQuery({ projectId, query: statement, signal })) {
    if (event.type === 'error') throw new Error(event.message);
    if (event.type === 'batch') {
      const first = columnsToRows(event.columns ?? [])[0]?.[0];
      const amount = typeof first === 'number' ? first : Number(scalarText(first));
      if (Number.isFinite(amount)) return amount;
    }
    if (event.type === 'summary') break;
  }
  return 0;
}

async function layerCensus(projectId: string, layer: GraphLayer, signal?: AbortSignal): Promise<LayerCensus> {
  const prefix = `USE LAYER ${layer} `;
  const [labels, relationshipTypes, nodes] = await Promise.all([
    groupedCounts(projectId, `${prefix}MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count`, signal),
    groupedCounts(projectId, `${prefix}MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count`, signal),
    singleCount(projectId, `${prefix}MATCH (n) RETURN count(n) AS count`, signal),
  ]);
  return {
    layer,
    nodes,
    edges: [...relationshipTypes.values()].reduce((sum, value) => sum + value, 0),
    labels,
    relationshipTypes,
  };
}

export async function fetchSchemaCensus(projectId: string, signal?: AbortSignal): Promise<SchemaCensus> {
  const byLayer = await Promise.all(GRAPH_LAYERS.map((layer) => layerCensus(projectId, layer, signal)));
  const labels = new Map<string, number>();
  const relationshipTypes = new Map<string, number>();
  const layersOfLabel = new Map<string, GraphLayer[]>();
  const layersOfRelationshipType = new Map<string, GraphLayer[]>();
  const fold = (
    source: Map<string, number>,
    totals: Map<string, number>,
    origin: Map<string, GraphLayer[]>,
    layer: GraphLayer,
  ) => source.forEach((count, name) => {
    totals.set(name, (totals.get(name) ?? 0) + count);
    origin.set(name, [...(origin.get(name) ?? []), layer]);
  });
  byLayer.forEach((census) => {
    fold(census.labels, labels, layersOfLabel, census.layer);
    fold(census.relationshipTypes, relationshipTypes, layersOfRelationshipType, census.layer);
  });
  return {
    nodes: byLayer.reduce((sum, census) => sum + census.nodes, 0),
    edges: byLayer.reduce((sum, census) => sum + census.edges, 0),
    labels,
    relationshipTypes,
    layersOfLabel,
    layersOfRelationshipType,
    byLayer,
  };
}
