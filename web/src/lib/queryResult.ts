import type { CompletionSchema, GraphEdge, GraphNode, QueryResult, QueryStreamEvent } from '../types';
import { columnsToRows } from './api';
import { isRecord, stringArray, stringField } from './guards';

export const EMPTY_RESULT: QueryResult = {
  columns: [],
  rows: [],
  nodes: [],
  edges: [],
  truncated: false,
};

function collectGraphValue(
  value: unknown,
  nodes: Map<string, GraphNode>,
  edges: Map<string, GraphEdge>,
): void {
  if (Array.isArray(value)) {
    value.forEach((item) => collectGraphValue(item, nodes, edges));
    return;
  }
  if (!isRecord(value)) return;

  if (value.__kind === 'node') {
    const id = stringField(value, 'id');
    if (id && !nodes.has(id)) {
      nodes.set(id, {
        id,
        labels: stringArray(value.labels),
        properties: isRecord(value.properties) ? value.properties : {},
      });
    }
    return;
  }
  if (value.__kind === 'relationship') {
    const id = stringField(value, 'id');
    const source = stringField(value, 'source');
    const target = stringField(value, 'target');
    if (id && source && target && !edges.has(id)) {
      edges.set(id, {
        id,
        source,
        target,
        relationshipType: stringField(value, 'relationshipType') ?? '',
        properties: isRecord(value.properties) ? value.properties : {},
      });
    }
    return;
  }
  Object.values(value).forEach((item) => collectGraphValue(item, nodes, edges));
}

export function applyQueryEvent(result: QueryResult, event: QueryStreamEvent): QueryResult {
  if (event.type === 'catalog' || event.type === 'error') return result;
  if (event.type === 'schema') return { ...result, columns: event.columns };
  if (event.type === 'summary') {
    return {
      ...result,
      bookmark: event.bookmark,
      statistics: event.statistics,
      truncated: event.truncated ?? result.truncated,
      truncationReason: event.truncationReason ?? result.truncationReason,
    };
  }

  const nodeMap = new Map(result.nodes.map((node) => [node.id, node]));
  const edgeMap = new Map(result.edges.map((edge) => [edge.id, edge]));
  let rows = result.rows;

  if (event.type === 'batch') {
    const incoming = event.rows ?? columnsToRows(event.columns ?? []);
    incoming.forEach((row) => row.forEach((value) => collectGraphValue(value, nodeMap, edgeMap)));
    rows = [...rows, ...incoming];
  } else {
    (event.nodes ?? []).forEach((node) => {
      nodeMap.set(node.id, node);
    });
    (event.edges ?? []).forEach((edge) => {
      edgeMap.set(edge.id, edge);
    });
  }

  return {
    ...result,
    rows,
    nodes: [...nodeMap.values()],
    edges: [...edgeMap.values()],
  };
}

/**
 * Folds a traversal's nodes and relationships into a result without touching its rows.
 *
 * Expanding a node answers a question the table was never asked: the rows belong to the Cypher the
 * user ran, and appending a traversal's rows to them would make the table claim results that query
 * never produced. Only the scene grows.
 *
 * The result is returned unchanged — the same object, so nothing downstream rebuilds or re-lays out
 * — when the traversal brought back nothing that was not already on screen.
 */
export function mergeGraphValues(result: QueryResult, values: unknown[]): QueryResult {
  const nodeMap = new Map(result.nodes.map((node) => [node.id, node]));
  const edgeMap = new Map(result.edges.map((edge) => [edge.id, edge]));
  const before = nodeMap.size + edgeMap.size;
  values.forEach((value) => collectGraphValue(value, nodeMap, edgeMap));
  if (nodeMap.size + edgeMap.size === before) return result;

  return {
    ...result,
    nodes: [...nodeMap.values()],
    edges: [...edgeMap.values()],
  };
}

export function mergeCompletionSchema(current: CompletionSchema, next: CompletionSchema): CompletionSchema {
  return {
    labels: [...new Set([...current.labels, ...next.labels])].sort(),
    relationshipTypes: [...new Set([...current.relationshipTypes, ...next.relationshipTypes])].sort(),
    properties: [...new Set([...current.properties, ...next.properties])].sort(),
    functions: [...new Set([...current.functions, ...next.functions])].sort(),
  };
}

export function deriveCompletionSchema(result: QueryResult): CompletionSchema {
  const labels = new Set<string>();
  const relationshipTypes = new Set<string>();
  const properties = new Set<string>();
  result.nodes.forEach((node) => {
    node.labels.forEach((label) => labels.add(label));
    Object.keys(node.properties).forEach((property) => properties.add(property));
  });
  result.edges.forEach((edge) => {
    relationshipTypes.add(edge.relationshipType);
    Object.keys(edge.properties).forEach((property) => properties.add(property));
  });
  return { labels: [...labels], relationshipTypes: [...relationshipTypes], properties: [...properties], functions: [] };
}
