import { DEFAULT_READ_LAYERS, type GraphLayer } from './api';

export const DEFAULT_GRAPH_QUERY = `MATCH (source)-[relationship]->(target)
RETURN source, relationship, target
LIMIT 100`;

/**
 * The traversal behind a double-click on a node.
 *
 * `id(n) = <literal>` is the one shape the planner turns into a stable-id seek rather than a scan,
 * so the traversal starts at the node instead of at the label. The pattern is undirected on
 * purpose: a relationship is no less part of a node's neighbourhood for pointing at it, and the
 * engine answers `-[r]-` with both directions (verified against a node whose only relationship is
 * incoming). Both endpoints and the relationship come back, so what lands is a graph.
 *
 * Node identity over the wire is the engine's `NodeId`, a u64 rendered as decimal. Anything else
 * is not addressable this way and is refused here rather than interpolated into Cypher.
 */
export function neighbourhoodQuery(nodeId: string): string | undefined {
  if (!/^\d+$/.test(nodeId)) return undefined;
  return `MATCH (source)-[relationship]-(target)
WHERE id(source) = ${nodeId}
RETURN source, relationship, target`;
}

/**
 * The layer prefix a generated statement needs, if any.
 *
 * An unprefixed statement reads OBSERVED and KNOWLEDGE only. A label that lives solely in
 * WORKSPACE therefore needs an explicit layer prefix. Where the census found the label decides
 * what is written.
 */
function layerPrefix(layers?: readonly GraphLayer[]): string {
  if (!layers || layers.length === 0) return '';
  if (layers.some((layer) => DEFAULT_READ_LAYERS.includes(layer))) return '';
  return `USE LAYER ${layers[0]}\n`;
}

/** The query a label in the schema list runs: the nodes carrying it. */
export function labelQuery(label: string, layers?: readonly GraphLayer[]): string {
  return `${layerPrefix(layers)}MATCH (n:${label})
RETURN n
LIMIT 100`;
}

/**
 * The query a relationship type in the schema list runs.
 *
 * It used to return the relationships alone, which plots as nothing at all: a relationship with no
 * endpoints has nowhere to be drawn, so picking a type from the schema opened an empty canvas. Both
 * ends come back with it, and the result is a graph.
 */
export function relationshipQuery(relationshipType: string, layers?: readonly GraphLayer[]): string {
  return `${layerPrefix(layers)}MATCH (source)-[relationship:${relationshipType}]->(target)
RETURN source, relationship, target
LIMIT 100`;
}

export function isProjectCatalogQuery(query: string): boolean {
  return /^\s*(?:(?:CREATE|ALTER|DROP)\s+PROJECT\b|SHOW\s+PROJECTS\b)/i.test(query);
}
