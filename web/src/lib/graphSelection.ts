import type { GraphEdge, GraphNode } from '../types';
import { isRecord } from './guards';

/**
 * What the canvas has selected, addressed the way the graph addresses it.
 *
 * The renderer used to keep this to itself and draw a card over the plot. The selection is now read
 * in the annotation margin, which is a different component in a different column, so it travels by
 * identity rather than by index: a traversal appends to the result, and an index into a list that
 * grows underneath a selection eventually names a different node than the one that was clicked.
 *
 * A cluster is the exception. It stands for a community the renderer merged and nothing outside the
 * scene knows about, so it carries its own summary rather than a key into the result.
 */
export interface ClusterSelection {
  kind: 'cluster';
  id: string;
  community: number;
  primaryLabel: string;
  memberCount: number;
  internalEdges: number;
  members: string[];
  color?: string;
}

/**
 * Colours travel with the selection, and only with it.
 *
 * What a node is painted in depends on what the canvas is colouring by, on a community pass that
 * ran there, and on the palette of the whole result — none of which exists outside the renderer.
 * Carrying the colour lets the margin draw the same swatch as the disc the reader clicked, which is
 * what makes a panel about one node findable among five hundred. The renderer refreshes it when the
 * scene restyles, so it cannot describe a colour the canvas has stopped using.
 */
export type GraphSelection =
  | { kind: 'node'; id: string; color?: string }
  | { kind: 'edge'; id: string; sourceColor?: string; targetColor?: string }
  | ClusterSelection;

/**
 * The properties worth showing. Internal engine-prefixed keys are bookkeeping,
 * and a property the engine answered as null carries no information the reader can act on — it
 * arrives either as a real null or, from an older wire format, as a typed null envelope.
 */
export function presentProperties(properties: Record<string, unknown>): [string, unknown][] {
  return Object.entries(properties).filter(([key, value]) => (
    !key.startsWith('__irongraph_') && !key.startsWith('__irongraph_')
    && value !== null
    && value !== undefined
    && !(isRecord(value) && value.type === 'null')
  ));
}

export interface AdjacentType {
  relationshipType: string;
  /** Relationships of this type that leave the node, and that arrive at it, in this result. */
  out: number;
  in: number;
}

/**
 * What a node is attached to *in the result on screen* — not in the graph.
 *
 * The distinction is the whole point of the count: it says how much of this node's neighbourhood is
 * already drawn, which is what tells the reader whether expanding it will bring anything back.
 */
export function adjacentTypes(edges: GraphEdge[], nodeId: string): AdjacentType[] {
  const counts = new Map<string, AdjacentType>();
  const bump = (relationshipType: string, direction: 'out' | 'in') => {
    const entry = counts.get(relationshipType) ?? { relationshipType, out: 0, in: 0 };
    entry[direction] += 1;
    counts.set(relationshipType, entry);
  };
  edges.forEach((edge) => {
    if (edge.source === nodeId) bump(edge.relationshipType || 'RELATED', 'out');
    if (edge.target === nodeId) bump(edge.relationshipType || 'RELATED', 'in');
  });
  return [...counts.values()].sort((left, right) => (
    (right.out + right.in) - (left.out + left.in) || left.relationshipType.localeCompare(right.relationshipType)
  ));
}

export function degreeInResult(adjacent: AdjacentType[]): number {
  return adjacent.reduce((total, entry) => total + entry.out + entry.in, 0);
}

/** Indexes a result's nodes by id, for the endpoint lookups a relationship's detail needs. */
export function nodeIndex(nodes: GraphNode[]): Map<string, GraphNode> {
  return new Map(nodes.map((node) => [node.id, node]));
}
