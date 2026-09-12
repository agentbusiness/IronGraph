import { describe, expect, it } from 'vitest';
import type { GraphEdge, GraphNode } from '../types';
import {
  analysisForMerged,
  buildMergedGraph,
  buildScene,
  communitySummaries,
  egoNodes,
  runAnalytics,
  shortestPath,
  styleScene,
  withAlpha,
  type AnalyticsOptions,
} from './graphScene';

function node(id: string, label = 'Thing', properties: Record<string, unknown> = {}): GraphNode {
  return { id, labels: [label], properties };
}

function edge(id: string, source: string, target: string, relationshipType = 'LINKS'): GraphEdge {
  return { id, source, target, relationshipType, properties: {} };
}

/** Two 4-cliques joined by a single relationship: an unambiguous two-community graph. */
function barbell(): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const left = ['a', 'b', 'c', 'd'];
  const right = ['e', 'f', 'g', 'h'];
  const edges: GraphEdge[] = [];
  [left, right].forEach((clique) => {
    clique.forEach((from, index) => {
      clique.slice(index + 1).forEach((to) => edges.push(edge(`${from}-${to}`, from, to)));
    });
  });
  edges.push(edge('bridge', 'd', 'e'));
  return { nodes: [...left, ...right].map((id) => node(id)), edges };
}

const ALL: AnalyticsOptions = { resolution: 1, includeCommunities: true, includePagerank: true, includeBetweenness: true };

describe('buildScene', () => {
  it('keeps parallel relationships on the rendered graph but weights them once for analysis', () => {
    const scene = buildScene([node('a'), node('b')], [
      edge('one', 'a', 'b', 'KNOWS'),
      edge('two', 'a', 'b', 'WORKS_WITH'),
      edge('three', 'b', 'a', 'MANAGES'),
    ]);
    expect(scene.display.size).toBe(3);
    expect(scene.analysis.size).toBe(1);
    const link = scene.analysis.edge('a', 'b');
    expect(link).toBeDefined();
    expect(scene.analysis.getEdgeAttribute(link, 'weight')).toBe(3);
  });

  it('ignores duplicate nodes and relationships whose endpoints are outside the result', () => {
    const scene = buildScene([node('a'), node('a'), node('b')], [
      edge('kept', 'a', 'b'),
      edge('dangling', 'a', 'missing'),
      edge('kept', 'b', 'a'),
    ]);
    expect(scene.display.order).toBe(2);
    expect(scene.display.size).toBe(1);
  });

  it('seeds every node at a distinct position so the layout has something to push apart', () => {
    const scene = buildScene([node('a'), node('b'), node('c')], []);
    const positions = scene.display.nodes().map((id) => {
      const { x, y } = scene.display.getNodeAttributes(id);
      return `${x},${y}`;
    });
    expect(new Set(positions).size).toBe(3);
  });

  it('labels a node by its naming property when it has one', () => {
    const scene = buildScene([node('a', 'Person', { name: 'Ada' })], []);
    expect(scene.display.getNodeAttribute('a', 'label')).toBe('Ada');
  });
});

describe('runAnalytics', () => {
  it('separates the two halves of a barbell into communities', () => {
    const { nodes, edges } = barbell();
    const analytics = runAnalytics(buildScene(nodes, edges), ALL);
    expect(analytics.communityCount).toBe(2);
    expect(analytics.communityOf.get('a')).toBe(analytics.communityOf.get('c'));
    expect(analytics.communityOf.get('f')).toBe(analytics.communityOf.get('h'));
    expect(analytics.communityOf.get('a')).not.toBe(analytics.communityOf.get('h'));
    expect(analytics.modularity).toBeGreaterThan(0);
  });

  it('skips every optional metric that no control asked for', () => {
    const { nodes, edges } = barbell();
    const analytics = runAnalytics(buildScene(nodes, edges), {
      resolution: 1,
      includeCommunities: false,
      includePagerank: false,
      includeBetweenness: false,
    });
    expect(analytics.communityCount).toBe(1);
    expect(analytics.betweennessOf).toBeUndefined();
    expect([...analytics.pagerankOf.values()].every((value) => value === 0)).toBe(true);
    // Degree comes free with the graph, so it is always available for sizing.
    expect(analytics.degreeOf.get('d')).toBe(4);
  });

  it('ranks the bridge nodes highest on betweenness', () => {
    const { nodes, edges } = barbell();
    const analytics = runAnalytics(buildScene(nodes, edges), ALL);
    const scores = analytics.betweennessOf;
    expect(scores).toBeDefined();
    expect(scores?.get('d') ?? 0).toBeGreaterThan(scores?.get('a') ?? 0);
    expect(scores?.get('e') ?? 0).toBeGreaterThan(scores?.get('h') ?? 0);
  });
});

describe('buildMergedGraph', () => {
  it('collapses each community into one node and aggregates the relationships that cross', () => {
    const { nodes, edges } = barbell();
    const scene = buildScene(nodes, edges);
    const analytics = runAnalytics(scene, ALL);
    const merged = buildMergedGraph(scene.display, analytics);

    expect(merged.order).toBe(2);
    expect(merged.size).toBe(1);
    const totalMembers = merged.nodes().reduce((total, id) => total + merged.getNodeAttribute(id, 'memberCount'), 0);
    expect(totalMembers).toBe(8);
    const internal = merged.nodes().reduce((total, id) => total + merged.getNodeAttribute(id, 'internalEdges'), 0);
    expect(internal).toBe(12);
    expect(merged.edges().map((id) => merged.getEdgeAttribute(id, 'aggregateCount'))).toEqual([1]);
    merged.forEachNode((_id, attributes) => {
      expect(attributes.kind).toBe('cluster');
      expect(attributes.members).toHaveLength(4);
      expect(attributes.sourceIndex).toBe(-1);
    });
  });

  it('bundles every crossing relationship between the same pair into one weighted link', () => {
    const nodes = ['a', 'b', 'c', 'd'].map((id) => node(id));
    const edges = [
      edge('a-b', 'a', 'b'), edge('c-d', 'c', 'd'),
      edge('cross-1', 'a', 'c'), edge('cross-2', 'b', 'd'),
    ];
    const scene = buildScene(nodes, edges);
    const analytics = runAnalytics(scene, ALL);
    // Force the two pairs apart regardless of what Louvain settles on for this tiny graph.
    analytics.communityOf.set('a', 0);
    analytics.communityOf.set('b', 0);
    analytics.communityOf.set('c', 1);
    analytics.communityOf.set('d', 1);
    const merged = buildMergedGraph(scene.display, analytics);
    expect(merged.size).toBe(1);
    expect(merged.getEdgeAttribute(merged.edges()[0] ?? '', 'aggregateCount')).toBe(2);
  });

  it('projects the merged graph onto a layout graph carrying the aggregate weights', () => {
    const { nodes, edges } = barbell();
    const scene = buildScene(nodes, edges);
    const merged = buildMergedGraph(scene.display, runAnalytics(scene, ALL));
    const analysis = analysisForMerged(merged);
    expect(analysis.order).toBe(2);
    expect(analysis.size).toBe(1);
    expect(analysis.getEdgeAttribute(analysis.edges()[0] ?? '', 'weight')).toBe(1);
  });
});

describe('styleScene', () => {
  it('sizes nodes by the chosen metric and leaves them flat when asked', () => {
    const { nodes, edges } = barbell();
    const scene = buildScene(nodes, edges);
    const analytics = runAnalytics(scene, ALL);

    styleScene(scene.display, analytics, { colorBy: 'label', sizeBy: 'degree' });
    expect(scene.display.getNodeAttribute('d', 'size')).toBeGreaterThan(scene.display.getNodeAttribute('a', 'size'));

    styleScene(scene.display, analytics, { colorBy: 'label', sizeBy: 'uniform' });
    expect(scene.display.getNodeAttribute('d', 'size')).toBe(scene.display.getNodeAttribute('a', 'size'));
  });

  it('colours every member of a community alike, and different communities apart', () => {
    const { nodes, edges } = barbell();
    const scene = buildScene(nodes, edges);
    const analytics = runAnalytics(scene, ALL);
    styleScene(scene.display, analytics, { colorBy: 'community', sizeBy: 'degree' });
    expect(scene.display.getNodeAttribute('a', 'color')).toBe(scene.display.getNodeAttribute('c', 'color'));
    expect(scene.display.getNodeAttribute('a', 'color')).not.toBe(scene.display.getNodeAttribute('h', 'color'));
  });
});

describe('egoNodes', () => {
  it('grows one hop at a time and ignores relationship direction', () => {
    const scene = buildScene(
      ['a', 'b', 'c', 'd'].map((id) => node(id)),
      [edge('a-b', 'a', 'b'), edge('c-b', 'c', 'b'), edge('c-d', 'c', 'd')],
    );
    expect([...egoNodes(scene.display, 'a', 1)].sort()).toEqual(['a', 'b']);
    expect([...egoNodes(scene.display, 'a', 2)].sort()).toEqual(['a', 'b', 'c']);
    expect([...egoNodes(scene.display, 'a', 3)].sort()).toEqual(['a', 'b', 'c', 'd']);
    expect(egoNodes(scene.display, 'missing', 2).size).toBe(0);
  });
});

describe('shortestPath', () => {
  it('finds the hop path across the bridge and reports when there is none', () => {
    const { nodes, edges } = barbell();
    const scene = buildScene(nodes, edges);
    expect(shortestPath(scene.analysis, 'a', 'e')).toEqual(['a', 'd', 'e']);

    const split = buildScene([node('a'), node('b')], []);
    expect(shortestPath(split.analysis, 'a', 'b')).toBeUndefined();
    expect(shortestPath(split.analysis, 'a', 'missing')).toBeUndefined();
  });
});

describe('communitySummaries', () => {
  it('lists the largest communities first with their dominant label', () => {
    const nodes = [
      ...['a', 'b', 'c', 'd'].map((id) => node(id, 'Person')),
      ...['e', 'f', 'g', 'h'].map((id) => node(id, 'Company')),
    ];
    const { edges } = barbell();
    const scene = buildScene(nodes, edges);
    const summaries = communitySummaries(scene.display, runAnalytics(scene, ALL), 10);
    expect(summaries).toHaveLength(2);
    expect(summaries.map((summary) => summary.dominantLabel).sort()).toEqual(['Company', 'Person']);
    expect(summaries.every((summary) => summary.nodeCount === 4)).toBe(true);
  });

  it('stops at the requested limit', () => {
    const nodes = Array.from({ length: 10 }, (_value, index) => node(`n${index}`));
    const scene = buildScene(nodes, []);
    const analytics = runAnalytics(scene, ALL);
    nodes.forEach((entry, index) => analytics.communityOf.set(entry.id, index));
    expect(communitySummaries(scene.display, analytics, 3)).toHaveLength(3);
  });
});

describe('withAlpha', () => {
  it('makes hex, rgb and hsl colours translucent, and leaves anything else alone', () => {
    expect(withAlpha('#6d7a88', 0.35)).toBe('rgba(109, 122, 136, 0.35)');
    expect(withAlpha('#abc', 1)).toBe('rgba(170, 187, 204, 1)');
    expect(withAlpha('hsl(210 46% 58%)', 0.1)).toBe('hsla(210 46% 58% / 0.1)');
    expect(withAlpha('rgb(1, 2, 3)', 0.5)).toBe('rgba(1, 2, 3, 0.5)');
    expect(withAlpha('rgb(1 2 3)', 0.5)).toBe('rgba(1 2 3 / 0.5)');
    expect(withAlpha('currentColor', 0.5)).toBe('currentColor');
  });
});
