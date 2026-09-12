import { describe, expect, it } from 'vitest';
import type { GraphEdge, GraphNode } from '../types';
import { buildScene, runAnalytics } from './graphScene';
import { centroidOf, communityHulls, convexHull, expandHull, trimOutliers, type Point } from './graphHulls';

function key(point: Point): string {
  return `${Math.round(point.x)},${Math.round(point.y)}`;
}

describe('convexHull', () => {
  it('drops interior points and keeps the corners', () => {
    const hull = convexHull([
      { x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 },
      { x: 5, y: 5 }, { x: 3, y: 7 },
    ]);
    expect(hull).toHaveLength(4);
    expect(new Set(hull.map(key))).toEqual(new Set(['0,0', '10,0', '10,10', '0,10']));
  });

  it('drops points that only sit on an edge of the hull', () => {
    const hull = convexHull([{ x: 0, y: 0 }, { x: 5, y: 0 }, { x: 10, y: 0 }, { x: 5, y: 10 }]);
    expect(new Set(hull.map(key))).toEqual(new Set(['0,0', '10,0', '5,10']));
  });

  it('returns anything too small to enclose unchanged', () => {
    expect(convexHull([])).toEqual([]);
    expect(convexHull([{ x: 1, y: 2 }, { x: 3, y: 4 }])).toHaveLength(2);
  });
});

describe('expandHull', () => {
  it('pushes each vertex away from the centroid by the padding', () => {
    const expanded = expandHull([{ x: 10, y: 0 }, { x: -10, y: 0 }], { x: 0, y: 0 }, 5);
    expect(expanded[0]).toEqual({ x: 15, y: 0 });
    expect(expanded[1]).toEqual({ x: -15, y: 0 });
  });

  it('nudges a vertex sitting exactly on the centroid instead of dividing by zero', () => {
    const expanded = expandHull([{ x: 0, y: 0 }], { x: 0, y: 0 }, 4);
    expect(expanded[0]).toEqual({ x: 4, y: 0 });
  });
});

describe('trimOutliers', () => {
  it('drops the strays a community has lent to its neighbours', () => {
    const cluster: Point[] = Array.from({ length: 20 }, (_value, index) => ({ x: index % 5, y: Math.floor(index / 5) }));
    const kept = trimOutliers([...cluster, { x: 900, y: 900 }, { x: -900, y: -900 }]);
    expect(kept).not.toContainEqual({ x: 900, y: 900 });
    expect(kept).not.toContainEqual({ x: -900, y: -900 });
    expect(kept.length).toBeGreaterThanOrEqual(15);
  });

  it('keeps everything when a community is too small to have a bulk', () => {
    const tiny: Point[] = [{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 500, y: 500 }];
    expect(trimOutliers(tiny)).toEqual(tiny);
  });

  it('leaves an evenly spread community alone', () => {
    const ring: Point[] = Array.from({ length: 12 }, (_value, index) => ({
      x: Math.round(Math.cos((index / 12) * Math.PI * 2) * 100),
      y: Math.round(Math.sin((index / 12) * Math.PI * 2) * 100),
    }));
    expect(trimOutliers(ring)).toHaveLength(12);
  });
});

describe('centroidOf', () => {
  it('averages the points, and is the origin when there are none', () => {
    expect(centroidOf([{ x: 0, y: 0 }, { x: 4, y: 8 }])).toEqual({ x: 2, y: 4 });
    expect(centroidOf([])).toEqual({ x: 0, y: 0 });
  });
});

describe('communityHulls', () => {
  function node(id: string): GraphNode {
    return { id, labels: ['Thing'], properties: {} };
  }
  function edge(source: string, target: string): GraphEdge {
    return { id: `${source}-${target}`, source, target, relationshipType: 'LINKS', properties: {} };
  }

  it('outlines each community that is big enough to enclose, largest first', () => {
    const ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
    const scene = buildScene(ids.map(node), [edge('a', 'b'), edge('e', 'f')]);
    const analytics = runAnalytics(scene, {
      resolution: 1,
      includeCommunities: false,
      includePagerank: false,
      includeBetweenness: false,
    });
    ['a', 'b', 'c', 'd'].forEach((id) => analytics.communityOf.set(id, 0));
    ['e', 'f', 'g'].forEach((id) => analytics.communityOf.set(id, 1));

    const hulls = communityHulls(scene.display, analytics, 10);
    expect(hulls.map((hull) => hull.community)).toEqual([0, 1]);
    expect(hulls[0]?.nodeCount).toBe(4);
    expect(hulls[0]?.points.length).toBeGreaterThanOrEqual(3);
  });

  it('outlines only the nodes still on screen when a focus is narrowing the view', () => {
    const ids = ['a', 'b', 'c', 'd', 'e', 'f'];
    const scene = buildScene(ids.map(node), []);
    const analytics = runAnalytics(scene, {
      resolution: 1,
      includeCommunities: false,
      includePagerank: false,
      includeBetweenness: false,
    });
    ids.forEach((id) => analytics.communityOf.set(id, 0));

    expect(communityHulls(scene.display, analytics, 10, new Set(['a', 'b', 'c', 'd']))[0]?.nodeCount).toBe(4);
    expect(communityHulls(scene.display, analytics, 10, new Set(['a', 'b']))).toEqual([]);
  });

  it('skips communities too small to enclose and honours the limit', () => {
    const ids = ['a', 'b', 'c', 'd'];
    const scene = buildScene(ids.map(node), []);
    const analytics = runAnalytics(scene, {
      resolution: 1,
      includeCommunities: false,
      includePagerank: false,
      includeBetweenness: false,
    });
    analytics.communityOf.set('a', 0);
    analytics.communityOf.set('b', 0);
    analytics.communityOf.set('c', 1);
    analytics.communityOf.set('d', 2);
    expect(communityHulls(scene.display, analytics, 10)).toEqual([]);
  });
});
