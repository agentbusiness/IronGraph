import type { Analytics, SceneGraph } from './graphScene';
import { communityColor } from './graphScene';

export interface Point { x: number; y: number }

export interface CommunityHull {
  community: number;
  color: string;
  nodeCount: number;
  points: Point[];
  centroid: Point;
}

/** Andrew's monotone chain, counter-clockwise, without the duplicated closing point. */
export function convexHull(points: Point[]): Point[] {
  if (points.length < 3) return [...points];
  const sorted = [...points].sort((left, right) => left.x - right.x || left.y - right.y);
  const cross = (origin: Point, a: Point, b: Point) =>
    (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);

  const half = (source: Point[]): Point[] => {
    const chain: Point[] = [];
    source.forEach((point) => {
      while (chain.length >= 2) {
        const last = chain[chain.length - 1];
        const previous = chain[chain.length - 2];
        if (!last || !previous || cross(previous, last, point) > 0) break;
        chain.pop();
      }
      chain.push(point);
    });
    chain.pop();
    return chain;
  };

  return [...half(sorted), ...half([...sorted].reverse())];
}

/** Pushes each hull vertex out from the centroid so the outline clears the nodes it wraps. */
export function expandHull(points: Point[], centroid: Point, padding: number): Point[] {
  return points.map((point) => {
    const dx = point.x - centroid.x;
    const dy = point.y - centroid.y;
    const distance = Math.hypot(dx, dy);
    if (distance < 1e-6) return { x: point.x + padding, y: point.y };
    return { x: point.x + (dx / distance) * padding, y: point.y + (dy / distance) * padding };
  });
}

export function centroidOf(points: Point[]): Point {
  if (points.length === 0) return { x: 0, y: 0 };
  return points.reduce(
    (total, point) => ({ x: total.x + point.x / points.length, y: total.y + point.y / points.length }),
    { x: 0, y: 0 },
  );
}

/**
 * Drops the members a community has lent to its neighbours. A convex hull is decided entirely by
 * its extremes, so a handful of nodes pulled across the canvas by their cross-community links would
 * stretch the region into a spike covering everything in between — the outline stops describing
 * where the community is. Keeping the bulk within the given quantile of the centroid distance
 * leaves the shape on the nodes that actually sit together.
 */
export function trimOutliers(points: Point[], quantile = 0.9, tolerance = 1.15): Point[] {
  if (points.length < 4) return points;
  const centroid = centroidOf(points);
  const distances = points.map((point) => Math.hypot(point.x - centroid.x, point.y - centroid.y));
  const cutoff = [...distances].sort((left, right) => left - right)[
    Math.min(distances.length - 1, Math.floor(distances.length * quantile))
  ] ?? 0;
  const limit = cutoff * tolerance;
  const kept = points.filter((_point, index) => (distances[index] ?? 0) <= limit);
  return kept.length >= 3 ? kept : points;
}

/**
 * One outline per community, largest first. Communities of one or two nodes get no hull: a shape
 * that small reads as noise rather than a region.
 *
 * `visible` narrows the outlines to the nodes actually on screen. A focus hides most of the graph,
 * and a region drawn around members that are no longer being rendered describes a shape the user
 * cannot see the reason for.
 */
export function communityHulls(
  display: SceneGraph,
  analytics: Analytics,
  limit: number,
  visible?: ReadonlySet<string>,
): CommunityHull[] {
  const grouped = new Map<number, Point[]>();
  display.forEachNode((id, attributes) => {
    if (attributes.kind === 'cluster') return;
    if (visible && !visible.has(id)) return;
    const community = analytics.communityOf.get(id) ?? 0;
    const bucket = grouped.get(community);
    if (bucket) bucket.push({ x: attributes.x, y: attributes.y });
    else grouped.set(community, [{ x: attributes.x, y: attributes.y }]);
  });

  return [...grouped.entries()]
    .filter(([, points]) => points.length >= 3)
    .sort((left, right) => right[1].length - left[1].length)
    .slice(0, limit)
    .map(([community, points]) => {
      const core = trimOutliers(points);
      const centroid = centroidOf(core);
      return {
        community,
        color: communityColor(community),
        nodeCount: points.length,
        centroid,
        points: expandHull(convexHull(core), centroid, 14),
      };
    });
}
