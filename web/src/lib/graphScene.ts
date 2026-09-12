import { MultiDirectedGraph, UndirectedGraph } from 'graphology';
import louvain from 'graphology-communities-louvain';
import betweennessCentrality from 'graphology-metrics/centrality/betweenness';
import pagerank from 'graphology-metrics/centrality/pagerank';
import { bidirectional } from 'graphology-shortest-path/unweighted';
import type { GraphEdge, GraphNode } from '../types';
import { GRAPH_SCENE_LIMITS } from './bounds';
import { nodeDisplayLabel } from './format';

export type ClusterMode = 'off' | 'merge';
export type ColorBy = 'label' | 'community' | 'degree';
export type SizeBy = 'uniform' | 'degree' | 'pagerank' | 'betweenness';

/**
 * One rendered node. `sourceIndex` points back into the query result so a click can open the real
 * record; a cluster node stands for many results at once and carries -1 instead.
 */
export interface SceneNode {
  x: number;
  y: number;
  /** Only the 3D view reads or writes depth; a scene that has never been solid simply has none. */
  z?: number;
  size: number;
  color: string;
  label: string;
  kind: 'entity' | 'cluster';
  sourceIndex: number;
  primaryLabel: string;
  community: number;
  memberCount: number;
  members: string[];
  internalEdges: number;
  [attribute: string]: unknown;
}

export interface SceneEdge {
  size: number;
  color: string;
  type: string;
  sourceIndex: number;
  relationshipType: string;
  weight: number;
  aggregateCount: number;
  [attribute: string]: unknown;
}

export type SceneGraph = MultiDirectedGraph<SceneNode, SceneEdge>;
/** Simple, undirected, weighted projection: what every algorithm and the layout actually run on. */
export type AnalysisGraph = UndirectedGraph<{ x: number; y: number; z?: number }, { weight: number }>;

export interface Scene {
  display: SceneGraph;
  analysis: AnalysisGraph;
}

export interface Analytics {
  communityOf: Map<string, number>;
  communityCount: number;
  modularity: number;
  degreeOf: Map<string, number>;
  pagerankOf: Map<string, number>;
  /** Absent above `GRAPH_SCENE_LIMITS.betweennessMaxNodes` — Brandes is O(V·E) and would stall. */
  betweennessOf?: Map<string, number>;
  maxDegree: number;
  maxPagerank: number;
  maxBetweenness: number;
}

export interface CommunitySummary {
  community: number;
  color: string;
  nodeCount: number;
  dominantLabel: string;
  anchorLabel: string;
}

/**
 * Re-expresses any colour the palette or a scale can produce as a translucent one. Theme variables
 * arrive as hex, community colours as `hsl()`, so dimming has to handle both.
 */
export function withAlpha(color: string, alpha: number): string {
  const trimmed = color.trim();
  for (const notation of ['hsl', 'rgb'] as const) {
    if (!trimmed.startsWith(`${notation}(`)) continue;
    // Legacy comma syntax and modern space syntax cannot be mixed, so the separator follows the input.
    const separator = trimmed.includes(',') ? `, ${alpha}` : ` / ${alpha}`;
    return trimmed.replace(`${notation}(`, `${notation}a(`).replace(')', `${separator})`);
  }
  if (trimmed.startsWith('#')) {
    const hex = trimmed.slice(1);
    const expanded = hex.length === 3 ? [...hex].map((character) => `${character}${character}`).join('') : hex;
    if (expanded.length >= 6) {
      const red = Number.parseInt(expanded.slice(0, 2), 16);
      const green = Number.parseInt(expanded.slice(2, 4), 16);
      const blue = Number.parseInt(expanded.slice(4, 6), 16);
      return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
    }
  }
  return trimmed;
}

/**
 * HSL is the natural space for generating a palette, but Sigma's WebGL programs only parse hex and
 * `rgb()`, and silently render anything else black — so every colour leaves here as hex.
 */
export function hslToHex(hue: number, saturation: number, lightness: number): string {
  const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
  const secondary = chroma * (1 - Math.abs(((hue / 60) % 2) - 1));
  const match = lightness - chroma / 2;
  const sextant = Math.floor((((hue % 360) + 360) % 360) / 60);
  const [red, green, blue] = [
    [chroma, secondary, 0], [secondary, chroma, 0], [0, chroma, secondary],
    [0, secondary, chroma], [secondary, 0, chroma], [chroma, 0, secondary],
  ][sextant] ?? [0, 0, 0];
  const channel = (value: number) => Math.round((value + match) * 255).toString(16).padStart(2, '0');
  return `#${channel(red ?? 0)}${channel(green ?? 0)}${channel(blue ?? 0)}`;
}

/**
 * The categorical inks, in the design's register.
 *
 * The first is the counter — the steel the design already writes schema and provenance in — and
 * the rest sit at its saturation and weight, spaced far apart in hue so ten categories stay ten.
 * Two hue families are deliberately absent: the rubric's red-orange, which belongs to selection
 * and to trouble and must never be a category a node merely *is*; and pure greys, which mean
 * "unlabelled". Mid lightness keeps every ink legible on the plate and the page alike. Past ten,
 * the wheel comes round again lighter, then darker, so forty categories still differ.
 */
const CATEGORY_HUES = [204, 45, 265, 150, 335, 95, 224, 180, 68, 290] as const;

export function paletteColor(index: number): string {
  const bounded = Math.max(0, Math.trunc(index));
  const slot = bounded % CATEGORY_HUES.length;
  const wrap = Math.floor(bounded / CATEGORY_HUES.length) % 3;
  const lightness = wrap === 1 ? 0.66 : wrap === 2 ? 0.47 : 0.56;
  return hslToHex(CATEGORY_HUES[slot] ?? 204, 0.36 - wrap * 0.05, lightness);
}

/**
 * Colours the labels present in one result. Hashing a label straight onto the wheel would throw
 * away the even spacing — two labels can hash to indices whose hues land on top of each other —
 * so the distinct labels are numbered densely first, and only then coloured.
 */
export function labelColorScale(labels: Iterable<string>): Map<string, string> {
  const distinct = [...new Set(labels)].filter((label) => label.length > 0).sort();
  return new Map(distinct.map((label, index) => [label, paletteColor(index)]));
}

/** Stands in until the renderer applies the theme's relationship colour. */
export const EDGE_FALLBACK_COLOR = 'rgba(128, 138, 158, 0.45)';

/** The colour for a node whose label carries no colour of its own. */
export const UNLABELLED_COLOR = hslToHex(220, 0.1, 0.55);

/**
 * Cool-to-warm ramp for a value already normalized to 0..1. It runs from the counter's steel to
 * the rubric's warmth — the two poles the design already means "reference" and "hot" by — at the
 * categorical inks' own saturation, so a degree-coloured scene reads as the same drawing.
 */
function rampColor(amount: number): string {
  const clamped = Math.max(0, Math.min(1, amount));
  return hslToHex(204 - clamped * 180, 0.32 + clamped * 0.22, 0.58 - clamped * 0.04);
}

/** Communities are numbered from zero, so they can index the palette directly and stay distinct. */
export function communityColor(community: number): string {
  return paletteColor(Math.max(0, community));
}

/**
 * Seeds positions on a sunflower spiral. Force Atlas 2 needs distinct, non-degenerate starting
 * coordinates; a spiral also means the first frame is readable before a single iteration lands.
 */
function seedPosition(index: number, total: number): { x: number; y: number } {
  const angle = index * Math.PI * (3 - Math.sqrt(5));
  const radius = Math.max(30, Math.sqrt(total) * 14) * Math.sqrt((index + 0.5) / Math.max(1, total));
  return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
}

export type NodePositions = ReadonlyMap<string, { x: number; y: number; z?: number }>;

/**
 * Where a node arriving into an existing scene should start.
 *
 * Expanding a node adds its neighbours to a graph the user is already reading. Seeding those
 * arrivals on the spiral would scatter them across the canvas and let the layout drag the whole
 * scene apart to reach them, so a node whose relationship reaches something already placed starts
 * beside it instead — the neighbourhood then unfolds out of the node that was opened. The offset
 * is the golden angle over the arrival's own index, which separates siblings without randomness.
 */
function anchoredSeed(
  id: string,
  index: number,
  total: number,
  anchors: Map<string, { x: number; y: number; count: number }>,
): { x: number; y: number } {
  const anchor = anchors.get(id);
  if (!anchor || anchor.count === 0) return seedPosition(index, total);
  const angle = index * Math.PI * (3 - Math.sqrt(5));
  return {
    x: anchor.x / anchor.count + Math.cos(angle) * ARRIVAL_RADIUS,
    y: anchor.y / anchor.count + Math.sin(angle) * ARRIVAL_RADIUS,
  };
}

/** How far from its placed neighbour a newly arrived node starts, before the layout takes over. */
const ARRIVAL_RADIUS = 46;

/**
 * Builds the rendered and analysis graphs for one result.
 *
 * `previous` carries the coordinates the user is currently looking at. A result that grew — a
 * double-click that pulled in a neighbourhood — rebuilds this scene from scratch, and without those
 * coordinates every node would jump to a fresh spiral seed and the layout would resettle the whole
 * canvas, which reads as a new query rather than as a traversal.
 */
export function buildScene(nodes: GraphNode[], edges: GraphEdge[], previous?: NodePositions): Scene {
  const display: SceneGraph = new MultiDirectedGraph();
  const analysis: AnalysisGraph = new UndirectedGraph();
  const labelColors = labelColorScale(nodes.map((node) => node.labels[0] ?? ''));

  // Arrivals are placed against the nodes already on screen, so the anchors are collected before
  // any node is added: an arrival's relationship may point either way, and to a node added later.
  const anchors = new Map<string, { x: number; y: number; count: number }>();
  if (previous && previous.size > 0) {
    const place = (arrival: string, placed: string) => {
      const position = previous.get(placed);
      if (!position || previous.has(arrival)) return;
      const anchor = anchors.get(arrival) ?? { x: 0, y: 0, count: 0 };
      anchor.x += position.x;
      anchor.y += position.y;
      anchor.count += 1;
      anchors.set(arrival, anchor);
    };
    edges.forEach((edge) => {
      place(edge.source, edge.target);
      place(edge.target, edge.source);
    });
  }

  nodes.forEach((node, index) => {
    if (display.hasNode(node.id)) return;
    const seed = previous?.get(node.id) ?? anchoredSeed(node.id, index, nodes.length, anchors);
    const primaryLabel = node.labels[0] ?? '';
    display.addNode(node.id, {
      ...seed,
      size: BASE_NODE_SIZE,
      color: labelColors.get(primaryLabel) ?? UNLABELLED_COLOR,
      label: nodeDisplayLabel(node),
      kind: 'entity',
      sourceIndex: index,
      primaryLabel,
      community: 0,
      memberCount: 1,
      members: [],
      internalEdges: 0,
    });
    analysis.addNode(node.id, seed);
  });

  edges.forEach((edge, index) => {
    if (!display.hasNode(edge.source) || !display.hasNode(edge.target) || display.hasEdge(edge.id)) return;
    display.addDirectedEdgeWithKey(edge.id, edge.source, edge.target, {
      size: 1,
      color: EDGE_FALLBACK_COLOR,
      type: 'line',
      sourceIndex: index,
      relationshipType: edge.relationshipType,
      weight: 1,
      aggregateCount: 1,
    });
    // The analysis projection is simple and undirected: parallel and reciprocal relationships
    // become one weighted link, which is what Louvain and Brandes expect.
    if (edge.source === edge.target) return;
    const existing = analysis.edge(edge.source, edge.target);
    if (existing === undefined) analysis.addUndirectedEdge(edge.source, edge.target, { weight: 1 });
    else analysis.setEdgeAttribute(existing, 'weight', analysis.getEdgeAttribute(existing, 'weight') + 1);
  });

  return { display, analysis };
}

function emptyAnalytics(display: SceneGraph): Analytics {
  const degreeOf = new Map<string, number>();
  display.forEachNode((id) => degreeOf.set(id, display.degree(id)));
  return {
    communityOf: new Map(display.nodes().map((id) => [id, 0])),
    communityCount: display.order > 0 ? 1 : 0,
    modularity: 0,
    degreeOf,
    pagerankOf: new Map(display.nodes().map((id) => [id, 0])),
    maxDegree: Math.max(1, ...degreeOf.values()),
    maxPagerank: 1,
    maxBetweenness: 1,
  };
}

export interface AnalyticsOptions {
  resolution: number;
  includeCommunities: boolean;
  includePagerank: boolean;
  includeBetweenness: boolean;
}

/**
 * Each metric is computed only when a control is asking for it. Degree comes free with the graph;
 * Louvain, PageRank and Brandes all cost real time on a 50,000-node result, and paying for a metric
 * nothing on screen is using would make every unrelated control feel slow.
 */
export function runAnalytics(scene: Scene, options: AnalyticsOptions): Analytics {
  const { display, analysis } = scene;
  if (display.order === 0) return emptyAnalytics(display);

  const analytics = emptyAnalytics(display);
  if (options.includeCommunities && analysis.size > 0 && analysis.order > 1) {
    const detected = louvain.detailed(analysis, { getEdgeWeight: 'weight', resolution: options.resolution });
    analytics.communityCount = detected.count;
    analytics.modularity = Number.isFinite(detected.modularity) ? detected.modularity : 0;
    Object.entries(detected.communities).forEach(([id, community]) => analytics.communityOf.set(id, community));
  }

  if (options.includePagerank && display.size > 0) {
    const ranks = pagerank(display, { getEdgeWeight: 'weight' });
    Object.entries(ranks).forEach(([id, rank]) => analytics.pagerankOf.set(id, rank));
    analytics.maxPagerank = Math.max(Number.EPSILON, ...analytics.pagerankOf.values());
  }

  if (options.includeBetweenness && analysis.order <= GRAPH_SCENE_LIMITS.betweennessMaxNodes && analysis.size > 0) {
    const scores = betweennessCentrality(analysis, { getEdgeWeight: null, normalized: true });
    const betweennessOf = new Map<string, number>();
    Object.entries(scores).forEach(([id, score]) => betweennessOf.set(id, score));
    analytics.betweennessOf = betweennessOf;
    analytics.maxBetweenness = Math.max(Number.EPSILON, ...betweennessOf.values());
  }

  return analytics;
}

function metricFor(id: string, analytics: Analytics, sizeBy: SizeBy): { value: number; maximum: number } {
  if (sizeBy === 'degree') return { value: analytics.degreeOf.get(id) ?? 0, maximum: analytics.maxDegree };
  if (sizeBy === 'pagerank') return { value: analytics.pagerankOf.get(id) ?? 0, maximum: analytics.maxPagerank };
  if (sizeBy === 'betweenness') return { value: analytics.betweennessOf?.get(id) ?? 0, maximum: analytics.maxBetweenness };
  return { value: 0, maximum: 1 };
}

/**
 * The rendered radii, in Sigma's node-size units.
 *
 * Half a percent of the canvas is not a node anyone can aim at: at the shipped scale the discs read
 * as specks against their own labels, and a double-click has to land on one. Every radius here is
 * the scene's original one raised by half again, which keeps the ratio between a hub and a leaf
 * exactly as it was and only changes how much of the canvas the graph claims.
 */
const BASE_NODE_SIZE = 6;
const UNIFORM_NODE_SIZE = 3.75;
const METRIC_NODE_FLOOR = 3;
const METRIC_NODE_SPAN = 12;

export function entitySize(id: string, analytics: Analytics, sizeBy: SizeBy): number {
  if (sizeBy === 'uniform') return UNIFORM_NODE_SIZE;
  const { value, maximum } = metricFor(id, analytics, sizeBy);
  return METRIC_NODE_FLOOR + Math.sqrt(Math.max(0, value) / Math.max(Number.EPSILON, maximum)) * METRIC_NODE_SPAN;
}

export function entityColor(
  id: string,
  node: SceneNode,
  analytics: Analytics,
  colorBy: ColorBy,
  labelColors: Map<string, string>,
): string {
  if (colorBy === 'community') return communityColor(analytics.communityOf.get(id) ?? 0);
  if (colorBy === 'degree') return rampColor((analytics.degreeOf.get(id) ?? 0) / Math.max(1, analytics.maxDegree));
  return labelColors.get(node.primaryLabel) ?? UNLABELLED_COLOR;
}

/** Restyles the rendered graph in place. Sigma re-reads attributes on the next refresh. */
export function styleScene(
  display: SceneGraph,
  analytics: Analytics,
  options: { colorBy: ColorBy; sizeBy: SizeBy },
): void {
  const colorBy = options.colorBy;
  const labelColors = labelColorScale(display.mapNodes((_id, attributes) => attributes.primaryLabel));
  display.updateEachNodeAttributes((id, attributes) => {
    if (attributes.kind === 'cluster') return attributes;
    return {
      ...attributes,
      community: analytics.communityOf.get(id) ?? 0,
      size: entitySize(id, analytics, options.sizeBy),
      color: entityColor(id, attributes, analytics, colorBy, labelColors),
    };
  });
}

/**
 * Collapses every community into a single node, the way Semantica's grouped view does: intra-community
 * relationships fold into the cluster's own weight, and every crossing relationship becomes one
 * aggregated link whose thickness is how many relationships it stands for.
 */
export function buildMergedGraph(display: SceneGraph, analytics: Analytics): SceneGraph {
  const merged: SceneGraph = new MultiDirectedGraph();
  const clusterKey = (community: number) => `cluster:${community}`;
  const members = new Map<number, string[]>();
  const labelCounts = new Map<number, Map<string, number>>();
  const centroids = new Map<number, { x: number; y: number; z: number }>();

  display.forEachNode((id, attributes) => {
    const community = analytics.communityOf.get(id) ?? 0;
    const bucket = members.get(community);
    if (bucket) bucket.push(id);
    else members.set(community, [id]);
    const counts = labelCounts.get(community) ?? new Map<string, number>();
    const label = attributes.primaryLabel || 'Node';
    counts.set(label, (counts.get(label) ?? 0) + 1);
    labelCounts.set(community, counts);
    const centroid = centroids.get(community) ?? { x: 0, y: 0, z: 0 };
    centroid.x += attributes.x;
    centroid.y += attributes.y;
    centroid.z += attributes.z ?? 0;
    centroids.set(community, centroid);
  });

  members.forEach((ids, community) => {
    const counts = labelCounts.get(community) ?? new Map<string, number>();
    const dominant = [...counts.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ?? 'Node';
    const centroid = centroids.get(community) ?? { x: 0, y: 0, z: 0 };
    // Anchoring the cluster on its members' centroid keeps the merge visually continuous: blobs
    // appear where their nodes already were rather than jumping to a fresh random layout.
    merged.addNode(clusterKey(community), {
      x: centroid.x / ids.length,
      y: centroid.y / ids.length,
      z: centroid.z / ids.length,
      size: Math.min(51, 9 + Math.sqrt(ids.length) * 3.6),
      color: communityColor(community),
      label: `${dominant} · ${ids.length.toLocaleString()}`,
      kind: 'cluster',
      type: 'cluster',
      borderColor: withAlpha(communityColor(community), 0.55),
      borderSize: 0.12,
      sourceIndex: -1,
      primaryLabel: dominant,
      community,
      memberCount: ids.length,
      members: ids,
      internalEdges: 0,
    });
  });

  const crossing = new Map<string, { source: string; target: string; count: number; types: Map<string, number> }>();
  display.forEachEdge((_edge, attributes, source, target) => {
    const from = analytics.communityOf.get(source) ?? 0;
    const to = analytics.communityOf.get(target) ?? 0;
    if (from === to) {
      merged.updateNodeAttribute(clusterKey(from), 'internalEdges', (value) => (typeof value === 'number' ? value : 0) + 1);
      return;
    }
    // The pair of community numbers, as one key. The separator used to be a literal NUL, which
    // made every text tool on the repository classify this file as binary — `grep` skips it, so a
    // search for anything in this file silently returned nothing. A colon cannot occur in a decimal
    // community number, so it separates the pair just as unambiguously and keeps the file text.
    const key = `${from}:${to}`;
    const bundle = crossing.get(key) ?? { source: clusterKey(from), target: clusterKey(to), count: 0, types: new Map() };
    bundle.count += 1;
    bundle.types.set(attributes.relationshipType, (bundle.types.get(attributes.relationshipType) ?? 0) + 1);
    crossing.set(key, bundle);
  });

  crossing.forEach((bundle, key) => {
    const dominantType = [...bundle.types.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ?? '';
    merged.addDirectedEdgeWithKey(`cluster-edge:${key}`, bundle.source, bundle.target, {
      size: Math.min(7, 0.8 + Math.log2(bundle.count + 1) * 0.7),
      color: EDGE_FALLBACK_COLOR,
      type: 'line',
      sourceIndex: -1,
      relationshipType: dominantType,
      weight: bundle.count,
      aggregateCount: bundle.count,
    });
  });

  return merged;
}

/** The layout projection for a merged graph: cluster nodes linked by their aggregate weights. */
export function analysisForMerged(merged: SceneGraph): AnalysisGraph {
  const analysis: AnalysisGraph = new UndirectedGraph();
  merged.forEachNode((id, attributes) => analysis.addNode(id, { x: attributes.x, y: attributes.y, z: attributes.z }));
  merged.forEachEdge((_edge, attributes, source, target) => {
    if (source === target) return;
    const existing = analysis.edge(source, target);
    if (existing === undefined) analysis.addUndirectedEdge(source, target, { weight: attributes.weight });
    else analysis.setEdgeAttribute(existing, 'weight', analysis.getEdgeAttribute(existing, 'weight') + attributes.weight);
  });
  return analysis;
}

/** Every node within `hops` relationships of `root`, direction-agnostic. */
export function egoNodes(display: SceneGraph, root: string, hops: number): Set<string> {
  const reached = new Set<string>();
  if (!display.hasNode(root)) return reached;
  reached.add(root);
  let frontier = [root];
  for (let hop = 0; hop < hops && frontier.length > 0; hop += 1) {
    const next: string[] = [];
    frontier.forEach((id) => display.forEachNeighbor(id, (neighbor) => {
      if (reached.has(neighbor)) return;
      reached.add(neighbor);
      next.push(neighbor);
    }));
    frontier = next;
  }
  return reached;
}

/** Shortest hop path between two nodes, ignoring direction. */
export function shortestPath(analysis: AnalysisGraph, from: string, to: string): string[] | undefined {
  if (!analysis.hasNode(from) || !analysis.hasNode(to)) return undefined;
  return bidirectional(analysis, from, to) ?? undefined;
}

export function communitySummaries(display: SceneGraph, analytics: Analytics, limit: number): CommunitySummary[] {
  const buckets = new Map<number, { count: number; labels: Map<string, number>; anchor: string; anchorScore: number }>();
  display.forEachNode((id, attributes) => {
    if (attributes.kind === 'cluster') return;
    const community = analytics.communityOf.get(id) ?? 0;
    const bucket = buckets.get(community) ?? { count: 0, labels: new Map<string, number>(), anchor: attributes.label, anchorScore: -1 };
    bucket.count += 1;
    const label = attributes.primaryLabel || 'Node';
    bucket.labels.set(label, (bucket.labels.get(label) ?? 0) + 1);
    const score = analytics.degreeOf.get(id) ?? 0;
    if (score > bucket.anchorScore) {
      bucket.anchor = attributes.label;
      bucket.anchorScore = score;
    }
    buckets.set(community, bucket);
  });

  return [...buckets.entries()]
    .map(([community, bucket]) => ({
      community,
      color: communityColor(community),
      nodeCount: bucket.count,
      dominantLabel: [...bucket.labels.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ?? 'Node',
      anchorLabel: bucket.anchor,
    }))
    .sort((left, right) => right.nodeCount - left.nodeCount || left.community - right.community)
    .slice(0, limit);
}
