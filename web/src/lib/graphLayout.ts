import { GRAPH_SCENE_LIMITS } from './bounds';
import type { AnalysisGraph, SceneGraph } from './graphScene';

export type LayoutTier = 'interactive' | 'staged';

export function layoutTierFor(nodeCount: number): LayoutTier {
  if (nodeCount <= GRAPH_SCENE_LIMITS.interactiveMaxNodes) return 'interactive';
  return 'staged';
}

/**
 * The backstop on how long the layout may run. The worker's cooling schedule normally finishes far
 * inside it; this only catches a run that never reports back, so a worker cannot spin for as long
 * as the tab is open.
 */
export function layoutBudgetMs(nodeCount: number): number {
  return Math.min(45_000, 8_000 + nodeCount * 1.2);
}

export interface LayoutRunner {
  start(): void;
  stop(): void;
  kill(): void;
  running(): boolean;
}

interface RunnerOptions {
  analysis: AnalysisGraph;
  display: SceneGraph;
  tier: LayoutTier;
  /** 2 lays the scene out flat for Sigma; 3 gives the solid view its depth. The worker's octree
   * already speaks both — this side only has to stride its buffers to match. */
  dimensions?: 2 | 3;
  /** Fired whenever positions land in the display graph, and once more when the run parks. */
  onChange: (running: boolean) => void;
}

/**
 * Runs the spring-electric layout off the main thread and lands its frames in the rendered graph.
 * The worker owns the physics and its own cooling schedule; this side owns the graphs. The layout
 * works on the simple undirected projection — fewer edges, same shape — and frames are coalesced
 * onto animation frames, so however fast the worker streams, Sigma re-indexes at most once per
 * painted frame.
 */
export function createLayoutRunner({ analysis, display, dimensions = 2, onChange }: RunnerOptions): LayoutRunner {
  if (analysis.order === 0) {
    return { start: () => undefined, stop: () => undefined, kill: () => undefined, running: () => false };
  }

  // The projection flattens once, up front: ids in insertion order, seeds from the graph's current
  // coordinates — which is what keeps a cluster merge visually continuous, its nodes start from the
  // centroids the merge anchored them on — and rendered radii so springs leave room for big discs.
  const ids = analysis.nodes();
  const index = new Map(ids.map((id, at) => [id, at]));
  const seeds = new Float32Array(ids.length * dimensions);
  const radii = new Float32Array(ids.length);
  ids.forEach((id, at) => {
    const { x, y, z } = analysis.getNodeAttributes(id);
    seeds[at * dimensions] = x;
    seeds[at * dimensions + 1] = y;
    // A scene that has only ever been flat seeds every depth at zero, and forces have no component
    // out of a plane every body shares — the layout would stay a sheet. A small deterministic
    // stagger breaks the symmetry; repulsion inflates it into a real volume from there.
    if (dimensions === 3) seeds[at * dimensions + 2] = z ?? ((at % 17) - 8) * 7;
    radii[at] = display.hasNode(id) ? display.getNodeAttribute(id, 'size') : 3;
  });
  const pairs: number[] = [];
  const multiplicities: number[] = [];
  analysis.forEachEdge((_edge, attributes, source, target) => {
    const sourceAt = index.get(source);
    const targetAt = index.get(target);
    if (sourceAt === undefined || targetAt === undefined) return;
    pairs.push(sourceAt, targetAt);
    multiplicities.push(attributes.weight);
  });

  let worker: Worker | undefined;
  let budgetTimer: number | undefined;
  let animationFrame: number | undefined;
  /** The newest positions the worker has sent and the canvas has not painted yet. */
  let pending: Float32Array | undefined;
  let active = false;
  let killed = false;
  const requestId = 1;

  const applyPositions = (positions: Float32Array) => {
    display.updateEachNodeAttributes((id, attributes) => {
      const at = index.get(id);
      if (at === undefined) return attributes;
      const moved = { ...attributes, x: positions[at * dimensions] ?? attributes.x, y: positions[at * dimensions + 1] ?? attributes.y };
      if (dimensions === 3) moved.z = positions[at * dimensions + 2] ?? attributes.z;
      return moved;
    });
    // The projection keeps the same coordinates, so whatever runs next — a merge, a re-style —
    // seeds from where the user last saw the nodes rather than from a stale snapshot.
    analysis.updateEachNodeAttributes((id, attributes) => {
      const at = index.get(id);
      if (at === undefined) return attributes;
      const moved = { ...attributes, x: positions[at * dimensions] ?? attributes.x, y: positions[at * dimensions + 1] ?? attributes.y };
      if (dimensions === 3) moved.z = positions[at * dimensions + 2] ?? attributes.z;
      return moved;
    });
  };

  const paint = () => {
    animationFrame = undefined;
    if (killed || !pending) return;
    const positions = pending;
    pending = undefined;
    applyPositions(positions);
    onChange(active);
  };

  const clearTimers = () => {
    if (budgetTimer !== undefined) window.clearTimeout(budgetTimer);
    if (animationFrame !== undefined) window.cancelAnimationFrame(animationFrame);
    budgetTimer = undefined;
    animationFrame = undefined;
  };

  const terminate = () => {
    worker?.terminate();
    worker = undefined;
  };

  const stop = () => {
    clearTimers();
    terminate();
    if (killed) return;
    active = false;
    // Whatever frame was still in flight is the best one there is; land it before parking.
    if (pending) {
      applyPositions(pending);
      pending = undefined;
    }
    onChange(false);
  };

  return {
    start: () => {
      if (killed || worker) return;
      try {
        worker = new Worker(new URL('../workers/layout.worker.ts', import.meta.url), { type: 'module' });
      } catch {
        // No worker, no layout: the seeded positions stay, exactly as when WebGL workers are barred.
        return;
      }
      active = true;
      worker.addEventListener('message', (event: MessageEvent<{ type: string; id: number; positions: Float32Array; done: boolean }>) => {
        if (event.data.type !== 'positions' || event.data.id !== requestId || killed) return;
        pending = event.data.positions;
        if (event.data.done) {
          clearTimers();
          terminate();
          active = false;
        }
        if (animationFrame === undefined) animationFrame = window.requestAnimationFrame(paint);
      });
      const edges = new Uint32Array(pairs);
      const weights = new Float32Array(multiplicities);
      worker.postMessage(
        { type: 'layout', id: requestId, dimensions, nodeCount: ids.length, edges, seeds, radii, weights },
        { transfer: [edges.buffer, seeds.buffer, radii.buffer, weights.buffer] },
      );
      budgetTimer = window.setTimeout(stop, layoutBudgetMs(ids.length));
      onChange(true);
    },
    stop,
    kill: () => {
      clearTimers();
      terminate();
      killed = true;
      pending = undefined;
    },
    running: () => active && !killed,
  };
}
