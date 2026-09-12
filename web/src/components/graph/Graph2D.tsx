import { layerBorder } from '@sigma/node-border';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Sigma from 'sigma';
import { extremityArrow, layerFill, pathCurved, pathLine, sdfDiamond, sdfSquare } from 'sigma/rendering';
import type { NodeDisplayData } from 'sigma/types';
import type { GraphEdge, GraphNode } from '../../types';
import { useCanvasPalette, type CanvasPalette } from '../../hooks/useCanvasPalette';
import { GRAPH_SCENE_LIMITS } from '../../lib/bounds';
import { communityHulls, type CommunityHull } from '../../lib/graphHulls';
import { createLayoutRunner, layoutTierFor, type LayoutRunner, type LayoutTier } from '../../lib/graphLayout';
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
  type Analytics,
  type SceneEdge,
  type SceneGraph,
  type SceneNode,
} from '../../lib/graphScene';
import type { GraphViewState } from '../../lib/graphView';
import type { GraphSelection } from '../../lib/graphSelection';
import { GraphControls } from './GraphControls';

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /**
   * The view, owned above.
   *
   * Colour by, Labels and Find nodes sit in the design's bar over the plot, not in the canvas, so
   * the state they change cannot live inside the canvas. The renderer reads it and reports back.
   */
  view: GraphViewState;
  onViewChange: (patch: Partial<GraphViewState>) => void;
  /**
   * The selection, owned above for the same reason the view is: what a click selects is read in the
   * annotation margin, which is neither this component nor a descendant of it.
   */
  selection?: GraphSelection;
  onSelectionChange: (selection: GraphSelection | undefined) => void;
  /** Double-clicking a node asks for its neighbourhood; the page runs the traversal. */
  onExpandNode?: (id: string) => void;
  /** The node whose traversal is in flight, drawn as the live one while it runs. */
  expandingNodeId?: string;
}

/**
 * Sigma v4 compiles one declarative WebGL program for the whole graph.
 *
 * The node mark is the design's own square — the same mark the rail keys, the menus and the
 * notices carry — and a merged community is the square stood on its corner, so an aggregate reads
 * as kin to the entities it holds. A zero-width outer border leaves ordinary entities as filled
 * marks; communities opt into the ring with the `borderSize` attribute their scene node carries,
 * and the selection draws the same ring in the rubric.
 */
const GRAPH_PRIMITIVES = {
  nodes: {
    shapes: [sdfSquare(), sdfDiamond()],
    layers: [
      layerFill(),
      layerBorder({
        borders: [
          {
            color: { attribute: 'borderColor', default: 'transparent' },
            size: { attribute: 'borderSize', default: 0 },
          },
          { color: { attribute: 'color' }, size: 0, fill: true },
        ],
      }),
    ],
  },
  edges: {
    paths: [pathLine(), pathCurved()],
    // The head is long and narrow so a hairline edge still states its direction: its size is a
    // multiple of the stroke's thickness, and the stroke is deliberately the quietest mark here.
    extremities: [extremityArrow({ lengthRatio: 4.5, widthRatio: 3 })],
  },
} as const;

/** The apparatus voice, for every label the canvas sets. */
const CANVAS_LABEL_FONT = 'Archivo, "Helvetica Neue", Arial, sans-serif';

interface VisualState {
  focusSet?: Set<string>;
  highlight?: Set<string>;
  pathEdges: Set<string>;
  selectedNodeId?: string;
  selectedEdgeId?: string;
  expandingNodeId?: string;
  pathFrom?: string;
  pathTo?: string;
  labelsAllowed: boolean;
  forceAllLabels: boolean;
  showEdges: boolean;
  curvedEdges: boolean;
  palette: CanvasPalette;
}

const TIER_LABEL: Record<LayoutTier, string> = {
  interactive: 'interactive layout',
  staged: 'staged layout',
};

export function Graph2D({ nodes, edges, view, onViewChange, selection, onSelectionChange, onExpandNode, expandingNodeId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma<SceneNode, SceneEdge> | undefined>(undefined);
  const runnerRef = useRef<LayoutRunner | undefined>(undefined);
  const palette = useCanvasPalette();
  const [panelOpen, setPanelOpen] = useState(false);
  const [pathEnds, setPathEnds] = useState<{ from?: string; to?: string }>({});
  const [rendererError, setRendererError] = useState<string>();
  /** A relationship is selected on the canvas but is not a node, so nothing on the canvas is lit. */
  const selectedNodeId = selection && selection.kind !== 'edge' ? selection.id : undefined;
  const selectedEdgeId = selection?.kind === 'edge' ? selection.id : undefined;

  /**
   * The scene, seeded from the one it replaces.
   *
   * A traversal appends to the result, which rebuilds this from scratch. Coordinates are carried
   * across so the nodes already on screen stay where the user last saw them and only the arrivals
   * move — the layout then relaxes a graph the reader recognises instead of dealing a new one.
   */
  const previousDisplayRef = useRef<SceneGraph | undefined>(undefined);
  const { scene, grown } = useMemo(() => {
    const previous = previousDisplayRef.current;
    const positions = new Map<string, { x: number; y: number }>();
    previous?.forEachNode((id, attributes) => positions.set(id, { x: attributes.x, y: attributes.y }));
    const built = buildScene(nodes, edges, positions);
    previousDisplayRef.current = built.display;
    // A scene that still holds everything the last one did is the last one, grown — a traversal
    // rather than a new question. The difference decides whether the camera is entitled to move.
    const kept = previous !== undefined && previous.order > 0
      && previous.nodes().every((id) => built.display.hasNode(id));
    return { scene: built, grown: kept };
  }, [nodes, edges]);

  // Every algorithm is opt-in: the default view needs only degree, which the graph already knows.
  // Louvain, PageRank and Brandes each run when — and only when — a control asks for them.
  const needsCommunities = view.clusterMode === 'merge' || view.colorBy === 'community';
  const analytics = useMemo(
    () => runAnalytics(scene, {
      resolution: view.resolution,
      includeCommunities: needsCommunities,
      includePagerank: view.sizeBy === 'pagerank',
      includeBetweenness: view.sizeBy === 'betweenness',
    }),
    [scene, view.resolution, view.sizeBy, needsCommunities],
  );

  const merged = view.clusterMode === 'merge';
  const rendered: SceneGraph = useMemo(
    () => (merged ? buildMergedGraph(scene.display, analytics) : scene.display),
    [merged, scene.display, analytics],
  );
  const renderedAnalysis = useMemo(
    () => (merged ? analysisForMerged(rendered) : scene.analysis),
    [merged, rendered, scene.analysis],
  );
  const tier = layoutTierFor(rendered.order);
  const grownRef = useRef(false);
  grownRef.current = grown;

  const betweennessAvailable = scene.display.order <= GRAPH_SCENE_LIMITS.betweennessMaxNodes;
  /**
   * Whether relationships can be clicked. Sigma allocates the picking buffer edges are hit-tested
   * against when it is constructed, not when the setting changes, so this has to be decided up front
   * — and Sigma rebuilt on the rare result that crosses the threshold. The buffer costs a second
   * render pass over every edge, which is worth it up to a point and not beyond it.
   */
  const edgeEventsEnabled = edges.length <= GRAPH_SCENE_LIMITS.edgeEventsMaxEdges;
  const communities = useMemo(
    () => (needsCommunities ? communitySummaries(scene.display, analytics, GRAPH_SCENE_LIMITS.legendMaxCommunities) : []),
    [needsCommunities, scene.display, analytics],
  );

  const focusSet = useMemo(() => {
    if (view.focusHops <= 0 || !selectedNodeId || !rendered.hasNode(selectedNodeId)) return undefined;
    return egoNodes(rendered, selectedNodeId, view.focusHops);
  }, [rendered, selectedNodeId, view.focusHops]);

  const searchMatches = useMemo(() => {
    const needle = view.search.trim().toLowerCase();
    if (needle.length === 0) return undefined;
    const matches = new Set<string>();
    rendered.forEachNode((id, attributes) => {
      if (attributes.label.toLowerCase().includes(needle) || attributes.primaryLabel.toLowerCase().includes(needle)) {
        matches.add(id);
      }
    });
    return matches;
  }, [rendered, view.search]);

  const path = useMemo(() => {
    if (!view.pathMode || !pathEnds.from || !pathEnds.to) return undefined;
    return shortestPath(renderedAnalysis, pathEnds.from, pathEnds.to);
  }, [view.pathMode, pathEnds.from, pathEnds.to, renderedAnalysis]);

  const pathNodes = useMemo(() => (path ? new Set(path) : undefined), [path]);
  const pathEdges = useMemo(() => {
    const keys = new Set<string>();
    if (!path) return keys;
    for (let step = 0; step + 1 < path.length; step += 1) {
      const from = path[step];
      const to = path[step + 1];
      if (from === undefined || to === undefined) continue;
      rendered.forEachEdge((edge, _attributes, source, target) => {
        if ((source === from && target === to) || (source === to && target === from)) keys.add(edge);
      });
    }
    return keys;
  }, [path, rendered]);

  const highlight = pathNodes ?? searchMatches;
  /** Whether focus, a path or a search is currently framing the view instead of the whole scene. */
  const narrowedRef = useRef(false);
  narrowedRef.current = Boolean(focusSet ?? pathNodes ?? searchMatches?.size);

  // Region outlines are geometry over live positions, so they are recomputed on the render pulse
  // rather than memoised on the graph: the layout moves nodes without changing any React input.
  const hullsEnabled = view.colorBy === 'community' && !merged && view.showHulls;
  const hullSourceRef = useRef<{ enabled: boolean; analytics: Analytics; visible?: Set<string> }>({ enabled: false, analytics });
  hullSourceRef.current = { enabled: hullsEnabled, analytics, visible: focusSet };
  const hullsRef = useRef<CommunityHull[]>([]);
  const hullsStaleRef = useRef(true);
  hullsStaleRef.current = true;
  const visualStateRef = useRef<VisualState>({
    pathEdges: new Set(),
    labelsAllowed: true,
    forceAllLabels: false,
    showEdges: true,
    curvedEdges: false,
    palette,
  });
  visualStateRef.current = {
    focusSet,
    highlight,
    pathEdges,
    selectedNodeId,
    selectedEdgeId,
    expandingNodeId,
    pathFrom: pathEnds.from,
    pathTo: pathEnds.to,
    labelsAllowed: view.labelMode !== 'off',
    forceAllLabels: view.labelMode === 'all' && rendered.order <= GRAPH_SCENE_LIMITS.forceLabelsMaxNodes,
    showEdges: view.showEdges,
    curvedEdges: view.curvedEdges,
    palette,
  };

  /**
   * Frames a set of nodes. Sigma's own reset fits node *centres* over the whole graph — it neither
   * ignores what a focus has hidden nor leaves room for a node's radius, so a merged cluster ends up
   * sliced against the canvas edge. Both cases are the same computation with a different node set.
   *
   * The camera works in Sigma's normalised "framed" space, so each corner is converted on the way
   * in. Positions come from the graph rather than from `getNodeDisplayData`, which reports raw
   * coordinates before Sigma has processed a change and normalised ones after: feeding those back
   * through the conversion normalises them twice and collapses the whole scene onto a point.
   */
  const frameNodes = useCallback((ids: Iterable<string>, padding = 1.4, duration = 420) => {
    const instance = sigmaRef.current;
    if (!instance) return;
    const keys = [...ids];
    // Sigma's normalisation is rebuilt when it renders, and the layout has usually just moved every
    // node, so measuring now would convert against the previous frame's scale and zoom to the wrong
    // depth. Asking for a render and measuring on its completion keeps the two in step.
    instance.once('afterRender', () => {
      const graph = instance.getGraph();
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const id of keys) {
        if (!graph.hasNode(id)) continue;
        const { x, y } = graph.getNodeAttributes(id);
        const framed = instance.viewportToFramedGraph(instance.graphToViewport({ x, y }));
        minX = Math.min(minX, framed.x);
        minY = Math.min(minY, framed.y);
        maxX = Math.max(maxX, framed.x);
        maxY = Math.max(maxY, framed.y);
      }
      if (!Number.isFinite(minX)) return;
      const ratio = Math.max(0.04, Math.max(maxX - minX, maxY - minY) * padding);
      void instance.getCamera().animate({ x: (minX + maxX) / 2, y: (minY + maxY) / 2, ratio }, { duration });
    });
    // A full refresh on purpose: indexation is the pass that rebuilds the normalisation this
    // measurement converts through, so skipping it would leave the scale a frame behind.
    instance.refresh();
  }, []);

  /** True while the view is still the layout's to frame — cleared as soon as the user pans or zooms. */
  const autoFitRef = useRef(true);
  const fitAll = useCallback((duration = 320) => {
    autoFitRef.current = true;
    const graph = sigmaRef.current?.getGraph();
    if (graph) frameNodes(graph.nodes(), 1.1, duration);
  }, [frameNodes]);
  const fitAllRef = useRef(fitAll);
  fitAllRef.current = fitAll;

  // One Sigma instance for the life of the mounted canvas; the graph it renders is swapped below.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let instance: Sigma<SceneNode, SceneEdge>;
    try {
      instance = new Sigma<SceneNode, SceneEdge>(scene.display, container, {
        primitives: GRAPH_PRIMITIVES,
        settings: {
          allowInvalidContainer: true,
          enableEdgeEvents: edgeEventsEnabled,
          labelDensity: 0.28,
          labelGridCellSize: 90,
          labelRenderedSizeThreshold: 5,
          // v4 defaults sizes to graph coordinates. Screen-relative sizes retain the incumbent
          // node, label and relationship weight at every camera depth.
          itemSizesReference: 'screen',
          // Relationships are drawn a hair thicker than they need to be, because the same thickness
          // is what Sigma picks against: a one-pixel line is visible but effectively unclickable.
          minEdgeThickness: 2.5,
          minCameraRatio: 0.02,
          maxCameraRatio: 24,
        },
        nodeReducer: (id, data, attributes, state) => {
          const current = visualStateRef.current;
          if (current.focusSet && !current.focusSet.has(id)) return { visibility: 'hidden' };
          const base: Partial<NodeDisplayData> = {
            // The design's mark: entities are the square everything else on the page carries;
            // a merged community is the same square stood on its corner.
            shape: attributes.kind === 'cluster' ? 'diamond' : 'square',
            cursor: 'pointer',
            labelColor: current.palette.foreground,
            labelFont: CANVAS_LABEL_FONT,
            labelSize: 10.5,
            // A label sits on a sliver of the ground, so it reads over relationships without
            // the halo the renderer would otherwise invent around it.
            labelBackgroundColor: current.palette.background,
            labelBackgroundPadding: 3,
            backdropVisibility: 'hidden',
          };
          if (id === current.expandingNodeId) {
            return {
              ...base,
              color: current.palette.live,
              size: data.size * 1.35,
              zIndex: 4,
              labelVisibility: 'visible',
            };
          }
          if (id === current.selectedNodeId || id === current.pathFrom || id === current.pathTo) {
            // Selection rules the mark in the rubric and raises its name on a plate of the
            // ground — the same panel grammar as the design's menus — while the fill keeps
            // saying what the node is.
            return {
              ...base,
              zIndex: 3,
              borderColor: current.palette.selected,
              borderSize: 0.24,
              labelVisibility: 'visible',
              labelSize: 11,
              labelBackgroundColor: 'transparent',
              backdropVisibility: 'visible',
              backdropArea: 'label',
              backdropColor: current.palette.background,
              backdropBorderColor: current.palette.selected,
              backdropBorderWidth: 1,
              backdropCornerRadius: 0,
              backdropPadding: 4,
              backdropShadowBlur: 0,
            };
          }
          if (current.highlight) {
            if (!current.highlight.has(id)) {
              return { ...base, color: withAlpha(current.palette.muted, 0.35), label: null, zIndex: 0 };
            }
            return {
              ...base,
              zIndex: 2,
              labelVisibility: current.labelsAllowed ? 'visible' : 'hidden',
            };
          }
          return {
            ...base,
            zIndex: state.isHovered ? 2 : undefined,
            labelVisibility: !current.labelsAllowed ? 'hidden' : current.forceAllLabels ? 'visible' : 'auto',
          };
        },
        edgeReducer: (id, data) => {
          const current = visualStateRef.current;
          const path = current.curvedEdges ? 'curved' : 'line';
          // Written on every edge because the renderer's display default is a headless stroke —
          // the declaration-level default never reaches edges that pass through a reducer.
          const head = 'arrow';
          if (!current.showEdges) return { visibility: 'hidden', path, head };
          if (current.selectedEdgeId === id) {
            return { color: current.palette.selected, size: Math.max(3, data.size * 2.2), zIndex: 4, path, head };
          }
          if (current.pathEdges.has(id)) {
            return { color: current.palette.selected, size: Math.max(2.5, data.size * 2), zIndex: 3, path, head };
          }
          if (current.highlight) {
            return { color: withAlpha(current.palette.muted, 0.18), path, head };
          }
          return { color: withAlpha(current.palette.edge, 0.5), path, head };
        },
      });
    } catch (cause) {
      setRendererError(cause instanceof Error && /webgl/i.test(cause.message)
        ? 'WebGL is unavailable. Use the 3D Graph or Table view on this browser.'
        : 'The 2D renderer could not start. Use the 3D Graph or Table view.');
      return;
    }
    sigmaRef.current = instance;

    // Community regions paint underneath the edges, in their own layer, on Sigma's render pulse.
    const hullCanvas = instance.createCanvas('community-hulls', { beforeLayer: 'stage' });
    const hullContext = hullCanvas.getContext('2d');
    const sizeHullCanvas = () => {
      const ratio = Math.min(2, window.devicePixelRatio);
      const { width, height } = instance.getDimensions();
      hullCanvas.setAttribute('width', `${width * ratio}px`);
      hullCanvas.setAttribute('height', `${height * ratio}px`);
      hullContext?.setTransform(ratio, 0, 0, ratio, 0, 0);
    };
    sizeHullCanvas();

    const drawHulls = () => {
      if (!hullContext) return;
      const { width, height } = instance.getDimensions();
      hullContext.clearRect(0, 0, width, height);
      if (hullsStaleRef.current) {
        const { enabled, analytics: current, visible } = hullSourceRef.current;
        hullsRef.current = enabled
          ? communityHulls(instance.getGraph(), current, GRAPH_SCENE_LIMITS.legendMaxCommunities, visible)
          : [];
        hullsStaleRef.current = false;
      }
      hullsRef.current.forEach((hull) => {
        if (hull.points.length < 3) return;
        hullContext.beginPath();
        hull.points.forEach((point, index) => {
          const viewport = instance.graphToViewport(point);
          if (index === 0) hullContext.moveTo(viewport.x, viewport.y);
          else hullContext.lineTo(viewport.x, viewport.y);
        });
        hullContext.closePath();
        hullContext.fillStyle = withAlpha(hull.color, 0.1);
        hullContext.fill();
        hullContext.strokeStyle = withAlpha(hull.color, 0.4);
        hullContext.lineWidth = 1;
        hullContext.stroke();
      });
    };

    const releaseCamera = () => { autoFitRef.current = false; };
    instance.getMouseCaptor().on('mousedown', releaseCamera);
    instance.getMouseCaptor().on('wheel', releaseCamera);
    instance.getTouchCaptor().on('touchdown', releaseCamera);
    instance.on('afterRender', drawHulls);
    instance.on('resize', sizeHullCanvas);

    return () => {
      instance.getMouseCaptor().off('mousedown', releaseCamera);
      instance.getMouseCaptor().off('wheel', releaseCamera);
      instance.getTouchCaptor().off('touchdown', releaseCamera);
      instance.off('afterRender', drawHulls);
      instance.off('resize', sizeHullCanvas);
      instance.kill();
      sigmaRef.current = undefined;
    };
    // Sigma owns its WebGL context for the mounted container; graph and settings update below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [edgeEventsEnabled]);

  useEffect(() => {
    const instance = sigmaRef.current;
    if (!instance || instance.getGraph() === rendered) return;
    instance.setGraph(rendered);
    // A new result is framed from scratch. A result that grew is not: the reader is looking at the
    // node they just opened, and yanking the camera out to fit the arrivals loses the place they
    // were reading. The layout's own fit still runs afterwards if they never took the camera over.
    if (!grownRef.current) fitAll(0);
  }, [rendered, fitAll]);

  // Entity styling is derived, so it is reapplied rather than stored; a merged graph is styled
  // where it is built, because its nodes do not exist outside that build.
  useEffect(() => {
    if (merged) return;
    styleScene(scene.display, analytics, { colorBy: view.colorBy, sizeBy: view.sizeBy });
  }, [merged, scene.display, analytics, view.colorBy, view.sizeBy]);

  useEffect(() => {
    const runner = createLayoutRunner({
      analysis: renderedAnalysis,
      display: rendered,
      tier,
      onChange: (running) => {
        hullsStaleRef.current = true;
        // Where the nodes ended up is only known once the run parks, and the framing chosen before
        // it started no longer describes them — so the fit happens here, unless the user has taken
        // the camera over in the meantime or is holding a narrowed view of their own.
        if (!running && autoFitRef.current && !narrowedRef.current) fitAllRef.current();
      },
    });
    runnerRef.current = runner;
    runner.start();
    return () => {
      runner.kill();
      runnerRef.current = undefined;
    };
  }, [rendered, renderedAnalysis, tier]);

  useEffect(() => {
    const instance = sigmaRef.current;
    if (!instance) return;
    const showAllLabels = view.labelMode === 'all';
    instance.setSettings({
      // Always on at the renderer level: the Labels toggle governs ordinary nodes through the
      // reducer, and the selected node's name must survive it being off.
      renderLabels: true,
      renderEdgeLabels: false,
      labelDensity: showAllLabels ? 2.4 : 0.28,
      hideEdgesOnMove: rendered.size > GRAPH_SCENE_LIMITS.edgeEventsMaxEdges,
      hideLabelsOnMove: rendered.order > GRAPH_SCENE_LIMITS.labelMaxNodes,
    });
    // Reducers are constructor-level in v4 and read the current React-derived state through a ref.
    // Refreshing without indexation updates only those visual decisions.
    instance.refresh({ skipIndexation: true });
  }, [rendered, focusSet, highlight, pathEdges, selectedNodeId, selectedEdgeId, expandingNodeId, pathEnds.from, pathEnds.to, view.labelMode, view.showEdges, view.curvedEdges, palette]);

  // Regions are painted from Sigma's render pulse, and toggling them changes nothing Sigma watches,
  // so the repaint has to be asked for. Without this the outlines only appear on the next pan.
  useEffect(() => {
    hullsStaleRef.current = true;
    sigmaRef.current?.refresh({ skipIndexation: true });
  }, [hullsEnabled, focusSet, analytics]);

  const selectNode = useCallback((id: string) => {
    const attributes = rendered.getNodeAttributes(id);
    onSelectionChange(attributes.kind === 'cluster'
      ? {
          kind: 'cluster',
          id,
          community: attributes.community,
          primaryLabel: attributes.primaryLabel,
          memberCount: attributes.memberCount,
          internalEdges: attributes.internalEdges,
          members: attributes.members,
          color: attributes.color,
        }
      : { kind: 'node', id, color: attributes.color });
  }, [rendered, onSelectionChange]);

  /**
   * Keeps the margin's swatches on the colour the canvas is actually using.
   *
   * Colouring by community after selecting a node repaints the disc; the panel beside it would go
   * on showing the colour the node had when it was clicked. Styling is applied in the effect above
   * this one, so by the time this runs the attributes are current. It only ever writes a selection
   * whose colour differs, which is what stops it re-entering.
   */
  useEffect(() => {
    if (!selection) return;
    if (selection.kind === 'edge') {
      if (!rendered.hasEdge(selection.id)) return;
      const [source, target] = rendered.extremities(selection.id);
      const sourceColor = rendered.getNodeAttribute(source, 'color');
      const targetColor = rendered.getNodeAttribute(target, 'color');
      if (sourceColor === selection.sourceColor && targetColor === selection.targetColor) return;
      onSelectionChange({ ...selection, sourceColor, targetColor });
      return;
    }
    if (!rendered.hasNode(selection.id)) return;
    const color = rendered.getNodeAttribute(selection.id, 'color');
    if (color === selection.color) return;
    onSelectionChange({ ...selection, color });
  }, [selection, rendered, onSelectionChange, view.colorBy, view.sizeBy, analytics]);

  useEffect(() => {
    const instance = sigmaRef.current;
    if (!instance) return;
    const clickNode = ({ node }: { node: string }) => {
      if (view.pathMode) {
        setPathEnds((current) => (current.from && !current.to && current.from !== node
          ? { from: current.from, to: node }
          : { from: node }));
      }
      selectNode(node);
    };
    const clickEdge = ({ edge }: { edge: string }) => {
      onSelectionChange({ kind: 'edge', id: edge });
    };
    const clickStage = () => onSelectionChange(undefined);
    /**
     * Double-click traverses. Sigma's own double-click zooms the camera, which would fight the
     * frame the arriving neighbourhood asks for, so the default is refused before the traversal
     * starts. A merged cluster is not a node in the graph and has nothing to traverse from.
     */
    const expandNode = (payload: { node: string; event: { preventSigmaDefault: () => void } }) => {
      payload.event.preventSigmaDefault();
      if (rendered.getNodeAttribute(payload.node, 'kind') === 'cluster') return;
      onExpandNode?.(payload.node);
    };
    instance.on('clickNode', clickNode);
    instance.on('clickEdge', clickEdge);
    instance.on('clickStage', clickStage);
    instance.on('doubleClickNode', expandNode);
    return () => {
      instance.off('clickNode', clickNode);
      instance.off('clickEdge', clickEdge);
      instance.off('clickStage', clickStage);
      instance.off('doubleClickNode', expandNode);
    };
  }, [rendered, selectNode, onSelectionChange, onExpandNode, view.pathMode]);

  // A highlight is invisible when what it marks is three screens away, so the camera goes to it:
  // to the whole ego neighbourhood or path when there is one, to the first match when searching.
  useEffect(() => {
    if (focusSet) frameNodes(focusSet);
    else if (pathNodes) frameNodes(pathNodes);
    else if (searchMatches?.size) frameNodes(searchMatches);
  }, [focusSet, pathNodes, searchMatches, frameNodes]);

  const changeView = useCallback((patch: Partial<GraphViewState>) => {
    onViewChange(patch);
    if (patch.pathMode === false) setPathEnds({});
  }, [onViewChange]);



  const pathStatus = view.pathMode
    ? path
      ? `Path found: ${path.length - 1} hops`
      : pathEnds.from && pathEnds.to
        ? 'No path between those two nodes'
        : pathEnds.from
          ? 'Pick the second node'
          : 'Pick the first node'
    : undefined;

  const searchStatus = searchMatches ? `${searchMatches.size.toLocaleString()} matching nodes` : undefined;

  return (
    <div className="graph-canvas-wrap">
      {rendererError && <div className="renderer-error" role="alert">{rendererError}</div>}
      <div className="graph-canvas" ref={containerRef} role="img" aria-label="Interactive 2D query result graph" />
      <GraphControls
        state={view}
        onChange={changeView}
        open={panelOpen}
        onOpenChange={setPanelOpen}
        layoutTierLabel={TIER_LABEL[tier]}
        onFit={fitAll}
        betweennessAvailable={betweennessAvailable}
        focusAvailable={selectedNodeId !== undefined}
        communities={communities}
        communityCount={analytics.communityCount}
        modularity={analytics.modularity}
        shownNodes={rendered.order}
        shownEdges={rendered.size}
        pathStatus={pathStatus}
        searchStatus={searchStatus}
      />
    </div>
  );
}
