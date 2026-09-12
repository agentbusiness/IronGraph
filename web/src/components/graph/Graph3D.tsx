import {
  Box3,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  Color,
  ConeGeometry,
  DirectionalLight,
  DoubleSide,
  DynamicDrawUsage,
  EdgesGeometry,
  Group,
  HemisphereLight,
  InstancedMesh,
  LineBasicMaterial,
  LineSegments,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshLambertMaterial,
  OctahedronGeometry,
  PerspectiveCamera,
  Quaternion,
  Raycaster,
  Scene,
  Vector2,
  Vector3,
  WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { GraphEdge, GraphNode } from '../../types';
import { useCanvasPalette, type CanvasPalette } from '../../hooks/useCanvasPalette';
import { GRAPH_SCENE_LIMITS } from '../../lib/bounds';
import { perspectiveFitDistance } from '../../lib/graphGeometry';
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
  type SceneGraph,
} from '../../lib/graphScene';
import type { GraphViewState } from '../../lib/graphView';
import type { GraphSelection } from '../../lib/graphSelection';
import { GraphControls } from './GraphControls';

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** The view, owned above — the same object the 2D plot reads, so switching views keeps it. */
  view: GraphViewState;
  onViewChange: (patch: Partial<GraphViewState>) => void;
  /** The selection, owned above: what a click selects is read in the annotation margin. */
  selection?: GraphSelection;
  onSelectionChange: (selection: GraphSelection | undefined) => void;
  /** Double-clicking a node asks for its neighbourhood; the page runs the traversal. */
  onExpandNode?: (id: string) => void;
  /** The node whose traversal is in flight, drawn as the live one while it runs. */
  expandingNodeId?: string;
}

/**
 * The design's marks, extruded.
 *
 * On the page an entity is the square everything else carries, and a merged community is that
 * square stood on its corner. In depth the same two marks become the cube and the octahedron —
 * every silhouette an octahedron throws is the diamond, exactly as every silhouette a cube throws
 * on the picture plane is the square. Nothing here invents a third shape.
 */
const ENTITY_SCALE = 3.4;
/** An octahedron of circumradius r has a waist square of side r·√2; this matches the cube's face. */
const CLUSTER_SCALE = 1.2;
/** How much the rubric outline stands off the mark it rules. */
const OUTLINE_SCALE = 1.32;
/** The scale a live traversal raises its node by — the same factor the 2D reducer uses. */
const EXPANDING_SCALE = 1.35;
/** Curved relationships are sampled polylines; eight chords read as an arc at any depth. */
const CURVE_SEGMENTS = 8;
/** Arrowhead proportions, in world units: long and narrow so a hairline still states direction. */
const ARROW_LENGTH = 4.6;
const ARROW_RADIUS = 1.5;

const TIER_LABEL: Record<LayoutTier, string> = {
  interactive: 'interactive layout',
  staged: 'staged layout',
};

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
  palette: CanvasPalette;
}

interface Runtime {
  renderer: WebGLRenderer;
  scene: Scene;
  camera: PerspectiveCamera;
  controls: OrbitControls;
  raycaster: Raycaster;
  resizeObserver: ResizeObserver;
  animation: number;
  resizeFrame?: number;
  /** A camera move in flight: fit and frame ease rather than jump, like the 2D camera. */
  tween?: { fromPosition: Vector3; fromTarget: Vector3; toPosition: Vector3; toTarget: Vector3; start: number; duration: number };
}

interface EdgeRecord {
  key: string;
  source: number;
  target: number;
  aggregate: number;
}

/**
 * Everything built for one rendered graph, and the writers that keep it current.
 *
 * Positions move on the layout's pulse and colours on React's, so the two are separate writers
 * over shared buffers rather than one rebuild: a layout frame rewrites fifty thousand matrices
 * without touching a colour, and a selection recolours without touching a position.
 */
interface SceneObjects {
  ids: string[];
  index: Map<string, number>;
  records: EdgeRecord[];
  /** Record indices currently drawn — the focus filter works by rewriting this, not the buffers. */
  order: number[];
  overlayOrder: number[];
  segments: number;
  clustered: boolean;
  arrowsEnabled: boolean;
  positions: Float32Array;
  nodeMesh: InstancedMesh;
  edgeLines: LineSegments;
  edgePositions: BufferAttribute;
  edgeColors: BufferAttribute;
  arrows: InstancedMesh;
  overlayLines: LineSegments;
  overlayPositions: BufferAttribute;
  outlines: [LineSegments, LineSegments, LineSegments];
  hulls: Group;
  layoutRunning: boolean;
}

interface LabelSlot {
  at: number;
  element: HTMLDivElement;
  offset: number;
}

function disposeObject(object: Mesh | LineSegments | InstancedMesh): void {
  object.geometry.dispose();
  const material = object.material;
  if (Array.isArray(material)) material.forEach((entry) => entry.dispose());
  else material.dispose();
}

function clearHulls(hulls: Group): void {
  for (const child of [...hulls.children]) {
    hulls.remove(child);
    if (child instanceof Mesh || child instanceof LineSegments) disposeObject(child);
  }
}

export function Graph3D({ nodes, edges, view, onViewChange, selection, onSelectionChange, onExpandNode, expandingNodeId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const runtimeRef = useRef<Runtime | undefined>(undefined);
  const objectsRef = useRef<SceneObjects | undefined>(undefined);
  const runnerRef = useRef<LayoutRunner | undefined>(undefined);
  const palette = useCanvasPalette();
  const [panelOpen, setPanelOpen] = useState(false);
  const [pathEnds, setPathEnds] = useState<{ from?: string; to?: string }>({});
  const [rendererError, setRendererError] = useState<string>();
  const selectedNodeId = selection && selection.kind !== 'edge' ? selection.id : undefined;
  const selectedEdgeId = selection?.kind === 'edge' ? selection.id : undefined;

  /**
   * The scene, seeded from the one it replaces — coordinates carried across a traversal so only
   * the arrivals move. Depth is carried with them: the layout runner staggers any node that has
   * never had a z, so a scene arriving from the 2D view inflates out of its plane on first layout.
   */
  const previousDisplayRef = useRef<SceneGraph | undefined>(undefined);
  const { scene, grown } = useMemo(() => {
    const previous = previousDisplayRef.current;
    const positions = new Map<string, { x: number; y: number; z?: number }>();
    previous?.forEachNode((id, attributes) => positions.set(id, { x: attributes.x, y: attributes.y, z: attributes.z }));
    const built = buildScene(nodes, edges, positions);
    previousDisplayRef.current = built.display;
    const kept = previous !== undefined && previous.order > 0
      && previous.nodes().every((id) => built.display.hasNode(id));
    return { scene: built, grown: kept };
  }, [nodes, edges]);

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
  /** Whether relationships can be clicked — the same budget the 2D picking buffer honours. */
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
  const narrowedRef = useRef(false);
  narrowedRef.current = Boolean(focusSet ?? pathNodes ?? searchMatches?.size);

  const hullsEnabled = view.colorBy === 'community' && !merged && view.showHulls;
  const visualStateRef = useRef<VisualState>({
    pathEdges: new Set(),
    labelsAllowed: true,
    forceAllLabels: false,
    showEdges: true,
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
    palette,
  };
  const renderedRef = useRef(rendered);
  renderedRef.current = rendered;
  const analyticsRef = useRef(analytics);
  analyticsRef.current = analytics;
  const hullsEnabledRef = useRef(false);
  hullsEnabledRef.current = hullsEnabled;

  /** The pooled label plates, and where each one sits this frame. Projection runs on the render. */
  const labelSlotsRef = useRef<LabelSlot[]>([]);
  const labelPoolRef = useRef<HTMLDivElement[]>([]);
  const hoverRef = useRef<number>(-1);
  const requestRenderRef = useRef<() => void>(() => undefined);

  /** Whether an entity or cluster is hidden by the current focus. Scale-zero instances draw nothing. */
  const hiddenAt = useCallback((at: number): boolean => {
    const objects = objectsRef.current;
    const focus = visualStateRef.current.focusSet;
    if (!objects || !focus) return false;
    const id = objects.ids[at];
    return id !== undefined && !focus.has(id);
  }, []);

  /**
   * Frames a set of nodes: their bounding box, held at the camera's current bearing, at the
   * distance the frustum needs to contain it. The same computation serves Fit, a fresh result,
   * and the focus/path/search framings — only the node set differs, exactly as in the 2D plot.
   */
  const frame = useCallback((ids: Iterable<string>, padding = 1.2, duration = 420) => {
    const runtime = runtimeRef.current;
    const graph = renderedRef.current;
    if (!runtime) return;
    const box = new Box3();
    const point = new Vector3();
    let any = false;
    for (const id of ids) {
      if (!graph.hasNode(id)) continue;
      const attributes = graph.getNodeAttributes(id);
      box.expandByPoint(point.set(attributes.x, attributes.y, attributes.z ?? 0));
      any = true;
    }
    if (!any) return;
    box.expandByScalar(graph.order > 1_000 ? 3 : 5);
    const center = box.getCenter(new Vector3());
    const size = box.getSize(new Vector3());
    const distance = Math.max(45, perspectiveFitDistance(size.x, size.y, size.z, runtime.camera.aspect, runtime.camera.fov, padding));
    // A camera written with a non-finite number renders nothing at all and cannot be recovered by
    // any later frame — so a frame that cannot be computed is refused rather than applied.
    if (!Number.isFinite(distance) || !Number.isFinite(center.x) || !Number.isFinite(center.y) || !Number.isFinite(center.z)) return;
    const direction = runtime.camera.position.clone().sub(runtime.controls.target);
    if (direction.lengthSq() < 0.001) direction.set(0, 0, 1);
    const toPosition = center.clone().add(direction.normalize().multiplyScalar(distance));
    runtime.camera.near = Math.max(0.1, distance / 2_000);
    runtime.camera.far = Math.max(20_000, distance * 20);
    runtime.camera.updateProjectionMatrix();
    if (!(duration > 0)) {
      runtime.tween = undefined;
      runtime.controls.target.copy(center);
      runtime.camera.position.copy(toPosition);
      runtime.controls.update();
    } else {
      runtime.tween = {
        fromPosition: runtime.camera.position.clone(),
        fromTarget: runtime.controls.target.clone(),
        toPosition,
        toTarget: center,
        start: performance.now(),
        duration,
      };
    }
    requestRenderRef.current();
  }, []);

  /** True while the view is still the layout's to frame — cleared as soon as the user takes over. */
  const autoFitRef = useRef(true);
  const fitAll = useCallback((duration = 320) => {
    autoFitRef.current = true;
    frame(renderedRef.current.nodes(), 1.15, duration);
  }, [frame]);
  const fitAllRef = useRef(fitAll);
  fitAllRef.current = fitAll;

  // One renderer for the life of the mounted canvas; the graph it draws is swapped below.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const scene3 = new Scene();
    scene3.background = new Color(palette.background);
    const camera = new PerspectiveCamera(48, 1, 0.1, 20_000);
    camera.position.set(0, 0, 360);
    let renderer: WebGLRenderer;
    try {
      renderer = new WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    } catch {
      const frameId = requestAnimationFrame(() => setRendererError('WebGL is unavailable. Use the Plot or Table view on this browser.'));
      return () => cancelAnimationFrame(frameId);
    }
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    renderer.outputColorSpace = 'srgb';
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controls.minDistance = 15;
    controls.maxDistance = 8_000;
    const releaseCamera = () => {
      autoFitRef.current = false;
      const runtime = runtimeRef.current;
      if (runtime) runtime.tween = undefined;
    };
    controls.addEventListener('start', releaseCamera);

    // A mark's colour is what says which kind of thing it is, so it has to survive being turned
    // away from the light: the fill light is nearly as strong underneath as above, and the key
    // only models the form on top of it. A face in shadow reads darker, never black.
    scene3.add(new HemisphereLight(0xffffff, 0xb4bcc8, 1.4));
    const keyLight = new DirectionalLight(0xffffff, 0.5);
    keyLight.position.set(150, 220, 300);
    scene3.add(keyLight);

    /** Puts every pooled label plate on its node's screen position, or hides it. */
    const placeLabels = () => {
      const objects = objectsRef.current;
      const overlay = overlayRef.current;
      if (!overlay) return;
      const rect = renderer.domElement.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      const point = new Vector3();
      const forward = camera.getWorldDirection(new Vector3());
      // Half the vertical field per pixel: what one world unit is worth on screen at a depth.
      const worldPerPixel = 2 * Math.tan(camera.fov * Math.PI / 360) / rect.height;
      for (const slot of labelSlotsRef.current) {
        if (!objects) break;
        const base = slot.at * 3;
        point.set(objects.positions[base] ?? 0, objects.positions[base + 1] ?? 0, objects.positions[base + 2] ?? 0);
        const depth = point.distanceTo(camera.position);
        const behind = point.clone().sub(camera.position).dot(forward) <= 0;
        point.project(camera);
        if (behind || point.x < -1.05 || point.x > 1.05 || point.y < -1.05 || point.y > 1.05) {
          slot.element.style.visibility = 'hidden';
          continue;
        }
        // The plate sits to the right of the mark it names, the way Sigma writes the 2D labels:
        // the mark's world radius converted to pixels at this depth, plus a hairline of clearance.
        const clearance = Math.min(90, slot.offset / Math.max(1e-6, depth * worldPerPixel)) + 5;
        const x = (point.x * 0.5 + 0.5) * rect.width;
        const y = (-point.y * 0.5 + 0.5) * rect.height;
        slot.element.style.visibility = 'visible';
        slot.element.style.transform = `translate(${(x + clearance).toFixed(1)}px, ${y.toFixed(1)}px) translate(0, -50%)`;
      }
    };

    const render = () => {
      renderer.render(scene3, camera);
      placeLabels();
    };

    const animate = () => {
      const runtime = runtimeRef.current;
      if (!runtime) return;
      runtime.animation = 0;
      let easing = false;
      const tween = runtime.tween;
      if (tween) {
        const t = Math.min(1, (performance.now() - tween.start) / tween.duration);
        const eased = 1 - (1 - t) ** 3;
        camera.position.lerpVectors(tween.fromPosition, tween.toPosition, eased);
        controls.target.lerpVectors(tween.fromTarget, tween.toTarget, eased);
        if (t >= 1) runtime.tween = undefined;
        else easing = true;
      }
      // Damping is the controls' own inertia, and it works by writing the camera position every
      // update. While a fit is easing, this loop owns that position — leaving damping on has the
      // two writing over each other, so the ease never arrives and the frame loop never parks.
      controls.enableDamping = !easing;
      const moving = controls.update();
      controls.enableDamping = true;
      render();
      if (moving || easing) runtime.animation = requestAnimationFrame(animate);
    };
    const requestRender = () => {
      const runtime = runtimeRef.current;
      if (runtime && runtime.animation === 0) runtime.animation = requestAnimationFrame(animate);
    };
    requestRenderRef.current = requestRender;
    controls.addEventListener('change', requestRender);

    const resizeObserver = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const width = Math.max(1, entry.contentRect.width);
      const height = Math.max(1, entry.contentRect.height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      const runtime = runtimeRef.current;
      if (!runtime) return;
      requestRender();
      if (!autoFitRef.current) return;
      if (runtime.resizeFrame !== undefined) cancelAnimationFrame(runtime.resizeFrame);
      runtime.resizeFrame = requestAnimationFrame(() => {
        runtime.resizeFrame = undefined;
        if (autoFitRef.current && !narrowedRef.current) fitAllRef.current(0);
      });
    });
    resizeObserver.observe(container);

    const contextLost = (event: Event) => {
      event.preventDefault();
      setRendererError('The WebGL context was lost. Use the Plot or Table view while the device recovers.');
    };
    renderer.domElement.addEventListener('webglcontextlost', contextLost);

    const runtime: Runtime = { renderer, scene: scene3, camera, controls, raycaster: new Raycaster(), resizeObserver, animation: 0 };
    runtimeRef.current = runtime;
    requestRender();

    return () => {
      renderer.domElement.removeEventListener('webglcontextlost', contextLost);
      cancelAnimationFrame(runtime.animation);
      if (runtime.resizeFrame !== undefined) cancelAnimationFrame(runtime.resizeFrame);
      resizeObserver.disconnect();
      controls.removeEventListener('start', releaseCamera);
      controls.removeEventListener('change', requestRender);
      controls.dispose();
      // A browser grants a page only a handful of WebGL contexts, and this view shares them with
      // the 2D canvas. Disposing returns the resources but not the context itself; losing it
      // deliberately hands the slot back, so switching views repeatedly cannot exhaust them.
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
      runtimeRef.current = undefined;
      requestRenderRef.current = () => undefined;
      labelSlotsRef.current = [];
      labelPoolRef.current = [];
    };
    // Renderer resources live for the mounted canvas; palette and graph updates land below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Entity styling is derived, so it is reapplied rather than stored — the same pass the 2D runs.
  useEffect(() => {
    if (merged) return;
    styleScene(scene.display, analytics, { colorBy: view.colorBy, sizeBy: view.sizeBy });
  }, [merged, scene.display, analytics, view.colorBy, view.sizeBy]);


  /** Samples one relationship into `target` starting at vertex `vertex`; returns the tip direction. */
  const sampleEdge = useCallback((
    objects: SceneObjects,
    record: EdgeRecord,
    target: BufferAttribute,
    vertex: number,
    scratch: { a: Vector3; b: Vector3; control: Vector3; step: Vector3; last: Vector3 },
  ): Vector3 => {
    const { positions, segments } = objects;
    const { a, b, control, step, last } = scratch;
    a.fromArray(positions, record.source * 3);
    b.fromArray(positions, record.target * 3);
    if (segments === 1) {
      target.setXYZ(vertex, a.x, a.y, a.z);
      target.setXYZ(vertex + 1, b.x, b.y, b.z);
      return last.copy(b).sub(a).normalize();
    }
    // A quadratic arc bowed to the left of travel, so a relationship and its reciprocal part ways
    // instead of writing over each other — the exact reason the 2D view offers curved edges.
    step.copy(b).sub(a);
    const length = step.length();
    control.set(0, 1, 0).cross(step);
    if (control.lengthSq() < 1e-6) control.set(1, 0, 0).cross(step);
    control.normalize().multiplyScalar(Math.min(22, length * 0.18));
    control.x += (a.x + b.x) / 2;
    control.y += (a.y + b.y) / 2;
    control.z += (a.z + b.z) / 2;
    let previousX = a.x;
    let previousY = a.y;
    let previousZ = a.z;
    for (let s = 1; s <= segments; s += 1) {
      const t = s / segments;
      const u = 1 - t;
      const x = u * u * a.x + 2 * u * t * control.x + t * t * b.x;
      const y = u * u * a.y + 2 * u * t * control.y + t * t * b.y;
      const z = u * u * a.z + 2 * u * t * control.z + t * t * b.z;
      target.setXYZ(vertex + (s - 1) * 2, previousX, previousY, previousZ);
      target.setXYZ(vertex + (s - 1) * 2 + 1, x, y, z);
      last.set(x - previousX, y - previousY, z - previousZ);
      previousX = x;
      previousY = y;
      previousZ = z;
    }
    return last.normalize();
  }, []);

  /** How far the surface of a mark sits from its centre, for arrows, labels and picking. */
  const markRadius = useCallback((size: number, clustered: boolean): number => (
    size * (clustered ? CLUSTER_SCALE : ENTITY_SCALE * 0.62)
  ), []);

  /** Node instance matrices, outline placement and the flat position array everything else reads. */
  const writeMatrices = useCallback(() => {
    const objects = objectsRef.current;
    if (!objects) return;
    const graph = renderedRef.current;
    const current = visualStateRef.current;
    const matrix = new Matrix4();
    const quaternion = new Quaternion();
    const position = new Vector3();
    const scale = new Vector3();
    const factor = objects.clustered ? CLUSTER_SCALE : ENTITY_SCALE;
    graph.forEachNode((id, attributes) => {
      const at = objects.index.get(id);
      if (at === undefined) return;
      objects.positions[at * 3] = attributes.x;
      objects.positions[at * 3 + 1] = attributes.y;
      objects.positions[at * 3 + 2] = attributes.z ?? 0;
      const hidden = current.focusSet !== undefined && !current.focusSet.has(id);
      let side = hidden ? 0 : attributes.size * factor;
      if (id === current.expandingNodeId) side *= EXPANDING_SCALE;
      position.set(attributes.x, attributes.y, attributes.z ?? 0);
      matrix.compose(position, quaternion, scale.set(side, side, side));
      objects.nodeMesh.setMatrixAt(at, matrix);
    });
    objects.nodeMesh.instanceMatrix.needsUpdate = true;

    const ruled: (string | undefined)[] = [current.selectedNodeId, current.pathFrom, current.pathTo];
    objects.outlines.forEach((outline, slot) => {
      const id = ruled[slot];
      const at = id === undefined ? undefined : objects.index.get(id);
      if (id === undefined || at === undefined || (current.focusSet && !current.focusSet.has(id))) {
        outline.visible = false;
        return;
      }
      const side = graph.getNodeAttribute(id, 'size') * factor * OUTLINE_SCALE;
      outline.position.set(objects.positions[at * 3] ?? 0, objects.positions[at * 3 + 1] ?? 0, objects.positions[at * 3 + 2] ?? 0);
      outline.scale.set(side, side, side);
      outline.visible = true;
    });
  }, []);

  /** Relationship geometry: positions for every drawn segment, arrow matrices, overlay positions. */
  const writeEdgeGeometry = useCallback(() => {
    const objects = objectsRef.current;
    if (!objects) return;
    const graph = renderedRef.current;
    const scratch = { a: new Vector3(), b: new Vector3(), control: new Vector3(), step: new Vector3(), last: new Vector3() };
    const matrix = new Matrix4();
    const quaternion = new Quaternion();
    const up = new Vector3(0, 1, 0);
    const position = new Vector3();
    const scale = new Vector3(1, 1, 1);
    let vertex = 0;
    let arrow = 0;
    for (const at of objects.order) {
      const record = objects.records[at];
      if (record === undefined) continue;
      const direction = sampleEdge(objects, record, objects.edgePositions, vertex, scratch);
      vertex += objects.segments * 2;
      if (!objects.arrowsEnabled) continue;
      const targetId = objects.ids[record.target];
      const size = targetId === undefined ? 3 : graph.getNodeAttribute(targetId, 'size');
      const surface = markRadius(size, objects.clustered);
      position.set(
        (objects.positions[record.target * 3] ?? 0) - direction.x * (surface + ARROW_LENGTH / 2),
        (objects.positions[record.target * 3 + 1] ?? 0) - direction.y * (surface + ARROW_LENGTH / 2),
        (objects.positions[record.target * 3 + 2] ?? 0) - direction.z * (surface + ARROW_LENGTH / 2),
      );
      quaternion.setFromUnitVectors(up, direction);
      objects.arrows.setMatrixAt(arrow, matrix.compose(position, quaternion, scale));
      arrow += 1;
    }
    objects.edgePositions.needsUpdate = true;
    objects.edgeLines.geometry.setDrawRange(0, objects.order.length * objects.segments * 2);
    objects.arrows.count = arrow;
    objects.arrows.instanceMatrix.needsUpdate = true;

    let overlayVertex = 0;
    for (const at of objects.overlayOrder) {
      const record = objects.records[at];
      if (record === undefined) continue;
      sampleEdge(objects, record, objects.overlayPositions, overlayVertex, scratch);
      overlayVertex += objects.segments * 2;
    }
    objects.overlayPositions.needsUpdate = true;
    objects.overlayLines.geometry.setDrawRange(0, overlayVertex);
  }, [sampleEdge, markRadius]);

  /** Which relationships are drawn at all, and which the rubric overlay restates. */
  const rebuildOrder = useCallback(() => {
    const objects = objectsRef.current;
    if (!objects) return;
    const current = visualStateRef.current;
    objects.order = [];
    objects.overlayOrder = [];
    objects.records.forEach((record, at) => {
      if (current.focusSet) {
        const source = objects.ids[record.source];
        const target = objects.ids[record.target];
        if (source === undefined || target === undefined) return;
        if (!current.focusSet.has(source) || !current.focusSet.has(target)) return;
      }
      objects.order.push(at);
      if (record.key === current.selectedEdgeId || current.pathEdges.has(record.key)) objects.overlayOrder.push(at);
    });
    const needed = objects.overlayOrder.length * objects.segments * 6;
    if (objects.overlayPositions.array.length < needed) {
      objects.overlayPositions = new BufferAttribute(new Float32Array(needed * 2), 3);
      objects.overlayLines.geometry.setAttribute('position', objects.overlayPositions);
    }
    objects.edgeLines.visible = current.showEdges;
    objects.arrows.visible = current.showEdges;
    objects.overlayLines.visible = current.showEdges && objects.overlayOrder.length > 0;
  }, []);

  /** Node fills, relationship inks and arrow inks — colours only, never positions. */
  const writeColors = useCallback(() => {
    const objects = objectsRef.current;
    if (!objects) return;
    const graph = renderedRef.current;
    const current = visualStateRef.current;
    const background = new Color(current.palette.background);
    // The design dims by translucency; over an opaque ground the same statement is a blend toward
    // it, which keeps every line and fill a solid colour the depth buffer can be honest about.
    const dimmedNode = new Color(current.palette.muted).lerp(background, 0.65);
    const dimmedEdge = new Color(current.palette.muted).lerp(background, 0.82);
    const normalEdge = new Color(current.palette.edge).lerp(background, 0.5);
    const liveColor = new Color(current.palette.live);
    const rubric = new Color(current.palette.selected);
    const color = new Color();

    graph.forEachNode((id, attributes) => {
      const at = objects.index.get(id);
      if (at === undefined) return;
      if (id === current.expandingNodeId) color.copy(liveColor);
      else if (current.highlight && !current.highlight.has(id)) color.copy(dimmedNode);
      else color.set(attributes.color);
      objects.nodeMesh.setColorAt(at, color);
    });
    if (objects.nodeMesh.instanceColor) objects.nodeMesh.instanceColor.needsUpdate = true;

    let vertex = 0;
    let arrow = 0;
    for (const at of objects.order) {
      const record = objects.records[at];
      if (record === undefined) continue;
      const marked = record.key === current.selectedEdgeId || current.pathEdges.has(record.key);
      if (marked) color.copy(rubric);
      else if (current.highlight) color.copy(dimmedEdge);
      else if (record.aggregate > 1) {
        // A merged link stands for many relationships; thickness is not available to a hairline,
        // so its weight is stated the way the 2D states it in size — here, in ink.
        color.copy(normalEdge).lerp(new Color(current.palette.edge), Math.min(0.7, Math.log2(record.aggregate + 1) * 0.12));
      } else color.copy(normalEdge);
      for (let s = 0; s < objects.segments * 2; s += 1) objects.edgeColors.setXYZ(vertex + s, color.r, color.g, color.b);
      vertex += objects.segments * 2;
      if (objects.arrowsEnabled) {
        objects.arrows.setColorAt(arrow, marked ? rubric : color);
        arrow += 1;
      }
    }
    objects.edgeColors.needsUpdate = true;
    if (objects.arrows.instanceColor) objects.arrows.instanceColor.needsUpdate = true;
    (objects.overlayLines.material as LineBasicMaterial).color.copy(rubric);
    objects.outlines.forEach((outline) => (outline.material as LineBasicMaterial).color.copy(rubric));
  }, []);

  /**
   * Community regions, as translucent convex volumes under the scene. Rebuilt when the layout
   * parks rather than on its pulse: a three-dimensional hull of a whole community is worth
   * computing once per settle, not once per frame.
   */
  const rebuildHulls = useCallback(() => {
    const objects = objectsRef.current;
    if (!objects) return;
    clearHulls(objects.hulls);
    if (!hullsEnabledRef.current || objects.layoutRunning) return;
    const graph = renderedRef.current;
    const current = visualStateRef.current;
    const analysis = analyticsRef.current;
    const summaries = communitySummaries(graph, analysis, GRAPH_SCENE_LIMITS.legendMaxCommunities);
    const points = new Map<number, Vector3[]>(summaries.map((summary) => [summary.community, []]));
    graph.forEachNode((id, attributes) => {
      if (current.focusSet && !current.focusSet.has(id)) return;
      const bucket = points.get(analysis.communityOf.get(id) ?? 0);
      if (bucket) bucket.push(new Vector3(attributes.x, attributes.y, attributes.z ?? 0));
    });
    summaries.forEach((summary) => {
      const bucket = points.get(summary.community);
      if (!bucket || bucket.length < 4) return;
      try {
        const geometry = new ConvexGeometry(bucket);
        const fill = new Mesh(geometry, new MeshBasicMaterial({
          color: summary.color,
          transparent: true,
          opacity: 0.08,
          depthWrite: false,
          side: DoubleSide,
        }));
        const rim = new LineSegments(new EdgesGeometry(geometry), new LineBasicMaterial({
          color: summary.color,
          transparent: true,
          opacity: 0.28,
        }));
        objects.hulls.add(fill, rim);
      } catch {
        // A perfectly flat community has no volume to wrap; it simply goes unregioned.
      }
    });
  }, []);

  // The layout runs off the main thread and lands its frames straight into the buffers.
  useEffect(() => {
    const runner = createLayoutRunner({
      analysis: renderedAnalysis,
      display: rendered,
      tier,
      dimensions: 3,
      onChange: (running) => {
        const objects = objectsRef.current;
        if (objects) {
          objects.layoutRunning = running;
          writeMatrices();
          writeEdgeGeometry();
          if (!running) rebuildHulls();
        }
        if (!running && autoFitRef.current && !narrowedRef.current) fitAllRef.current();
        requestRenderRef.current();
      },
    });
    runnerRef.current = runner;
    runner.start();
    return () => {
      runner.kill();
      runnerRef.current = undefined;
    };
  }, [rendered, renderedAnalysis, tier, writeMatrices, writeEdgeGeometry, rebuildHulls]);

  /**
   * Assigns the pooled label plates. The selected node, the path ends and the live traversal
   * always carry their names — the Labels toggle governs only ordinary nodes, exactly as the 2D
   * reducer keeps the selected name alive with labels off. `auto` names the largest marks; `all`
   * names every node the force-label budget admits.
   */
  const assignLabels = useCallback(() => {
    const objects = objectsRef.current;
    const overlay = overlayRef.current;
    if (!objects || !overlay) return;
    const graph = renderedRef.current;
    const current = visualStateRef.current;
    const chosen = new Map<number, boolean>();
    const claim = (id: string | undefined, ruled: boolean) => {
      if (id === undefined) return;
      const at = objects.index.get(id);
      if (at === undefined) return;
      if (current.focusSet && !current.focusSet.has(id)) return;
      if (!chosen.has(at) || ruled) chosen.set(at, ruled || (chosen.get(at) ?? false));
    };
    claim(current.selectedNodeId, true);
    claim(current.pathFrom, true);
    claim(current.pathTo, true);
    claim(current.expandingNodeId, true);
    const hovered = hoverRef.current;
    if (hovered >= 0) {
      const id = objects.ids[hovered];
      if (id !== undefined) claim(id, false);
    }
    if (current.labelsAllowed) {
      if (current.highlight) {
        let budget = 60;
        for (const id of current.highlight) {
          if (budget <= 0) break;
          if (current.focusSet && !current.focusSet.has(id)) continue;
          claim(id, false);
          budget -= 1;
        }
      } else {
        const visible: { at: number; size: number }[] = [];
        graph.forEachNode((id, attributes) => {
          if (current.focusSet && !current.focusSet.has(id)) return;
          const at = objects.index.get(id);
          if (at !== undefined) visible.push({ at, size: attributes.size });
        });
        const budget = current.forceAllLabels ? visible.length : 24;
        visible.sort((left, right) => right.size - left.size);
        for (let rank = 0; rank < Math.min(budget, visible.length); rank += 1) {
          const entry = visible[rank];
          if (entry && !chosen.has(entry.at)) chosen.set(entry.at, false);
        }
      }
    }

    const pool = labelPoolRef.current;
    while (pool.length < chosen.size) {
      const element = document.createElement('div');
      element.className = 'graph-label';
      // Born hidden: it has no position until the next render projects it.
      element.style.visibility = 'hidden';
      overlay.appendChild(element);
      pool.push(element);
    }
    const slots: LabelSlot[] = [];
    let used = 0;
    chosen.forEach((ruled, at) => {
      const element = pool[used];
      const id = objects.ids[at];
      if (!element || id === undefined) return;
      used += 1;
      element.textContent = graph.getNodeAttribute(id, 'label');
      element.classList.toggle('ruled', ruled);
      slots.push({ at, element, offset: markRadius(graph.getNodeAttribute(id, 'size'), objects.clustered) });
    });
    for (let rest = used; rest < pool.length; rest += 1) {
      const element = pool[rest];
      if (element) element.style.visibility = 'hidden';
    }
    labelSlotsRef.current = slots;
  }, [markRadius]);

  /**
   * Builds the meshes for one rendered graph: one instanced mark per node, one position/colour
   * buffer pair for the relationships, an instanced arrowhead per drawn relationship, the rubric
   * overlays, and the community hull group. Alive until the graph or the curve setting changes.
   */
  const lastFittedRef = useRef<SceneGraph | undefined>(undefined);
  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    const clustered = rendered.someNode((_id, attributes) => attributes.kind === 'cluster');
    const ids = rendered.nodes();
    const index = new Map(ids.map((id, at) => [id, at]));
    const records: EdgeRecord[] = [];
    rendered.forEachEdge((key, attributes, source, target) => {
      const sourceAt = index.get(source);
      const targetAt = index.get(target);
      // A relationship from a node to itself has no extent to draw a line along; the mark itself
      // already stands where both of its ends are.
      if (sourceAt === undefined || targetAt === undefined || sourceAt === targetAt) return;
      records.push({ key, source: sourceAt, target: targetAt, aggregate: attributes.aggregateCount });
    });
    const segments = view.curvedEdges ? CURVE_SEGMENTS : 1;

    const nodeGeometry = clustered ? new OctahedronGeometry(1, 0) : new BoxGeometry(1, 1, 1);
    const nodeMesh = new InstancedMesh(nodeGeometry, new MeshLambertMaterial(), Math.max(1, ids.length));
    nodeMesh.instanceMatrix.setUsage(DynamicDrawUsage);
    nodeMesh.count = ids.length;
    nodeMesh.frustumCulled = false;

    const edgeGeometry = new BufferGeometry();
    const edgePositions = new BufferAttribute(new Float32Array(records.length * segments * 6), 3);
    edgePositions.setUsage(DynamicDrawUsage);
    const edgeColors = new BufferAttribute(new Float32Array(records.length * segments * 6), 3);
    edgeGeometry.setAttribute('position', edgePositions);
    edgeGeometry.setAttribute('color', edgeColors);
    const edgeLines = new LineSegments(edgeGeometry, new LineBasicMaterial({ vertexColors: true }));
    edgeLines.frustumCulled = false;

    /**
     * Arrowheads are instanced cones, one per drawn relationship, oriented on the layout's pulse.
     * Past the picking budget the orientation pass would cost more per frame than the hairlines
     * it decorates, so a very large result states direction only once it is narrowed.
     */
    const arrowsEnabled = records.length <= GRAPH_SCENE_LIMITS.edgeEventsMaxEdges;
    const arrows = new InstancedMesh(
      new ConeGeometry(ARROW_RADIUS, ARROW_LENGTH, 6),
      new MeshBasicMaterial(),
      Math.max(1, arrowsEnabled ? records.length : 1),
    );
    arrows.instanceMatrix.setUsage(DynamicDrawUsage);
    arrows.count = 0;
    arrows.frustumCulled = false;

    // The rubric overlay redraws the selected relationship and a found path over the scene, the
    // way the 2D reducer restates them; depth is refused so the statement is never occluded.
    const overlayGeometry = new BufferGeometry();
    const overlayPositions = new BufferAttribute(new Float32Array(Math.max(1, segments) * 6 * 8), 3);
    overlayGeometry.setAttribute('position', overlayPositions);
    const overlayLines = new LineSegments(overlayGeometry, new LineBasicMaterial({ depthTest: false }));
    overlayLines.renderOrder = 10;
    overlayLines.frustumCulled = false;
    overlayGeometry.setDrawRange(0, 0);

    // Selection rules the mark in the rubric: the same wireframe the mark itself is, stood off it.
    const outlineGeometry = new EdgesGeometry(clustered ? new OctahedronGeometry(1, 0) : new BoxGeometry(1, 1, 1));
    const makeOutline = (): LineSegments => {
      const outline = new LineSegments(outlineGeometry, new LineBasicMaterial({ depthTest: false }));
      outline.renderOrder = 11;
      outline.visible = false;
      outline.frustumCulled = false;
      return outline;
    };
    const outlines: [LineSegments, LineSegments, LineSegments] = [makeOutline(), makeOutline(), makeOutline()];

    const hulls = new Group();
    hulls.renderOrder = -1;

    runtime.scene.add(nodeMesh, edgeLines, arrows, overlayLines, hulls, ...outlines);
    const objects: SceneObjects = {
      ids,
      index,
      records,
      order: [],
      overlayOrder: [],
      segments,
      clustered,
      arrowsEnabled,
      positions: new Float32Array(ids.length * 3),
      nodeMesh,
      edgeLines,
      edgePositions,
      edgeColors,
      arrows,
      overlayLines,
      overlayPositions,
      outlines,
      hulls,
      layoutRunning: false,
    };
    objectsRef.current = objects;
    // The slots and the hover index point into the ids of the graph they were assigned against.
    labelSlotsRef.current = [];
    hoverRef.current = -1;
    // Painted here as well as in the visual pass below, because a curve toggle rebuilds these
    // buffers without changing anything that pass watches — a rebuild must never show empty.
    rebuildOrder();
    writeMatrices();
    writeEdgeGeometry();
    writeColors();
    assignLabels();
    rebuildHulls();
    if (lastFittedRef.current !== rendered) {
      lastFittedRef.current = rendered;
      if (!grownRef.current) fitAll(0);
    }
    requestRenderRef.current();

    return () => {
      runtime.scene.remove(nodeMesh, edgeLines, arrows, overlayLines, hulls, ...outlines);
      clearHulls(hulls);
      disposeObject(nodeMesh);
      disposeObject(edgeLines);
      disposeObject(arrows);
      disposeObject(overlayLines);
      outlineGeometry.dispose();
      outlines.forEach((outline) => (outline.material as LineBasicMaterial).dispose());
      objectsRef.current = undefined;
    };
  }, [rendered, view.curvedEdges, fitAll, rebuildOrder, writeMatrices, writeEdgeGeometry, writeColors, assignLabels, rebuildHulls]);

  // The visual pass: everything a state change can restyle, reapplied over the standing buffers.
  useEffect(() => {
    if (!objectsRef.current) return;
    rebuildOrder();
    writeMatrices();
    writeEdgeGeometry();
    writeColors();
    assignLabels();
    requestRenderRef.current();
  }, [rendered, analytics, focusSet, highlight, pathEdges, selectedNodeId, selectedEdgeId, expandingNodeId,
    pathEnds.from, pathEnds.to, view.labelMode, view.showEdges, view.colorBy, view.sizeBy, palette,
    rebuildOrder, writeMatrices, writeEdgeGeometry, writeColors, assignLabels]);

  // Regions are rebuilt when their inputs change; the layout rebuilds them itself when it parks.
  useEffect(() => {
    rebuildHulls();
    requestRenderRef.current();
  }, [hullsEnabled, focusSet, analytics, rendered, rebuildHulls]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    runtime.scene.background = new Color(palette.background);
    requestRenderRef.current();
  }, [palette]);

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

  // Keeps the margin's swatches on the colour the canvas is actually using — same rule as the 2D.
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

  /**
   * Picking, by hand. Three's raycaster tests every triangle of every instance, which is the
   * whole scene per click at fifty thousand cubes; a mark is honestly described by its centre
   * and radius, and a hairline by its segments, so the arithmetic is done directly instead.
   */
  const pick = useCallback((event: PointerEvent | MouseEvent): { kind: 'node' | 'edge'; at: number } | undefined => {
    const runtime = runtimeRef.current;
    const objects = objectsRef.current;
    if (!runtime || !objects) return undefined;
    const rect = runtime.renderer.domElement.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return undefined;
    const pointer = new Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    runtime.raycaster.setFromCamera(pointer, runtime.camera);
    const ray = runtime.raycaster.ray;
    const graph = renderedRef.current;
    const current = visualStateRef.current;
    const worldPerPixel = 2 * Math.tan(runtime.camera.fov * Math.PI / 360) / rect.height;
    const point = new Vector3();

    let nodeAt = -1;
    let nodeAlong = Infinity;
    for (let at = 0; at < objects.ids.length; at += 1) {
      if (hiddenAt(at)) continue;
      point.fromArray(objects.positions, at * 3);
      const along = point.clone().sub(ray.origin).dot(ray.direction);
      if (along <= 0 || along >= nodeAlong) continue;
      const id = objects.ids[at];
      const size = id === undefined ? 3 : graph.getNodeAttribute(id, 'size');
      const reach = Math.max(markRadius(size, objects.clustered), along * worldPerPixel * 7);
      if (ray.distanceSqToPoint(point) <= reach * reach) {
        nodeAt = at;
        nodeAlong = along;
      }
    }

    let edgeAt = -1;
    let edgeAlong = Infinity;
    if (current.showEdges && edgeEventsEnabled) {
      const a = new Vector3();
      const b = new Vector3();
      const onRay = new Vector3();
      const onSegment = new Vector3();
      const array = objects.edgePositions.array as Float32Array;
      objects.order.forEach((recordAt, drawn) => {
        for (let s = 0; s < objects.segments; s += 1) {
          const base = (drawn * objects.segments + s) * 6;
          a.set(array[base] ?? 0, array[base + 1] ?? 0, array[base + 2] ?? 0);
          b.set(array[base + 3] ?? 0, array[base + 4] ?? 0, array[base + 5] ?? 0);
          const distanceSq = ray.distanceSqToSegment(a, b, onRay, onSegment);
          const along = onRay.distanceTo(ray.origin);
          const reach = Math.max(1, along * worldPerPixel * 6);
          if (distanceSq <= reach * reach && along < edgeAlong) {
            edgeAt = recordAt;
            edgeAlong = along;
          }
        }
      });
    }

    // The mark wins over the hairline unless the hairline is distinctly nearer — the 2D grants
    // the same precedence through its threshold.
    if (nodeAt >= 0 && (edgeAt < 0 || nodeAlong <= edgeAlong + nodeAlong * worldPerPixel * 20)) {
      return { kind: 'node', at: nodeAt };
    }
    if (edgeAt >= 0) return { kind: 'edge', at: edgeAt };
    return undefined;
  }, [hiddenAt, markRadius, edgeEventsEnabled]);

  // Pointer wiring: click selects, empty space clears, double-click traverses, hovering names.
  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    const element = runtime.renderer.domElement;
    let pointerStart: { x: number; y: number } | undefined;
    let hoverFrame = 0;
    const recordPointerStart = (event: PointerEvent) => {
      pointerStart = { x: event.clientX, y: event.clientY };
    };
    const dragged = (event: PointerEvent | MouseEvent) => (
      pointerStart !== undefined && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 5
    );
    const selectElement = (event: PointerEvent) => {
      if (dragged(event)) {
        pointerStart = undefined;
        return;
      }
      pointerStart = undefined;
      const objects = objectsRef.current;
      const hit = pick(event);
      if (!hit || !objects) {
        onSelectionChange(undefined);
        return;
      }
      if (hit.kind === 'node') {
        const id = objects.ids[hit.at];
        if (id === undefined) return;
        if (view.pathMode) {
          setPathEnds((current) => (current.from && !current.to && current.from !== id
            ? { from: current.from, to: id }
            : { from: id }));
        }
        selectNode(id);
        return;
      }
      const key = objects.records[hit.at]?.key;
      if (key !== undefined) onSelectionChange({ kind: 'edge', id: key });
    };
    const expandElement = (event: MouseEvent) => {
      const objects = objectsRef.current;
      const hit = pick(event);
      if (!hit || hit.kind !== 'node' || !objects) return;
      const id = objects.ids[hit.at];
      if (id === undefined || rendered.getNodeAttribute(id, 'kind') === 'cluster') return;
      onExpandNode?.(id);
    };
    const hover = (event: PointerEvent) => {
      if (hoverFrame !== 0) return;
      hoverFrame = requestAnimationFrame(() => {
        hoverFrame = 0;
        const hit = pick(event);
        const at = hit?.kind === 'node' ? hit.at : -1;
        element.style.cursor = hit ? 'pointer' : '';
        if (at === hoverRef.current) return;
        hoverRef.current = at;
        assignLabels();
        requestRenderRef.current();
      });
    };
    const leave = () => {
      element.style.cursor = '';
      if (hoverRef.current === -1) return;
      hoverRef.current = -1;
      assignLabels();
      requestRenderRef.current();
    };
    element.addEventListener('pointerdown', recordPointerStart);
    element.addEventListener('pointerup', selectElement);
    element.addEventListener('dblclick', expandElement);
    element.addEventListener('pointermove', hover);
    element.addEventListener('pointerleave', leave);
    return () => {
      if (hoverFrame !== 0) cancelAnimationFrame(hoverFrame);
      element.removeEventListener('pointerdown', recordPointerStart);
      element.removeEventListener('pointerup', selectElement);
      element.removeEventListener('dblclick', expandElement);
      element.removeEventListener('pointermove', hover);
      element.removeEventListener('pointerleave', leave);
    };
  }, [rendered, view.pathMode, pick, selectNode, onSelectionChange, onExpandNode, assignLabels]);

  // A highlight three screens away is invisible, so the camera goes to it — same rule as the 2D.
  useEffect(() => {
    if (focusSet) frame(focusSet, 1.35);
    else if (pathNodes) frame(pathNodes, 1.35);
    else if (searchMatches?.size) frame(searchMatches, 1.35);
  }, [focusSet, pathNodes, searchMatches, frame]);

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
      <div className="graph-canvas" ref={containerRef} role="img" aria-label="Interactive 3D query result graph" />
      <div className="graph-label-layer" ref={overlayRef} aria-hidden="true" />
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
