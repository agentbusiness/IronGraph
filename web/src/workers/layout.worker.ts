/// <reference lib="webworker" />

interface LayoutRequest {
  type: 'layout';
  id: number;
  dimensions: 2 | 3;
  nodeCount: number;
  /** Flat source/target index pairs. */
  edges: Uint32Array;
  /** Starting coordinates, nodeCount × dimensions. Absent: a deterministic sunflower seed. */
  seeds?: Float32Array;
  /** Rendered radius per node; springs leave room for what the endpoints draw. */
  radii?: Float32Array;
  /** Collapsed multiplicity per edge pair; a reinforced relationship pulls slightly shorter. */
  weights?: Float32Array;
}

interface CancelRequest { type: 'cancel'; id: number }

let cancelledId = -1;

/** Barnes-Hut opening angle: a cell this much smaller than its distance is treated as one mass. */
const THETA = 0.9;
/** Co-located nodes would otherwise subdivide forever; past this depth they share a cell. */
const MAX_DEPTH = 22;

/**
 * The spring-electric ("force field") model, scaled around one number: the rest length of a
 * relationship. Everything else is expressed against it so the shape survives retuning.
 */
const REST_LENGTH = 60;
/**
 * Repulsion charge. The force between two bodies is CHARGE·mass/d — the 1/d falloff (not 1/d²) is
 * what fans a hub's leaves into an open halo: it stays strong at ring distance, where an
 * inverse-square kick has already died off and lets the ring collapse into a filled blob.
 */
const CHARGE = 900;
/**
 * Beyond this, bodies stop repelling. Without a cutoff every disconnected fragment is pushed until
 * gravity finally balances a whole graph's worth of charge, which parks singletons several screens
 * of empty space away; with one, separation is local and components pack instead of scattering.
 * Anything unconnected parks at about this distance from the mass that pushed it, so it is chosen
 * as "one community-width away", not further. Swept together with GRAVITY on the two-star result
 * and the six-community benchmark: 650 leaves a dead band wider than the stars themselves, 450
 * squeezes community separation below 2× their spread; 550 with gravity 0.055 measured best on
 * every axis at once (scene extent, halo spacing, separation ratio, singleton band).
 */
const CUTOFF = 550;
/** Spring-to-origin pull that bounds the scene; grows with distance, so nothing escapes it. */
const GRAVITY = 0.055;
/** Below this separation the interaction is treated as at this distance, so forces stay finite. */
const MIN_DISTANCE = 4;
/** Velocity carried between ticks; the rest is friction. */
const VELOCITY_RETAIN = 0.6;
/** The cooling floor: the simulation stops when alpha decays to this. */
const ALPHA_MIN = 0.003;

function seededCoordinate(index: number, axis: number): number {
  let value = Math.imul(index + 1, 0x9e3779b1) ^ Math.imul(axis + 11, 0x85ebca6b);
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb352d);
  value ^= value >>> 15;
  return ((value >>> 0) / 0xffffffff - 0.5) * 160;
}

/**
 * A quadtree (2D) or octree (3D) in flat typed arrays. Repulsion over a 50,000-node result is the
 * whole cost of this layout: every node against every other is 2.5 billion pairs a tick, while
 * approximating distant cells by their centre of mass brings it down to O(n log n).
 */
class MassTree {
  private readonly branches: number;
  private capacity: number;
  private children: Int32Array;
  private body: Int32Array;
  private mass: Float32Array;
  private centreOfMass: Float32Array;
  private centre: Float32Array;
  private half: Float32Array;
  private used = 1;

  constructor(private readonly dimensions: number, expectedNodes: number) {
    this.branches = 1 << dimensions;
    this.capacity = Math.max(64, expectedNodes * 2);
    this.children = new Int32Array(this.capacity * this.branches).fill(-1);
    this.body = new Int32Array(this.capacity).fill(-1);
    this.mass = new Float32Array(this.capacity);
    this.centreOfMass = new Float32Array(this.capacity * dimensions);
    this.centre = new Float32Array(this.capacity * dimensions);
    this.half = new Float32Array(this.capacity);
  }

  private grow(): void {
    const capacity = this.capacity * 2;
    const children = new Int32Array(capacity * this.branches).fill(-1);
    children.set(this.children);
    const body = new Int32Array(capacity).fill(-1);
    body.set(this.body);
    const mass = new Float32Array(capacity);
    mass.set(this.mass);
    const centreOfMass = new Float32Array(capacity * this.dimensions);
    centreOfMass.set(this.centreOfMass);
    const centre = new Float32Array(capacity * this.dimensions);
    centre.set(this.centre);
    const half = new Float32Array(capacity);
    half.set(this.half);
    this.capacity = capacity;
    this.children = children;
    this.body = body;
    this.mass = mass;
    this.centreOfMass = centreOfMass;
    this.centre = centre;
    this.half = half;
  }

  reset(positions: Float32Array, nodeCount: number): void {
    let extent = 1;
    for (let index = 0; index < nodeCount * this.dimensions; index += 1) {
      extent = Math.max(extent, Math.abs(positions[index] ?? 0));
    }
    this.children.fill(-1);
    this.body.fill(-1);
    this.mass.fill(0);
    this.centreOfMass.fill(0);
    this.centre.fill(0);
    this.used = 1;
    this.half[0] = extent * 1.05;
  }

  private childIndex(cell: number, positions: Float32Array, offset: number): number {
    let octant = 0;
    for (let axis = 0; axis < this.dimensions; axis += 1) {
      if ((positions[offset + axis] ?? 0) >= (this.centre[cell * this.dimensions + axis] ?? 0)) octant |= 1 << axis;
    }
    return octant;
  }

  private allocate(parent: number, octant: number): number {
    if (this.used >= this.capacity) this.grow();
    const cell = this.used;
    this.used += 1;
    const parentHalf = this.half[parent] ?? 1;
    this.half[cell] = parentHalf / 2;
    for (let axis = 0; axis < this.dimensions; axis += 1) {
      const sign = (octant & (1 << axis)) === 0 ? -1 : 1;
      this.centre[cell * this.dimensions + axis] = (this.centre[parent * this.dimensions + axis] ?? 0) + sign * parentHalf / 2;
    }
    this.children[parent * this.branches + octant] = cell;
    return cell;
  }

  insert(node: number, positions: Float32Array): void {
    const offset = node * this.dimensions;
    let cell = 0;
    for (let depth = 0; depth < MAX_DEPTH; depth += 1) {
      const occupant = this.body[cell] ?? -1;
      const cellMass = this.mass[cell] ?? 0;

      if (cellMass === 0) {
        this.body[cell] = node;
        this.mass[cell] = 1;
        for (let axis = 0; axis < this.dimensions; axis += 1) {
          this.centreOfMass[cell * this.dimensions + axis] = positions[offset + axis] ?? 0;
        }
        return;
      }

      // Accumulate on the way down: an internal cell's centre of mass is the running average of
      // everything below it, which is exactly what the far-field approximation reads later.
      this.mass[cell] = cellMass + 1;
      for (let axis = 0; axis < this.dimensions; axis += 1) {
        const slot = cell * this.dimensions + axis;
        this.centreOfMass[slot] = ((this.centreOfMass[slot] ?? 0) * cellMass + (positions[offset + axis] ?? 0)) / (cellMass + 1);
      }

      if (occupant >= 0) {
        this.body[cell] = -1;
        const octant = this.childIndex(cell, positions, occupant * this.dimensions);
        const existing = this.children[cell * this.branches + octant] ?? -1;
        const target = existing >= 0 ? existing : this.allocate(cell, octant);
        this.body[target] = occupant;
        this.mass[target] = 1;
        for (let axis = 0; axis < this.dimensions; axis += 1) {
          this.centreOfMass[target * this.dimensions + axis] = positions[occupant * this.dimensions + axis] ?? 0;
        }
      }

      const octant = this.childIndex(cell, positions, offset);
      const existing = this.children[cell * this.branches + octant] ?? -1;
      cell = existing >= 0 ? existing : this.allocate(cell, octant);
    }
    // Depth exhausted: the node shares a cell with its co-located neighbours.
    this.mass[cell] = (this.mass[cell] ?? 0) + 1;
  }

  /** Adds the repulsion every body within the cutoff exerts on `node` into `velocity`. */
  applyRepulsion(node: number, positions: Float32Array, velocity: Float32Array, alpha: number, stack: Int32Array): void {
    const offset = node * this.dimensions;
    let top = 0;
    stack[top] = 0;
    top += 1;

    while (top > 0) {
      top -= 1;
      const cell = stack[top] ?? 0;
      const cellMass = this.mass[cell] ?? 0;
      if (cellMass === 0) continue;
      const occupant = this.body[cell] ?? -1;
      if (occupant === node) continue;

      let distanceSquared = 0;
      for (let axis = 0; axis < this.dimensions; axis += 1) {
        const delta = (positions[offset + axis] ?? 0) - (this.centreOfMass[cell * this.dimensions + axis] ?? 0);
        distanceSquared += delta * delta;
      }
      const width = (this.half[cell] ?? 0) * 2;

      // A cell whose nearest possible body is past the cutoff cannot contribute, nor can anything
      // inside it — the centre of mass is at most a diagonal away from its farthest member.
      const reach = CUTOFF + width;
      if (distanceSquared > reach * reach) continue;

      if (occupant >= 0 || width * width < THETA * THETA * distanceSquared) {
        if (distanceSquared > CUTOFF * CUTOFF) continue;
        const floored = Math.max(distanceSquared, MIN_DISTANCE * MIN_DISTANCE);
        // |force| = CHARGE·mass/d: dividing by d² here and multiplying by the raw delta below
        // leaves exactly that magnitude along the separation direction.
        const kick = CHARGE * cellMass * alpha / floored;
        for (let axis = 0; axis < this.dimensions; axis += 1) {
          let delta = (positions[offset + axis] ?? 0) - (this.centreOfMass[cell * this.dimensions + axis] ?? 0);
          // Coincident bodies have no direction; a deterministic nudge picks one so they separate.
          if (distanceSquared === 0) delta = ((node + axis) % 2 === 0 ? 1 : -1) * 0.5;
          velocity[offset + axis] = (velocity[offset + axis] ?? 0) + delta * kick;
        }
        continue;
      }

      for (let branch = 0; branch < this.branches; branch += 1) {
        const child = this.children[cell * this.branches + branch] ?? -1;
        if (child >= 0) {
          stack[top] = child;
          top += 1;
        }
      }
    }
  }
}

function run(request: LayoutRequest): void {
  const { id, dimensions, nodeCount, edges, seeds, radii, weights } = request;
  if (nodeCount === 0) {
    const positions = new Float32Array();
    self.postMessage({ type: 'positions', id, positions, done: true }, { transfer: [positions.buffer] });
    return;
  }

  const positions = new Float32Array(nodeCount * dimensions);
  if (seeds && seeds.length === nodeCount * dimensions) {
    positions.set(seeds);
  } else {
    for (let node = 0; node < nodeCount; node += 1) {
      for (let axis = 0; axis < dimensions; axis += 1) positions[node * dimensions + axis] = seededCoordinate(node, axis);
    }
  }
  const velocity = new Float32Array(nodeCount * dimensions);

  // Degrees drive two things d3 established: a spring between two hubs is weakened (1/min degree)
  // so bridges do not fuse communities, and each spring moves the lighter endpoint — a leaf swings
  // around its hub, never the hub around its leaf.
  const degree = new Float32Array(nodeCount);
  for (let edge = 0; edge + 1 < edges.length; edge += 2) {
    const source = edges[edge];
    const target = edges[edge + 1];
    if (source === undefined || target === undefined || source >= nodeCount || target >= nodeCount) continue;
    degree[source] = (degree[source] ?? 0) + 1;
    degree[target] = (degree[target] ?? 0) + 1;
  }

  const tree = new MassTree(dimensions, nodeCount);
  const stack = new Int32Array(MAX_DEPTH * (1 << dimensions) + 64);
  let iteration = 0;
  const totalIterations = nodeCount > 20_000 ? 100 : nodeCount > 2_000 ? 200 : 300;
  // Large results pay for a tree build per iteration, so they emit fewer, larger steps: streaming
  // 50,000 positions every few iterations costs more in postMessage than it shows on screen.
  const iterationsPerChunk = nodeCount > 20_000 ? 2 : 3;
  // Exponential cooling that lands exactly on the floor at the last tick, the d3 schedule.
  const alphaDecay = 1 - Math.pow(ALPHA_MIN, 1 / totalIterations);
  let alpha = 1;

  const chunk = () => {
    if (cancelledId === id) return;
    const chunkEnd = Math.min(totalIterations, iteration + iterationsPerChunk);
    for (; iteration < chunkEnd; iteration += 1) {
      alpha += (ALPHA_MIN - alpha) * alphaDecay;

      tree.reset(positions, nodeCount);
      for (let node = 0; node < nodeCount; node += 1) tree.insert(node, positions);
      for (let node = 0; node < nodeCount; node += 1) tree.applyRepulsion(node, positions, velocity, alpha, stack);

      for (let edge = 0; edge + 1 < edges.length; edge += 2) {
        const source = edges[edge];
        const target = edges[edge + 1];
        if (source === undefined || target === undefined || source >= nodeCount || target >= nodeCount) continue;
        const sourceOffset = source * dimensions;
        const targetOffset = target * dimensions;
        let distanceSquared = 0;
        for (let axis = 0; axis < dimensions; axis += 1) {
          const delta = (positions[targetOffset + axis] ?? 0) - (positions[sourceOffset + axis] ?? 0);
          distanceSquared += delta * delta;
        }
        const distance = Math.max(MIN_DISTANCE, Math.sqrt(distanceSquared));
        const weight = weights?.[edge / 2] ?? 1;
        // A reinforced relationship sits a little shorter; endpoint radii keep big discs clear.
        const rest = REST_LENGTH / (1 + 0.12 * Math.log(weight)) + (radii?.[source] ?? 0) + (radii?.[target] ?? 0);
        const sourceDegree = Math.max(1, degree[source] ?? 1);
        const targetDegree = Math.max(1, degree[target] ?? 1);
        const strength = 1 / Math.min(sourceDegree, targetDegree);
        const pull = (distance - rest) / distance * strength * alpha;
        const bias = sourceDegree / (sourceDegree + targetDegree);
        for (let axis = 0; axis < dimensions; axis += 1) {
          const delta = (positions[targetOffset + axis] ?? 0) - (positions[sourceOffset + axis] ?? 0);
          velocity[targetOffset + axis] = (velocity[targetOffset + axis] ?? 0) - delta * pull * bias;
          velocity[sourceOffset + axis] = (velocity[sourceOffset + axis] ?? 0) + delta * pull * (1 - bias);
        }
      }

      for (let node = 0; node < nodeCount; node += 1) {
        const offset = node * dimensions;
        for (let axis = 0; axis < dimensions; axis += 1) {
          const pulled = (velocity[offset + axis] ?? 0) - (positions[offset + axis] ?? 0) * GRAVITY * alpha;
          positions[offset + axis] = (positions[offset + axis] ?? 0) + pulled;
          velocity[offset + axis] = pulled * VELOCITY_RETAIN;
        }
      }
    }

    const snapshot = positions.slice();
    self.postMessage(
      { type: 'positions', id, positions: snapshot, done: iteration >= totalIterations },
      { transfer: [snapshot.buffer] },
    );
    if (iteration < totalIterations) setTimeout(chunk, 0);
  };

  chunk();
}

self.addEventListener('message', (event: MessageEvent<LayoutRequest | CancelRequest>) => {
  if (event.data.type === 'cancel') cancelledId = event.data.id;
  else run(event.data);
});

export {};
