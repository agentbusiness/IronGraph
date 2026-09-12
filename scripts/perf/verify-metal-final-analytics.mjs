import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const read = (relative) => JSON.parse(fs.readFileSync(path.join(root, relative), 'utf8'));
const baseline = read('performance-results/latest/results.json');
const measurement = (report, operation, nodes, backend = 'metal') => {
  const entry = report.measurements.find((candidate) =>
    candidate.operation === operation && candidate.nodes === nodes && candidate.backend === backend);
  if (!entry || entry.status !== 'ok' || !Number.isFinite(entry.p50) || !Number.isFinite(entry.p95)) {
    throw new Error(`missing valid ${backend}/${nodes}/${operation}`);
  }
  return entry;
};
const speedup = (before, after) => before.p50 / after.p50;

const degree = read('performance-results/metal-final/degree-fast-large/results.json');
for (const nodes of [1_000_000, 2_000_000]) {
  const ratio = speedup(measurement(baseline, 'degree', nodes), measurement(degree, 'degree', nodes));
  if (ratio < 1.05) throw new Error(`degree speedup is not material at ${nodes}: ${ratio}`);
}

const shortest = read('performance-results/metal-final/shortest-bounded-narrow-reconstruct/results.json');
for (const nodes of [100_000, 1_000_000, 2_000_000]) {
  const before = measurement(baseline, 'shortest_path', nodes);
  const after = measurement(shortest, 'shortest_path', nodes);
  if (speedup(before, after) < 1.05 || after.p95 >= before.p95) {
    throw new Error(`shortest-path improvement is not stable at ${nodes}`);
  }
}

const dfs = read('performance-results/metal-final/dfs-adaptive-width/results.json');
for (const nodes of [100_000, 1_000_000, 2_000_000]) {
  const before = measurement(baseline, 'dfs', nodes);
  const after = measurement(dfs, 'dfs', nodes);
  if (speedup(before, after) < 1.15 || after.p95 >= before.p95) {
    throw new Error(`DFS improvement is not stable at ${nodes}`);
  }
}

for (const [relative, operation] of [
  ['pagerank-phase-readbacks/results.json', 'pagerank'],
  ['pagerank-contribution-bank/results.json', 'pagerank'],
  ['pagerank-power2-divide/results.json', 'pagerank'],
]) {
  const report = read(`performance-results/metal-final/${relative}`);
  const bestLarge = Math.max(...[1_000_000, 2_000_000].map((nodes) =>
    speedup(measurement(baseline, operation, nodes), measurement(report, operation, nodes))));
  if (bestLarge >= 1.05) throw new Error(`rejected ${relative} unexpectedly clears materiality: ${bestLarge}`);
}

const wideBfs = read('performance-results/metal-final/persistent-full-threadgroup/results.json');
for (const operation of ['bfs', 'shortest_path', 'dijkstra', 'wcc', 'scc']) {
  const ratio = speedup(measurement(baseline, operation, 2_000_000), measurement(wideBfs, operation, 2_000_000));
  if (ratio >= 1) throw new Error(`rejected full-threadgroup ${operation} did not regress: ${ratio}`);
}

const exactWidth = read('performance-results/metal-final/sparse-row-exact-width/results.json');
for (const operation of ['dfs', 'shortest_path']) {
  const retained = operation === 'dfs' ? dfs : shortest;
  const ratios = [100_000, 1_000_000, 2_000_000].map((nodes) =>
    speedup(measurement(retained, operation, nodes), measurement(exactWidth, operation, nodes)));
  if (operation === 'shortest_path' && ratios.some((ratio) => ratio >= 1)) {
    throw new Error(`rejected four-lane shortest path did not consistently regress: ${ratios}`);
  }
}

console.log('final Metal analytics evidence verified');
