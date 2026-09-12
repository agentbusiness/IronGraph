import fs from 'node:fs';
import path from 'node:path';

const supplied = process.argv[2];
if (!supplied) throw new Error('usage: node scripts/perf/verify-results.mjs <results file or directory>');
const resolved = path.resolve(supplied);
const file = fs.statSync(resolved).isDirectory() ? path.join(resolved, 'results.json') : resolved;
const report = JSON.parse(fs.readFileSync(file, 'utf8'));

if (report.schema !== 'irongraph.performance-matrix.v1') throw new Error('unexpected result schema');
const requiredSizes = [100, 10_000, 100_000, 1_000_000, 2_000_000];
if (JSON.stringify(report.config?.sizes) !== JSON.stringify(requiredSizes)) {
  throw new Error(`required sizes missing or reordered: ${JSON.stringify(report.config?.sizes)}`);
}
if (!Number.isInteger(report.config?.fanout) || report.config.fanout < 1) throw new Error('invalid fanout');
if (!Number.isInteger(report.config?.samples) || report.config.samples < 1) throw new Error('invalid sample count');
if (!Array.isArray(report.measurements) || report.measurements.length === 0) throw new Error('no measurements');
if (typeof report.cpu_model !== 'string' || report.cpu_model.length === 0) throw new Error('missing CPU model');
if (!Number.isFinite(report.physical_memory_bytes) || report.physical_memory_bytes <= 0) throw new Error('missing physical memory');
if (!Array.isArray(report.metal_devices) || report.metal_devices.length === 0) throw new Error('missing Metal device identity');

const errors = report.measurements.filter((entry) => entry.status !== 'ok');
if (errors.length > 0) {
  throw new Error(`matrix has ${errors.length} failed measurements; first=${JSON.stringify(errors[0])}`);
}

const requiredCategories = ['read', 'write', 'traversal', 'aggregate', 'algorithm'];
const requiredOperations = [
  'point_lookup', 'range_count', 'one_hop', 'two_hop', 'variable_1_3', 'count_nodes',
  'count_edges', 'sum', 'avg', 'min_max', 'group_count', 'distinct', 'top_k', 'degree',
  'bfs', 'dfs', 'shortest_path', 'dijkstra', 'wcc', 'scc', 'pagerank', 'triangle_count',
  'clustering', 'k_core', 'louvain', 'resident_batch_publish', 'resident_edge_insert_publish',
];

for (const nodes of requiredSizes) {
  const expectedEdges = nodes * report.config.fanout;
  for (const backend of ['cpu', 'metal']) {
    const entries = report.measurements.filter((entry) => entry.nodes === nodes && entry.backend === backend);
    if (entries.length === 0) throw new Error(`missing ${backend} entries at ${nodes} nodes`);
    for (const category of requiredCategories) {
      if (!entries.some((entry) => entry.category === category)) {
        throw new Error(`missing ${backend}/${nodes}/${category}`);
      }
    }
    for (const operation of requiredOperations) {
      const entry = entries.find((candidate) => candidate.operation === operation);
      if (!entry) throw new Error(`missing ${backend}/${nodes}/${operation}`);
      if (entry.edges !== expectedEdges) throw new Error(`wrong edge count for ${backend}/${nodes}/${operation}`);
      if (!(entry.samples >= 1) || !Number.isFinite(entry.p50) || !Number.isFinite(entry.p95) || !Number.isFinite(entry.p99)) {
        throw new Error(`invalid distribution for ${backend}/${nodes}/${operation}`);
      }
      if (entry.samples !== report.config.samples || entry.warmups !== report.config.warmups) {
        throw new Error(`incomplete configured distribution for ${backend}/${nodes}/${operation}`);
      }
    }
  }
  for (const operation of ['wal_background_append_single', 'wal_background_append_batch']) {
    const entry = report.measurements.find((candidate) => candidate.nodes === nodes && candidate.backend === 'storage' && candidate.operation === operation);
    if (!entry || entry.durability !== 'eventual' || !Number.isFinite(entry.p50)) {
      throw new Error(`missing eventual WAL append measurement ${nodes}/${operation}`);
    }
  }
  for (const operation of ['eventual_write_ack_single', 'eventual_write_ack_concurrent']) {
    const entry = report.measurements.find((candidate) => candidate.nodes === nodes && candidate.backend === 'standalone' && candidate.operation === operation);
    if (!entry || entry.durability !== 'eventual' || entry.samples !== report.config.samples || !Number.isFinite(entry.p50)) {
      throw new Error(`missing standalone eventual acknowledgement measurement ${nodes}/${operation}`);
    }
  }
  for (const operation of ['wal_fsync_single', 'wal_fsync_batch']) {
    const entry = report.measurements.find((candidate) => candidate.nodes === nodes && candidate.backend === 'storage' && candidate.operation === operation);
    if (!entry || entry.durability !== 'background-fsync' || !Number.isFinite(entry.p50)) {
      throw new Error(`missing background durability-barrier measurement ${nodes}/${operation}`);
    }
  }
  for (const operation of ['initial_node_ingest', 'initial_edge_ingest']) {
    const entry = report.measurements.find((candidate) => candidate.nodes === nodes && candidate.backend === 'cpu-canonical' && candidate.operation === operation);
    if (!entry || !Number.isFinite(entry.resident_bytes) || entry.resident_bytes <= 0) {
      throw new Error(`missing canonical resident memory ${nodes}/${operation}`);
    }
  }
}

console.log('required CPU and Metal performance matrix verified');
