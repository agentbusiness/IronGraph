import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = process.cwd();
const cargo = fs.readFileSync(path.join(root, 'Cargo.toml'), 'utf8');
const source = fs.readFileSync(path.join(root, 'benches/performance_matrix.rs'), 'utf8');

const requireText = (text, label) => {
  if (!source.includes(text)) throw new Error(`performance harness is missing ${label}: ${text}`);
};

if (!/\[\[bench\]\]\s*name = "performance_matrix"\s*harness = false/m.test(cargo)) {
  throw new Error('Cargo.toml does not register the custom performance_matrix harness');
}

for (const size of ['100', '10_000', '100_000', '1_000_000', '2_000_000']) {
  requireText(size, `required graph size ${size}`);
}
for (const backend of ['"cpu"', '"metal"']) requireText(backend, `${backend} backend`);
for (const category of ['"read"', '"write"', '"traversal"', '"aggregate"', '"algorithm"']) {
  requireText(category, `${category} workload category`);
}
for (const operation of [
  'initial_node_ingest',
  'initial_edge_ingest',
  'eventual_write_ack_single',
  'eventual_write_ack_concurrent',
  'wal_background_append_single',
  'wal_background_append_batch',
  'wal_fsync_single',
  'wal_fsync_batch',
  'resident_batch_publish',
  'resident_edge_insert_publish',
  'point_lookup',
  'range_count',
  'one_hop',
  'two_hop',
  'variable_1_3',
  'group_count',
  'top_k',
  'bfs',
  'dfs',
  'shortest_path',
  'dijkstra',
  'wcc',
  'scc',
  'pagerank',
  'triangle_count',
  'clustering',
  'k_core',
  'louvain',
]) requireText(operation, `operation ${operation}`);

for (const field of [
  'git_revision',
  'git_dirty',
  'cpu_model',
  'physical_memory_bytes',
  'metal_devices',
  'resident_bytes',
  'durability',
  'throughput_per_second',
  'p50',
  'p95',
  'p99',
  'error',
]) requireText(field, `result field ${field}`);

if (!source.includes('IGPERF_FANOUT') || !source.includes('IGPERF_OUTPUT') || !source.includes('IGPERF_DURABLE_BATCH_ROWS') || !source.includes('IGPERF_OPERATIONS')) {
  throw new Error('performance harness lacks reproducible fanout/output controls');
}
if (!source.includes('status: "error"') || !source.includes('admit_project') || !source.includes('construct_backend')) {
  throw new Error('performance harness does not preserve explicit failure reporting');
}

const smokeOutput = path.join(root, 'performance-results/smoke/results.json');
const smoke = spawnSync('cargo', ['bench', '--profile', 'dev', '--bench', 'performance_matrix'], {
  cwd: root,
  encoding: 'utf8',
  env: {
    ...process.env,
    IGPERF_SIZES: '100',
    IGPERF_BACKENDS: 'cpu,metal',
    IGPERF_SAMPLES: '1',
    IGPERF_WARMUPS: '1',
    IGPERF_FANOUT: '4',
    IGPERF_DIRTY_BODY_BYTES: '128',
    IGPERF_BATCH_ROWS: '8',
    IGPERF_DURABLE_BATCH_ROWS: '8',
    IGPERF_OUTPUT: smokeOutput,
  },
});
if (smoke.status !== 0) {
  throw new Error(`performance smoke run failed\n${smoke.stdout}\n${smoke.stderr}`);
}
const report = JSON.parse(fs.readFileSync(smokeOutput, 'utf8'));
if (report.schema !== 'irongraph.performance-matrix.v1') throw new Error('smoke result schema mismatch');
const failures = report.measurements.filter((entry) => entry.status !== 'ok');
if (failures.length > 0) throw new Error(`smoke matrix contains failures: ${JSON.stringify(failures)}`);
if (!report.cpu_model || !report.physical_memory_bytes || report.metal_devices.length === 0) {
  throw new Error('smoke matrix lacks hardware identity');
}
for (const backend of ['cpu', 'metal']) {
  for (const category of ['read', 'write', 'traversal', 'aggregate', 'algorithm']) {
    if (!report.measurements.some((entry) => entry.backend === backend && entry.category === category)) {
      throw new Error(`smoke matrix lacks ${backend}/${category}`);
    }
  }
}

console.log('performance harness verification passed');
