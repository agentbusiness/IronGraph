import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const readBytes = (relative) => fs.readFileSync(path.join(root, relative));
const read = (relative) => JSON.parse(readBytes(relative));
const baselinePath = 'performance-results/latest/results.json';
const baselineSha = crypto.createHash('sha256').update(readBytes(baselinePath)).digest('hex');
if (baselineSha !== 'd48539241130cd8d1d2c62547c538d00389778033f2a26c63ac184c1b9c8a431') {
  throw new Error(`immutable baseline changed: ${baselineSha}`);
}

const audit = read('performance-results/metal-final/publication-audit/results.json');
const retained = read('performance-results/metal-final/publication-tail-validation/results.json');
const descriptorRebase = read('performance-results/metal-final/publication-incremental-rebase/results.json');
const publicationDeltaPages = read('performance-results/metal-final/publication-delta-pages/results.json');
const measurement = (report, backend, nodes, operation = 'resident_batch_publish') => {
  const entry = report.measurements.find((candidate) =>
    candidate.backend === backend && candidate.nodes === nodes && candidate.operation === operation);
  if (!entry || entry.status !== 'ok' || !Number.isFinite(entry.p50) || !Number.isFinite(entry.p95)) {
    throw new Error(`missing valid ${backend}/${nodes}/${operation}`);
  }
  return entry;
};

const sizes = [100_000, 1_000_000, 2_000_000];
for (const nodes of sizes) {
  const before = measurement(audit, 'metal', nodes);
  const after = measurement(retained, 'metal', nodes);
  if (before.p50 / after.p50 < 1.5) {
    throw new Error(`Metal publication speedup is not material at ${nodes}`);
  }
  if (after.elements_per_sample !== 256 || after.p50 / after.elements_per_sample >= 1) {
    throw new Error(`Metal publication is not sub-microsecond per batched row at ${nodes}`);
  }
  const cpu = measurement(retained, 'cpu', nodes);
  if (cpu.p50 > 450) throw new Error(`CPU reference publication regressed materially at ${nodes}`);
  if (measurement(retained, 'metal', nodes, 'resident_edge_insert_publish').p50 > 80) {
    throw new Error(`Metal atomic edge publication regressed materially at ${nodes}`);
  }
}
const retainedMetal = sizes.map((nodes) => measurement(retained, 'metal', nodes).p50);
if (Math.max(...retainedMetal) / Math.min(...retainedMetal) > 1.25) {
  throw new Error(`Metal publication still scales with unrelated graph size: ${retainedMetal}`);
}
if (measurement(audit, 'metal', 2_000_000).p50 / measurement(retained, 'metal', 2_000_000).p50 < 10) {
  throw new Error('2M Metal publication did not clear the required order-of-magnitude gain');
}

for (const [label, report] of [
  ['descriptor rebase', descriptorRebase],
  ['host delta pages', publicationDeltaPages],
]) {
  const largeRatio = measurement(audit, 'metal', 2_000_000).p50
    / measurement(report, 'metal', 2_000_000).p50;
  if (largeRatio >= 1) {
    throw new Error(`rejected ${label} did not regress the target 2M publication: ${largeRatio}`);
  }
}

const adjacency = fs.readFileSync(path.join(root, 'crates/graph/src/adjacency.rs'), 'utf8');
const columns = fs.readFileSync(path.join(root, 'crates/graph/src/columns.rs'), 'utf8');
const accelerator = fs.readFileSync(path.join(root, 'crates/gpu/src/accelerator.rs'), 'utf8');
for (const [source, text] of [
  ['adjacency tail validator', 'rebase_shared_offsets_extension'],
  ['packed-list tail validator', 'rebase_shared_extension'],
]) {
  if (!adjacency.includes(text) && !columns.includes(text)) throw new Error(`missing ${source}`);
}
if (!accelerator.includes('.rebase_shared_offsets_extension(')
    || !accelerator.includes('.rebase_shared_extension(')) {
  throw new Error('Metal node append does not use the guarded tail validators');
}

console.log('final publication and CPU evidence verified');
