import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const baselinePath = path.join(root, 'performance-results/latest/results.json');
const expectedHash = 'd48539241130cd8d1d2c62547c538d00389778033f2a26c63ac184c1b9c8a431';
const bytes = fs.readFileSync(baselinePath);
const actualHash = crypto.createHash('sha256').update(bytes).digest('hex');
if (actualHash !== expectedHash) throw new Error(`immutable baseline changed: ${actualHash}`);

const baseline = JSON.parse(bytes);
for (const size of [100, 10_000, 100_000, 1_000_000, 2_000_000]) {
  for (const backend of ['cpu', 'metal']) {
    if (!baseline.measurements.some((entry) => entry.nodes === size && entry.backend === backend)) {
      throw new Error(`baseline lacks ${backend}/${size}`);
    }
  }
}

for (const relative of [
  'degree-fast-large/results.json',
  'pagerank-phase-readbacks/results.json',
  'pagerank-contribution-bank/results.json',
  'pagerank-power2-divide/results.json',
  'persistent-full-threadgroup/results.json',
  'shortest-bounded-bfs/results.json',
  'shortest-bounded-narrow-reconstruct/results.json',
  'dfs-adaptive-width/results.json',
  'sparse-row-exact-width/results.json',
]) {
  const file = path.join(root, 'performance-results/metal-final', relative);
  if (!fs.existsSync(file)) throw new Error(`missing diagnostic artifact ${relative}`);
  const report = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (report.schema !== 'irongraph.performance-matrix.v1') {
    throw new Error(`unexpected diagnostic schema in ${relative}`);
  }
  if (!Array.isArray(report.config?.operations) || report.config.operations.length === 0) {
    throw new Error(`diagnostic did not record its operation filter: ${relative}`);
  }
  if (report.measurements.some((entry) => entry.status !== 'ok')) {
    throw new Error(`diagnostic contains a failed measurement: ${relative}`);
  }
}

console.log('final Metal baseline and diagnostic coverage verified');
