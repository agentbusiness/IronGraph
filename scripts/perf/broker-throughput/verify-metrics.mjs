import { readFileSync } from 'node:fs';

const report = JSON.parse(readFileSync(process.argv[2], 'utf8'));
if (!report.machine?.cpu || !report.build?.rustc || !report.build?.profile) throw new Error('machine/build metadata is incomplete');
const fields = ['messages_per_second', 'mib_per_second', 'wall_seconds', 'cpu_user_seconds',
  'cpu_system_seconds', 'peak_rss_bytes', 'latency_p50_us', 'latency_p95_us', 'latency_p99_us'];
for (const [index, result] of report.results.entries()) {
  for (const field of fields) {
    if (!Number.isFinite(result[field]) || result[field] < 0) throw new Error(`cell ${index} has invalid ${field}`);
  }
  if (result.messages_per_second <= 0 || result.wall_seconds <= 0 || result.peak_rss_bytes <= 0) {
    throw new Error(`cell ${index} has empty performance evidence`);
  }
  if (result.latency_p50_us > result.latency_p95_us || result.latency_p95_us > result.latency_p99_us) {
    throw new Error(`cell ${index} latency percentiles are not monotonic`);
  }
}
console.log('broker throughput metrics verification passed');
