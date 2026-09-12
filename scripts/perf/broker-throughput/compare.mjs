import { readFileSync } from 'node:fs';

const baseline = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const current = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const key = (row) => [row.protocol, row.direction, row.payload_bytes, row.pipeline_depth, row.concurrency].join('/');
const median = (values) => {
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
};
const groups = (report) => Map.groupBy(report.results, key);
const before = groups(baseline);
const after = groups(current);
if (before.size !== after.size) throw new Error('baseline/current matrix sizes differ');
let maximumGain = -Infinity;
for (const [cell, beforeRows] of before) {
  const afterRows = after.get(cell);
  if (!afterRows || afterRows.length !== beforeRows.length) throw new Error(`matrix differs at ${cell}`);
  const beforeRate = median(beforeRows.map((row) => row.messages_per_second));
  const afterRate = median(afterRows.map((row) => row.messages_per_second));
  const gain = afterRate / beforeRate - 1;
  maximumGain = Math.max(maximumGain, gain);
  if (gain < -0.05) throw new Error(`${cell} throughput regressed ${(gain * 100).toFixed(2)}%`);
  const beforeRss = median(beforeRows.map((row) => row.peak_rss_bytes));
  const afterRss = median(afterRows.map((row) => row.peak_rss_bytes));
  if (afterRss > beforeRss * 1.05) throw new Error(`${cell} peak RSS regressed more than 5%`);
}
if (maximumGain < 0.15) throw new Error(`largest throughput gain was only ${(maximumGain * 100).toFixed(2)}%`);
console.log('broker throughput comparison passed');
