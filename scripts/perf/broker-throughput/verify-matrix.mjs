import { readFileSync } from 'node:fs';

const report = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const matrix = report.matrix;
if (!matrix || !Array.isArray(report.results)) throw new Error('report has no matrix/results');
const expected = matrix.protocols.length * matrix.directions.length * matrix.payload_bytes.length
  * matrix.pipeline_depths.length * matrix.concurrencies.length * matrix.samples;
if (report.results.length !== expected) throw new Error(`expected ${expected} cells, found ${report.results.length}`);
for (const payload of [64, 1024, 65536, 1048576, 7340032]) {
  if (!matrix.payload_bytes.includes(payload)) throw new Error(`required payload ${payload} is absent`);
}
for (const protocol of ['kafka', 'amqp']) if (!matrix.protocols.includes(protocol)) throw new Error(`${protocol} is absent`);
for (const direction of ['ingress', 'egress', 'full_duplex']) if (!matrix.directions.includes(direction)) throw new Error(`${direction} is absent`);
for (const result of report.results) {
  const expectedBytes = result.messages * result.payload_bytes;
  if (result.verified_messages !== result.messages || result.verified_bytes !== expectedBytes) {
    throw new Error(`incorrect totals in ${result.protocol}/${result.direction}/${result.payload_bytes}`);
  }
}
console.log('broker throughput matrix verification passed');
