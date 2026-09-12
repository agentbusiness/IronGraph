import { execFileSync, spawnSync } from 'node:child_process';
import { mkdirSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '../../..');
const output = path.resolve(root, process.env.IRONGRAPH_BROKER_OUTPUT ?? 'performance-results/broker-throughput/latest.json');
const list = (name, fallback) => (process.env[name] ?? fallback)
  .split(',').map((value) => value.trim()).filter(Boolean);
const integers = (name, fallback) => list(name, fallback).map((value) => {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error(`${name} contains an invalid integer`);
  return parsed;
});

if (process.env.IRONGRAPH_BROKER_SKIP_BUILD !== '1') {
  execFileSync('cargo', ['bench', '--bench', 'broker_throughput', '--no-run'], {
    cwd: root,
    stdio: 'inherit',
  });
}
const dependencyDir = path.join(root, 'target/release/deps');
const executable = readdirSync(dependencyDir)
  .filter((name) => /^broker_throughput-[0-9a-f]+$/.test(name))
  .map((name) => path.join(dependencyDir, name))
  .filter((candidate) => (statSync(candidate).mode & 0o111) !== 0)
  .sort((left, right) => statSync(right).mtimeMs - statSync(left).mtimeMs)[0];
if (!executable) throw new Error('broker throughput executable was not found after the build');

const matrix = {
  protocols: list('IRONGRAPH_BROKER_PROTOCOLS', 'kafka,amqp'),
  directions: list('IRONGRAPH_BROKER_DIRECTIONS', 'ingress,egress,full_duplex'),
  payload_bytes: integers('IRONGRAPH_BROKER_PAYLOADS', '64,1024,65536,1048576,7340032'),
  pipeline_depths: integers('IRONGRAPH_BROKER_DEPTHS', '1,64,1024'),
  concurrencies: integers('IRONGRAPH_BROKER_CONCURRENCIES', '1'),
  samples: Number.parseInt(process.env.IRONGRAPH_BROKER_SAMPLES ?? '3', 10),
  target_bytes: Number.parseInt(process.env.IRONGRAPH_BROKER_TARGET_BYTES ?? `${32 * 1024 * 1024}`, 10),
};
if (!Number.isSafeInteger(matrix.samples) || matrix.samples <= 0) throw new Error('IRONGRAPH_BROKER_SAMPLES must be positive');

const results = [];
for (const protocol of matrix.protocols) {
  for (const direction of matrix.directions) {
    for (const payloadBytes of matrix.payload_bytes) {
      for (const pipelineDepth of matrix.pipeline_depths) {
        for (const concurrency of matrix.concurrencies) {
          for (let sample = 1; sample <= matrix.samples; sample += 1) {
            const completed = spawnSync(executable, [], {
              cwd: root,
              encoding: 'utf8',
              env: {
                ...process.env,
                IRONGRAPH_BROKER_PROTOCOL: protocol,
                IRONGRAPH_BROKER_DIRECTION: direction,
                IRONGRAPH_BROKER_PAYLOAD_BYTES: String(payloadBytes),
                IRONGRAPH_BROKER_PIPELINE_DEPTH: String(pipelineDepth),
                IRONGRAPH_BROKER_CONCURRENCY: String(concurrency),
                IRONGRAPH_BROKER_SAMPLE: String(sample),
                IRONGRAPH_BROKER_TARGET_BYTES: String(matrix.target_bytes),
              },
              maxBuffer: 16 * 1024 * 1024,
            });
            const combined = `${completed.stdout ?? ''}\n${completed.stderr ?? ''}`;
            if (completed.status !== 0) {
              throw new Error(`broker cell failed (${protocol}/${direction}/${payloadBytes}/${pipelineDepth}/${concurrency}/${sample}):\n${combined}`);
            }
            const marker = combined.split(/\r?\n/).find((line) => line.startsWith('IRONGRAPH_BROKER_RESULT='));
            if (!marker) throw new Error(`broker cell emitted no result: ${combined}`);
            const result = JSON.parse(marker.slice('IRONGRAPH_BROKER_RESULT='.length));
            if (result.concurrency !== concurrency) {
              throw new Error(`broker cell did not honor concurrency=${concurrency}`);
            }
            results.push(result);
            process.stdout.write(`${protocol} ${direction} ${payloadBytes}B depth=${pipelineDepth} clients=${concurrency} sample=${sample}: ${result.messages_per_second.toFixed(2)} msg/s ${result.mib_per_second.toFixed(2)} MiB/s\n`);
          }
        }
      }
    }
  }
}

const command = (program, args) => {
  try { return execFileSync(program, args, { cwd: root, encoding: 'utf8' }).trim(); }
  catch { return ''; }
};
const report = {
  schema_version: 1,
  generated_at: new Date().toISOString(),
  machine: {
    platform: os.platform(),
    release: os.release(),
    architecture: os.arch(),
    cpu: os.cpus()[0]?.model ?? '',
    logical_cpus: os.cpus().length,
    total_memory_bytes: os.totalmem(),
  },
  build: {
    git_revision: command('git', ['rev-parse', 'HEAD']),
    rustc: command('rustc', ['--version']),
    profile: 'bench',
  },
  matrix,
  results,
};
mkdirSync(path.dirname(output), { recursive: true });
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(`broker throughput report written: ${output}`);
