import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, readdirSync, statSync } from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '../../..');
const target = path.resolve(root, process.env.CARGO_TARGET_DIR ?? 'target');
const temporary = mkdtempSync(path.join(os.tmpdir(), 'irongraph-broker-external-'));
const dataDirectory = path.join(temporary, 'data');

const reservePort = () => new Promise((resolve, reject) => {
  const server = net.createServer();
  server.once('error', reject);
  server.listen(0, '127.0.0.1', () => {
    const { port } = server.address();
    server.close((error) => error ? reject(error) : resolve(port));
  });
});

const [httpPort, boltPort, kafkaPort, amqpPort] = await Promise.all([
  reservePort(), reservePort(), reservePort(), reservePort(),
]);

if (process.env.IRONGRAPH_BROKER_SKIP_BUILD !== '1') {
  execFileSync('cargo', ['build', '--release', '--bin', 'irongraph'], { cwd: root, stdio: 'inherit' });
  execFileSync('cargo', ['bench', '--bench', 'broker_throughput', '--no-run'], { cwd: root, stdio: 'inherit' });
}

const client = readdirSync(path.join(target, 'release/deps'))
  .filter((name) => /^broker_throughput-[0-9a-f]+$/.test(name))
  .map((name) => path.join(target, 'release/deps', name))
  .filter((candidate) => (statSync(candidate).mode & 0o111) !== 0)
  .sort((left, right) => statSync(right).mtimeMs - statSync(left).mtimeMs)[0];
if (!client) throw new Error('external broker client executable is missing');

const baseEnvironment = {
  ...process.env,
  IRONGRAPH_DATA_DIR: dataDirectory,
  IRONGRAPH_HTTP_ADDR: `127.0.0.1:${httpPort}`,
  IRONGRAPH_BOLT_ADDR: `127.0.0.1:${boltPort}`,
  IRONGRAPH_STREAM_ADDR: `127.0.0.1:${kafkaPort}`,
  IRONGRAPH_QUEUE_ADDR: `127.0.0.1:${amqpPort}`,
  IRONGRAPH_EXECUTION_BACKEND: 'cpu',
  IRONGRAPH_DISABLE_MAINTENANCE: '1',
  RUST_LOG: 'warn',
};

let server;
const startServer = (project) => {
  const descriptor = spawn(path.join(target, 'release/irongraph'), [], {
    cwd: root,
    env: project ? { ...baseEnvironment, IRONGRAPH_BROKER_PROJECT: project } : baseEnvironment,
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  descriptor.stderr.on('data', (chunk) => process.stderr.write(chunk));
  server = descriptor;
};

const stopServer = async () => {
  if (!server || server.exitCode !== null) return;
  const exited = new Promise((resolve) => server.once('exit', resolve));
  server.kill('SIGINT');
  await Promise.race([exited, new Promise((resolve) => setTimeout(resolve, 15_000))]);
  if (server.exitCode === null) server.kill('SIGKILL');
  await exited;
};

const query = async (statement) => {
  const response = await fetch(`http://127.0.0.1:${httpPort}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson' },
    body: JSON.stringify({
      request_id: crypto.randomUUID(),
      project_id: null,
      query: statement,
      parameters: {},
      consistency: 'PUBLISHED',
    }),
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`query failed (${response.status}): ${text}`);
  return text.trim().split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
};

const waitForHttp = async () => {
  let lastError;
  for (let attempt = 0; attempt < 300; attempt += 1) {
    if (server.exitCode !== null) throw new Error('server exited during startup');
    try {
      await query('SHOW PROJECTS');
      return;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw lastError ?? new Error('server HTTP listener did not become ready');
};

try {
  startServer(null);
  await waitForHttp();
  await query('CREATE PROJECT broker_performance');
  const projects = await query('SHOW PROJECTS');
  const batch = projects.find((event) => event.type === 'batch');
  const project = batch?.columns?.[0]?.values?.[0]?.value;
  if (typeof project !== 'string' || project.length === 0) {
    throw new Error(`created project identity was not returned: ${JSON.stringify(projects)}`);
  }
  await stopServer();
  startServer(project);
  await waitForHttp();

  const requestedProtocols = new Set((process.env.IRONGRAPH_BROKER_PROTOCOLS ?? 'kafka,amqp').split(','));
  const cells = [
    ['kafka', `127.0.0.1:${kafkaPort}`],
    ['amqp', `amqp://guest:guest@127.0.0.1:${amqpPort}/%2f`],
  ].filter(([protocol]) => requestedProtocols.has(protocol));
  for (const [protocol, endpoint] of cells) {
    const completed = spawnSync(client, [], {
      cwd: root,
      encoding: 'utf8',
      env: {
        ...process.env,
        IRONGRAPH_BROKER_PROTOCOL: protocol,
        IRONGRAPH_BROKER_DIRECTION: process.env.IRONGRAPH_BROKER_DIRECTION ?? 'ingress',
        IRONGRAPH_BROKER_PAYLOAD_BYTES: '64',
        IRONGRAPH_BROKER_MESSAGES: process.env.IRONGRAPH_BROKER_MESSAGES ?? '1000',
        IRONGRAPH_BROKER_PIPELINE_DEPTH: process.env.IRONGRAPH_BROKER_PIPELINE_DEPTH ?? '64',
        IRONGRAPH_BROKER_SAMPLE: '1',
        ...(protocol === 'kafka'
          ? { IRONGRAPH_BROKER_KAFKA_ENDPOINT: endpoint }
          : { IRONGRAPH_BROKER_AMQP_ENDPOINT: endpoint }),
      },
      maxBuffer: 16 * 1024 * 1024,
    });
    const output = `${completed.stdout ?? ''}\n${completed.stderr ?? ''}`;
    if (completed.status !== 0) throw new Error(`${protocol} external cell failed:\n${output}`);
    const marker = output.split(/\r?\n/).find((line) => line.startsWith('IRONGRAPH_BROKER_RESULT='));
    if (!marker) throw new Error(`${protocol} external cell emitted no result:\n${output}`);
    const result = JSON.parse(marker.slice('IRONGRAPH_BROKER_RESULT='.length));
    if (result.environment !== 'external_server') throw new Error('client did not mark external provenance');
    console.log(`${protocol}: ${result.messages_per_second.toFixed(2)} msg/s p99=${result.latency_p99_us}us verified=${result.verified_messages}`);
  }
  console.log(`EXTERNAL_BROKER_SMOKE_PASSED server_pid=${server.pid} client_process=separate data_dir=${dataDirectory}`);
} finally {
  await stopServer();
}
