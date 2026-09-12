'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const { execFile, execFileSync, spawn } = require('node:child_process');
const { promisify } = require('node:util');
const execute = promisify(execFile);
const source = path.resolve(__dirname, '..');
const metadata = require('../package.json');
const { address, options, processIdentity } = require('../cli.cjs');
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const suite = fs.mkdtempSync(path.join(os.tmpdir(), 'irongraph-launcher-tests-'));
const fixture = path.join(suite, 'fixture');
execFileSync('cc', ['-Wall', '-Wextra', '-Werror', `-DFIXTURE_VERSION="${metadata.version}"`, path.join(__dirname, 'fixture.c'), '-o', fixture]);
test.after(() => fs.rmSync(suite, { recursive: true, force: true }));

async function availablePorts() {
  const servers = [];
  const ports = [];
  try {
    for (let i = 0; i < 5; i++) {
      const server = net.createServer();
      await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
      servers.push(server);
      ports.push(server.address().port);
    }
    return ports;
  } finally { await Promise.all(servers.map((server) => new Promise((resolve) => server.close(resolve)))); }
}

async function setup(t, extra = {}) {
  const root = fs.mkdtempSync(path.join(suite, 'case with spaces-'));
  const npm = path.join(root, 'npm cache', 'irongraph');
  const suffix = process.platform === 'darwin' ? 'darwin-arm64' : `linux-${process.arch === 'x64' ? 'x64' : 'arm64'}-gnu`;
  const name = `@irongraph/cli-${suffix}`;
  const native = path.join(npm, 'node_modules', name);
  fs.mkdirSync(path.join(native, 'bin'), { recursive: true });
  fs.copyFileSync(path.join(source, 'cli.cjs'), path.join(npm, 'cli.cjs'));
  fs.writeFileSync(path.join(npm, 'package.json'), JSON.stringify(metadata));
  fs.writeFileSync(path.join(native, 'package.json'), JSON.stringify({ name, version: metadata.version }));
  fs.copyFileSync(fixture, path.join(native, 'bin', 'irongraph'));
  fs.copyFileSync(fixture, path.join(native, 'bin', 'irongraph-mcp'));
  const ports = await availablePorts();
  const data = path.join(root, 'database with spaces');
  const env = { ...process.env, IRONGRAPH_CLI_HOME: path.join(root, 'runtime with spaces'), IRONGRAPH_DATA_DIR: data, IRONGRAPH_EXECUTION_BACKEND: 'cpu', ...extra };
  for (const [index, service] of ['HTTP', 'MCP', 'BOLT', 'STREAM', 'QUEUE'].entries()) env[`IRONGRAPH_${service}_ADDR`] = `127.0.0.1:${ports[index]}`;
  const cli = path.join(npm, 'cli.cjs');
  const run = async (...args) => {
    try { const result = await execute(process.execPath, [cli, ...args], { env, timeout: 15000 }); return { code: 0, ...result }; }
    catch (error) { return { code: error.code, stdout: error.stdout || '', stderr: error.stderr || '' }; }
  };
  const statePath = () => {
    const runtimes = path.join(env.IRONGRAPH_CLI_HOME, 'run');
    return path.join(runtimes, fs.readdirSync(runtimes)[0], 'instance.json');
  };
  const state = () => JSON.parse(fs.readFileSync(statePath(), 'utf8'));
  t.after(async () => {
    // Restore valid recorded identity when a test deliberately corrupts metadata.
    try {
      const value = state();
      const identity = processIdentity(value.pid);
      if (identity && identity.command.startsWith(value.binary)) {
        value.born = identity.born;
        fs.writeFileSync(statePath(), JSON.stringify(value));
      }
    } catch {}
    await run('stop');
    fs.rmSync(root, { recursive: true, force: true });
  });
  return { root, npm, native, cli, data, env, ports, run, state, statePath };
}

test('help, version, strict flags, and numeric loopback validation', async () => {
  assert.equal(options(['start', '--background', '--execution-backend', 'cpu']).background, true);
  assert.throws(() => options(['stop', '--background']), /Unknown option/);
  assert.throws(() => options(['start', '--background', '--background']), /Repeated/);
  assert.throws(() => options(['start', '--data-dir']), /requires/);
  assert.throws(() => options(['start', '--execution-backend', 'cuda']), /must be/);
  assert.deepEqual(address('[::1]:18484', 'http'), { host: '::1', port: 18484 });
  for (const value of ['0.0.0.0:18484', 'localhost:18484', '127.0.0.1:0', '127.0.0.1:65536', '127.999.0.1:12']) assert.throws(() => address(value, 'http'));
  const output = await execute(process.execPath, [path.join(source, 'cli.cjs'), '--version']);
  assert.equal(output.stdout.trim(), `irongraph ${metadata.version}`);
  assert.match((await execute(process.execPath, [path.join(source, 'cli.cjs'), '--help'])).stdout, /IRONGRAPH_CLI_HOME/);
});

test('background lifecycle persists data and executable outside npm cache', async (t) => {
  const c = await setup(t);
  assert.match((await c.run('status')).stdout, /stopped/);
  const started = await c.run('start', '--background');
  assert.equal(started.code, 0, started.stderr);
  assert.match(started.stdout, /background/);
  const state = c.state();
  assert.ok(state.binary.startsWith(c.env.IRONGRAPH_CLI_HOME));
  assert.ok(state.mcpBinary.startsWith(path.join(c.env.IRONGRAPH_CLI_HOME, 'bin')));
  assert.equal(execFileSync(state.mcpBinary, ['--version'], { encoding: 'utf8' }).trim(), `irongraph-mcp ${metadata.version}`);
  assert.equal(fs.statSync(c.statePath()).mode & 0o777, 0o600);
  assert.match((await c.run('status')).stdout, /ready/);
  const duplicate = await c.run('start', '--background');
  assert.notEqual(duplicate.code, 0);
  assert.match(duplicate.stderr, /already running/);
  // A subsequent launcher install/version can manage the stable native process.
  fs.rmSync(path.join(c.native, 'bin'), { recursive: true });
  assert.match((await c.run('status')).stdout, /ready/);
  assert.match((await c.run('logs')).stdout, /fixture ready/);
  const stopped = await c.run('stop');
  assert.equal(stopped.code, 0, stopped.stderr);
  assert.match(stopped.stdout, /Data preserved/);
  assert.equal(processIdentity(state.pid), null);
  assert.equal(fs.existsSync(state.binary), false);
  assert.equal(fs.existsSync(state.mcpBinary), true, 'installed MCP host executable remains valid after stop');
  assert.match((await c.run('logs')).stdout, new RegExp(`http://127\\.0\\.0\\.1:${c.ports[0]}`));
  assert.match(fs.readFileSync(path.join(c.data, 'fixture-persistence.txt'), 'utf8'), /clean shutdown/);
  fs.mkdirSync(path.join(c.native, 'bin'));
  for (const file of ['irongraph', 'irongraph-mcp']) fs.copyFileSync(fixture, path.join(c.native, 'bin', file));
  assert.equal((await c.run('start', '--background')).code, 0);
  assert.equal(c.state().mcpBinary, state.mcpBinary);
  assert.equal((await c.run('stop')).code, 0);
  assert.equal(fs.readFileSync(path.join(c.data, 'fixture-persistence.txt'), 'utf8').match(/clean shutdown/g).length, 2);
});

test('foreground forwards SIGTERM and records a clean shutdown', async (t) => {
  const c = await setup(t);
  const child = spawn(process.execPath, [c.cli, 'start'], { env: c.env, stdio: ['ignore', 'pipe', 'pipe'] });
  let output = '';
  child.stdout.on('data', (data) => { output += data; });
  child.stderr.on('data', (data) => { output += data; });
  const closed = new Promise((resolve) => child.once('exit', (code, signal) => resolve({ code, signal })));
  for (let i = 0; i < 100 && !output.includes('fixture ready'); i++) await delay(25);
  assert.match(output, /fixture ready/);
  child.kill('SIGTERM');
  const result = await closed;
  assert.equal(result.code, 0, output);
  assert.match(fs.readFileSync(path.join(c.data, 'fixture-persistence.txt'), 'utf8'), /clean shutdown/);
  assert.match((await c.run('status')).stdout, /stopped/);
});

test('starting is reported while the model preparation equivalent runs', async (t) => {
  const c = await setup(t, { FIXTURE_DELAY_MS: '2500' });
  assert.equal((await c.run('start', '--background')).code, 0);
  assert.match((await c.run('status')).stdout, /starting/);
  assert.equal((await c.run('stop')).code, 0);
});

test('native manifest and executable version mismatch cannot start', async (t) => {
  const c = await setup(t);
  const manifestFile = path.join(c.native, 'package.json');
  const manifest = JSON.parse(fs.readFileSync(manifestFile));
  fs.writeFileSync(manifestFile, JSON.stringify({ ...manifest, version: '99.0.0' }));
  assert.match((await c.run('start', '--background')).stderr, /Native package version mismatch/);
  fs.writeFileSync(manifestFile, JSON.stringify(manifest));
  execFileSync('cc', ['-DFIXTURE_VERSION="99.0.0"', path.join(__dirname, 'fixture.c'), '-o', path.join(c.native, 'bin', 'irongraph')]);
  assert.match((await c.run('start', '--background')).stderr, /Native executable version mismatch/);
});

test('occupied ancillary ports fail explicitly before the database launches', async (t) => {
  const c = await setup(t);
  const server = net.createServer();
  await new Promise((resolve) => server.listen(c.ports[3], '127.0.0.1', resolve));
  try {
    const result = await c.run('start', '--background');
    assert.notEqual(result.code, 0);
    assert.match(result.stderr, /IRONGRAPH_STREAM_ADDR=.*unavailable/);
    assert.match((await c.run('status')).stdout, /stopped/);
  } finally { await new Promise((resolve) => server.close(resolve)); }
});

test('startup failure leaves useful logs and permits a later restart', async (t) => {
  const c = await setup(t, { FIXTURE_FAIL: '1' });
  const failed = await c.run('start', '--background');
  // A detached process may not receive a CPU timeslice before start returns.
  // Start acknowledges launch; status establishes readiness or termination.
  if (failed.code !== 0) assert.match(failed.stderr, /fixture startup failure/);
  else assert.match(failed.stdout, /starting/);
  let stopped;
  for (let attempt = 0; attempt < 100; attempt++) {
    stopped = await c.run('status');
    if (/stopped/.test(stopped.stdout)) break;
    await delay(50);
  }
  assert.match(stopped.stdout, /stopped/);
  assert.match((await c.run('logs')).stdout, /fixture startup failure/);
  delete c.env.FIXTURE_FAIL;
  assert.equal((await c.run('start', '--background')).code, 0);
  assert.match((await c.run('status')).stdout, /ready/);
});

test('a stale PID is recoverable, but reused or altered live identities are never signaled', async (t) => {
  const c = await setup(t);
  assert.equal((await c.run('start', '--background')).code, 0);
  const original = c.state();
  const state = { ...original, born: 'different birth time' };
  fs.writeFileSync(c.statePath(), JSON.stringify(state));
  assert.match((await c.run('stop')).stderr, /Refusing to signal/);
  assert.ok(processIdentity(original.pid));
  assert.match((await c.run('start', '--background')).stderr, /live PID does not match/);
  state.pid = process.pid;
  state.born = processIdentity(process.pid).born;
  fs.writeFileSync(c.statePath(), JSON.stringify(state));
  assert.match((await c.run('stop')).stderr, /Refusing to signal/);
  fs.writeFileSync(c.statePath(), JSON.stringify(original));
  assert.equal((await c.run('stop')).code, 0);
  assert.equal((await c.run('start', '--background')).code, 0);
  assert.notEqual(c.state().pid, original.pid);
});

test('concurrent starts admit exactly one database for a directory', async (t) => {
  const c = await setup(t);
  const results = await Promise.all([c.run('start', '--background'), c.run('start', '--background')]);
  assert.equal(results.filter((result) => result.code === 0).length, 1, JSON.stringify(results));
  assert.match(results.find((result) => result.code !== 0).stderr, /already running/);
  assert.match((await c.run('status')).stdout, /ready/);
});

test('a crashed command lock is reclaimed without admitting concurrent starts', async (t) => {
  const c = await setup(t);
  await c.run('status');
  const lock = path.join(path.dirname(c.statePath()), 'command.lock');
  fs.mkdirSync(lock);
  fs.writeFileSync(path.join(lock, 'owner.json'), JSON.stringify({ pid: process.pid, born: 'previous process birth' }));
  const results = await Promise.all([c.run('start', '--background'), c.run('start', '--background')]);
  assert.equal(results.filter((result) => result.code === 0).length, 1, JSON.stringify(results));
  assert.match((await c.run('status')).stdout, /ready/);
  assert.equal((await c.run('stop')).code, 0);
  fs.mkdirSync(lock);
  const old = new Date(Date.now() - 60000);
  fs.utimesSync(lock, old, old);
  assert.equal((await c.run('start', '--background')).code, 0);
});

test('abrupt native exit is detected and the same data can restart', async (t) => {
  const c = await setup(t);
  assert.equal((await c.run('start', '--background')).code, 0);
  const previous = c.state();
  process.kill(previous.pid, 'SIGKILL');
  for (let attempt = 0; attempt < 100 && processIdentity(previous.pid); attempt++) await delay(25);
  assert.match((await c.run('status')).stdout, /stopped/);
  assert.equal((await c.run('start', '--background')).code, 0);
  assert.equal(fs.existsSync(previous.binary), false);
  assert.equal(fs.existsSync(previous.mcpBinary), true);
});

test('explicit MCP configuration is preserved and relative paths are made stable', async (t) => {
  const custom = path.join(suite, 'explicit-mcp');
  fs.copyFileSync(fixture, custom);
  const c = await setup(t, { IRONGRAPH_MCP_BINARY: custom, IRONGRAPH_MCP_URL: 'http://127.0.0.1:24567' });
  assert.equal((await c.run('start', '--background')).code, 0);
  assert.equal(c.state().mcpBinary, custom);
  for (let attempt = 0; attempt < 100; attempt++) {
    if (/ready/.test((await c.run('status')).stdout)) break;
    await delay(25);
  }
  assert.match((await c.run('status')).stdout, /ready/);
  assert.match((await c.run('logs')).stdout, /http:\/\/127\.0\.0\.1:24567/);
  assert.equal((await c.run('stop')).code, 0);
  assert.equal(fs.existsSync(custom), true);
});
