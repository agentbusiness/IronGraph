#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const net = require('node:net');
const http = require('node:http');
const { spawn, execFileSync } = require('node:child_process');
const { version } = require('./package.json');

const DEFAULT_ADDRESSES = {
  IRONGRAPH_HTTP_ADDR: '127.0.0.1:18484',
  IRONGRAPH_MCP_ADDR: '127.0.0.1:18488',
  IRONGRAPH_BOLT_ADDR: '127.0.0.1:18485',
  IRONGRAPH_STREAM_ADDR: '127.0.0.1:18486',
  IRONGRAPH_QUEUE_ADDR: '127.0.0.1:18487',
};
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function help() {
  console.log(`IronGraph ${version} — standalone database and web console

  npx irongraph start [--background] [--data-dir PATH]
                     [--http-addr LOOPBACK:PORT] [--execution-backend auto|cpu|metal]
  npx irongraph status [--data-dir PATH]
  npx irongraph logs [--data-dir PATH]
  npx irongraph stop [--data-dir PATH]
  npx irongraph --version

Data defaults to ~/.irongraph/data. IRONGRAPH_DATA_DIR and native IRONGRAPH_*
configuration are supported; explicit flags take precedence. First startup
downloads and warms the local embedding model. Foreground mode stops with Ctrl-C.
Background mode survives terminal closure; it does not start at system boot.
Logs shows the latest 200 lines. Status prints the console URL when ready.
IRONGRAPH_CLI_HOME overrides the launcher's runtime/log directory (~/.irongraph)
without changing your database directory or embedding model cache.`);
}

function options(argv) {
  const command = argv.shift() || '--help';
  if (['--help', '-h', 'help', '--version', '-v', 'version'].includes(command)) {
    if (argv.length) throw new Error(`Unexpected argument: ${argv[0]}`);
    return { command };
  }
  if (!['start', 'stop', 'status', 'logs'].includes(command)) throw new Error(`Unknown command: ${command}. Run npx irongraph --help.`);
  const result = { command, background: false };
  const seen = new Set();
  while (argv.length) {
    const flag = argv.shift();
    if (seen.has(flag)) throw new Error(`Repeated option: ${flag}`);
    seen.add(flag);
    if (flag === '--background' && command === 'start') result.background = true;
    else if (flag === '--data-dir' || (command === 'start' && ['--http-addr', '--execution-backend'].includes(flag))) {
      const value = argv.shift();
      if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value.`);
      result[flag.slice(2)] = value;
    } else throw new Error(`Unknown option for ${command}: ${flag}`);
  }
  const backend = result['execution-backend'] || process.env.IRONGRAPH_EXECUTION_BACKEND || 'auto';
  if (command === 'start' && !['auto', 'cpu', 'metal'].includes(backend)) throw new Error('Execution backend must be auto, cpu, or metal.');
  result.backend = backend;
  return result;
}

function privateDirectory(directory) {
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  const info = fs.lstatSync(directory);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`Runtime directory must be a real directory: ${directory}`);
  if (typeof process.getuid === 'function' && info.uid !== process.getuid()) throw new Error(`Runtime directory belongs to another user: ${directory}`);
  fs.chmodSync(directory, 0o700);
}

function context(opts) {
  let data = path.resolve(opts['data-dir'] || process.env.IRONGRAPH_DATA_DIR || path.join(os.homedir(), '.irongraph', 'data'));
  if (opts.command === 'start') fs.mkdirSync(data, { recursive: true, mode: 0o700 });
  // Resolve existing parents as well as existing data directories so status/stop
  // address the same instance through a symlinked parent.
  let parent = data;
  const suffix = [];
  while (!fs.existsSync(parent)) {
    suffix.unshift(path.basename(parent));
    const next = path.dirname(parent);
    if (next === parent) break;
    parent = next;
  }
  data = path.join(fs.realpathSync(parent), ...suffix);
  const base = path.resolve(process.env.IRONGRAPH_CLI_HOME || path.join(os.homedir(), '.irongraph'));
  privateDirectory(base);
  const runtime = path.join(base, 'run');
  privateDirectory(runtime);
  const directory = path.join(runtime, crypto.createHash('sha256').update(data).digest('hex'));
  privateDirectory(directory);
  return { base, data, directory, state: path.join(directory, 'instance.json'), log: path.join(directory, 'irongraph.log'), lock: path.join(directory, 'command.lock') };
}

function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); }
  catch (error) { if (error.code === 'ENOENT') return null; throw new Error(`Cannot read runtime state ${file}: ${error.message}`); }
}

function writeJson(file, value) {
  const temporary = `${file}.${crypto.randomUUID()}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600, flag: 'wx' });
  fs.renameSync(temporary, file);
}

function processIdentity(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 1) return null;
  try {
    const output = execFileSync('ps', ['-ww', '-p', String(pid), '-o', 'lstart=', '-o', 'stat=', '-o', 'command='], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], env: { ...process.env, LC_ALL: 'C' } }).trim();
    const match = output.match(/^(\w{3}\s+\w{3}\s+\d+\s+\d\d:\d\d:\d\d\s+\d{4})\s+(\S+)\s+([\s\S]+)$/);
    if (!match) throw new Error(`Cannot parse process identity for PID ${pid}; no process will be signaled.`);
    if (match[2].includes('Z')) return null;
    return { born: match[1].replace(/\s+/g, ' '), command: match[3] };
  } catch (error) {
    if (error.code === 'ENOENT') throw new Error('The ps system utility is required to safely manage the database process.');
    if (error.status === 1) return null;
    throw error;
  }
}

function inspected(state, ctx) {
  if (!state) return 'stopped';
  if (state.data !== ctx.data || !Number.isSafeInteger(state.pid) || state.pid <= 1 || typeof state.binary !== 'string' || typeof state.born !== 'string' || typeof state.run !== 'string' || !/^[0-9a-f-]{36}$/.test(state.run)) throw new Error(`Invalid instance state: ${ctx.state}`);
  const identity = processIdentity(state.pid);
  if (!identity) return 'stopped';
  const runs = path.join(ctx.directory, 'runs') + path.sep;
  // Both a unique executable pathname and the kernel-reported birth time must
  // match. A stale PID can never authorize a signal to an unrelated process.
  if (!state.binary.startsWith(runs) || state.binary !== path.join(runs, state.run, 'irongraph') || identity.born !== state.born || !identity.command.startsWith(state.binary)) return 'unverified';
  const remainder = identity.command.slice(state.binary.length);
  if (remainder && !/^\s/.test(remainder)) return 'unverified';
  return 'running';
}

async function withLock(ctx, work) {
  const deadline = Date.now() + 5000;
  while (true) {
    try {
      fs.mkdirSync(ctx.lock, { mode: 0o700 });
      const owner = processIdentity(process.pid);
      if (!owner) throw new Error('Cannot establish launcher process identity.');
      writeJson(path.join(ctx.lock, 'owner.json'), { pid: process.pid, born: owner.born });
      break;
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
      try {
        const owner = readJson(path.join(ctx.lock, 'owner.json'));
        const identity = owner && processIdentity(owner.pid);
        const lockInfo = fs.statSync(ctx.lock);
        const age = Date.now() - lockInfo.mtimeMs;
        if ((owner && (!identity || identity.born !== owner.born)) || (!owner && age > 30000)) {
          // Only one waiter may reclaim a stale directory. Recheck the owner
          // after claiming it: a competing waiter may have replaced the lock.
          const reaping = path.join(ctx.lock, 'reaping');
          try {
            fs.mkdirSync(reaping);
            const latest = readJson(path.join(ctx.lock, 'owner.json'));
            const alive = latest && processIdentity(latest.pid);
            const sameDirectory = fs.statSync(ctx.lock).ino === lockInfo.ino;
            if (sameDirectory && ((latest && (!alive || alive.born !== latest.born)) || (!latest && age > 30000))) {
              const stale = `${ctx.lock}.${crypto.randomUUID()}.stale`;
              fs.renameSync(ctx.lock, stale);
              fs.rmSync(stale, { recursive: true });
              continue;
            }
            fs.rmdirSync(reaping);
          } catch (race) { if (!['EEXIST', 'ENOENT'].includes(race.code)) throw race; }
        }
      } catch (race) { if (race.code === 'ENOENT') continue; throw race; }
      if (Date.now() >= deadline) throw new Error('Another IronGraph command is running for this data directory. Retry after it finishes.');
      await delay(50);
    }
  }
  try { return await work(); }
  finally { fs.rmSync(ctx.lock, { recursive: true, force: true }); }
}

function nativePackage() {
  const suffix = { 'darwin-arm64': 'darwin-arm64', 'linux-arm64': 'linux-arm64-gnu', 'linux-x64': 'linux-x64-gnu' }[`${process.platform}-${process.arch}`];
  if (!suffix) throw new Error(`No standalone package is available for ${process.platform}/${process.arch}. Supported: Apple Silicon macOS, Linux ARM64, Linux AMD64.`);
  if (process.platform === 'linux' && !process.report.getReport().header.glibcVersionRuntime) throw new Error('The Linux standalone package requires glibc; musl/Alpine is unsupported.');
  const name = `@irongraph/cli-${suffix}`;
  let metadata;
  try { metadata = require.resolve(`${name}/package.json`); }
  catch { throw new Error(`Missing native package ${name}@${version}. Reinstall irongraph with npm optional dependencies enabled (--include=optional).`); }
  const manifest = readJson(metadata);
  if (manifest.name !== name || manifest.version !== version) throw new Error(`Native package version mismatch: expected ${name}@${version}, found ${manifest.name}@${manifest.version}. Reinstall irongraph.`);
  const binary = path.join(path.dirname(metadata), 'bin', 'irongraph');
  const mcp = path.join(path.dirname(metadata), 'bin', 'irongraph-mcp');
  for (const file of [binary, mcp]) {
    if (!fs.statSync(file).isFile()) throw new Error(`Native executable is missing: ${file}`);
    fs.accessSync(file, fs.constants.X_OK);
  }
  for (const file of [binary, mcp]) {
    let actual;
    try { actual = execFileSync(file, ['--version'], { encoding: 'utf8', timeout: 10000, stdio: ['ignore', 'pipe', 'pipe'] }).trim(); }
    catch (error) { throw new Error(`Cannot execute ${path.basename(file)}: ${error.message}`); }
    const expected = `${path.basename(file)} ${version}`;
    if (actual !== expected) throw new Error(`Native executable version mismatch: expected ${expected}, received ${actual}.`);
  }
  return { binary, mcp };
}

function fileHash(file) {
  const hash = crypto.createHash('sha256');
  const buffer = Buffer.alloc(1024 * 1024);
  const fd = fs.openSync(file, 'r');
  try {
    let count;
    while ((count = fs.readSync(fd, buffer, 0, buffer.length, null)) > 0) hash.update(buffer.subarray(0, count));
  } finally { fs.closeSync(fd); }
  return hash.digest('hex');
}

function permanentMcp(native, ctx) {
  const digest = fileHash(native.mcp);
  const bins = path.join(ctx.base, 'bin');
  privateDirectory(bins);
  const directory = path.join(bins, `${version}-${digest}`);
  privateDirectory(directory);
  const destination = path.join(directory, 'irongraph-mcp');
  if (!fs.existsSync(destination) || fileHash(destination) !== digest) {
    const temporary = `${destination}.${crypto.randomUUID()}.tmp`;
    fs.copyFileSync(native.mcp, temporary, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(temporary, 0o700);
    fs.renameSync(temporary, destination);
  }
  return destination;
}

function address(value, name) {
  const match = value.match(/^(127(?:\.\d{1,3}){3}|\[::1\]):(\d+)$/);
  if (!match || Number(match[2]) < 1 || Number(match[2]) > 65535) throw new Error(`${name} must be a numeric loopback address with port 1–65535, such as 127.0.0.1:18484.`);
  const host = match[1].replace(/^\[|\]$/g, '');
  if (!net.isIP(host)) throw new Error(`Invalid address for ${name}: ${value}`);
  return { host, port: Number(match[2]) };
}

async function checkPorts(env) {
  const held = [];
  try {
    for (const key of Object.keys(DEFAULT_ADDRESSES)) {
      const target = address(env[key], key);
      await new Promise((resolve, reject) => {
        const server = net.createServer();
        server.once('error', (error) => reject(new Error(`${key}=${env[key]} is unavailable (${error.code}). Choose a free port using ${key}${key === 'IRONGRAPH_HTTP_ADDR' ? ' or --http-addr' : ''}; IronGraph has not started.`)));
        server.listen({ ...target, exclusive: true }, () => { held.push(server); resolve(); });
      });
    }
  } finally { await Promise.all(held.map((server) => new Promise((resolve) => server.close(resolve)))); }
}

function consoleReady(url) {
  return new Promise((resolve) => {
    const request = http.get(url, { timeout: 1000 }, (response) => { response.resume(); resolve(response.statusCode === 200); });
    request.on('timeout', () => request.destroy());
    request.on('error', () => resolve(false));
  });
}

function logTail(file) {
  let fd;
  try {
    fd = fs.openSync(file, 'r');
    const size = fs.fstatSync(fd).size;
    const count = Math.min(size, 128 * 1024);
    const buffer = Buffer.alloc(count);
    fs.readSync(fd, buffer, 0, count, size - count);
    const lines = buffer.toString('utf8').split('\n');
    if (size > count) lines.shift();
    return lines.slice(-201).join('\n');
  } catch (error) { if (error.code === 'ENOENT') return ''; throw error; }
  finally { if (fd !== undefined) fs.closeSync(fd); }
}

function cleanRun(state, ctx) {
  if (state && typeof state.run === 'string' && /^[0-9a-f-]{36}$/.test(state.run)) fs.rmSync(path.join(ctx.directory, 'runs', state.run), { recursive: true, force: true });
}

async function start(opts, ctx) {
  let child;
  let state;
  let exit;
  const forward = (signal) => {
    if (child && child.exitCode === null && child.signalCode === null && (!state || inspected(state, ctx) === 'running')) child.kill(signal);
  };
  const interrupt = () => forward('SIGINT');
  const terminate = () => forward('SIGTERM');
  await withLock(ctx, async () => {
    const previous = readJson(ctx.state);
    const previousStatus = inspected(previous, ctx);
    if (previousStatus === 'running') throw new Error(`IronGraph is already running for ${ctx.data} (PID ${previous.pid}). Console: ${previous.url}`);
    if (previousStatus === 'unverified') throw new Error(`A live PID does not match the recorded IronGraph process. No process was signaled. Inspect ${ctx.state} before starting another instance.`);
    const native = nativePackage();
    const env = { ...process.env, IRONGRAPH_DATA_DIR: ctx.data, IRONGRAPH_EXECUTION_BACKEND: opts.backend };
    for (const [key, value] of Object.entries(DEFAULT_ADDRESSES)) env[key] ||= value;
    if (opts['http-addr']) env.IRONGRAPH_HTTP_ADDR = opts['http-addr'];
    await checkPorts(env);
    env.IRONGRAPH_MCP_BINARY = env.IRONGRAPH_MCP_BINARY ? path.resolve(env.IRONGRAPH_MCP_BINARY) : permanentMcp(native, ctx);
    env.IRONGRAPH_MCP_URL ||= `http://${env.IRONGRAPH_HTTP_ADDR}`;
    cleanRun(previous, ctx);
    const run = crypto.randomUUID();
    const runDirectory = path.join(ctx.directory, 'runs', run);
    privateDirectory(runDirectory);
    const binary = path.join(runDirectory, 'irongraph');
    fs.copyFileSync(native.binary, binary, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(binary, 0o700);
    const log = fs.openSync(ctx.log, 'a', 0o600);
    fs.fchmodSync(log, 0o600);
    fs.writeSync(log, `\nIronGraph ${version}: starting ${new Date().toISOString()}\n`);
    try {
      child = spawn(binary, [], { env, cwd: ctx.data, detached: opts.background, stdio: opts.background ? ['ignore', log, log] : ['ignore', 'pipe', 'pipe'] });
      if (!opts.background) {
        // Install forwarding before the first await after spawn, including the
        // interval before the child identity has been persisted.
        process.on('SIGINT', interrupt);
        process.on('SIGTERM', terminate);
      }
      exit = new Promise((resolve) => {
        child.once('error', (error) => resolve({ code: 1, error }));
        child.once('exit', (code, signal) => resolve({ code: code ?? (signal ? 1 : 0), signal }));
      });
      if (!opts.background) {
        child.stdout.on('data', (chunk) => { fs.writeSync(log, chunk); process.stdout.write(chunk); });
        child.stderr.on('data', (chunk) => { fs.writeSync(log, chunk); process.stderr.write(chunk); });
        child.once('close', () => fs.closeSync(log));
      }
      await new Promise((resolve, reject) => { child.once('spawn', resolve); child.once('error', reject); });
      let identity;
      for (let attempt = 0; attempt < 20; attempt++) {
        identity = processIdentity(child.pid);
        if (identity) break;
        if (child.exitCode !== null || child.signalCode !== null) break;
        await delay(25);
      }
      if (!identity) throw new Error(`Database exited during startup. Logs: ${ctx.log}\n${logTail(ctx.log)}`);
      state = { pid: child.pid, born: identity.born, binary, mcpBinary: env.IRONGRAPH_MCP_BINARY, run, version, data: ctx.data, url: `http://${env.IRONGRAPH_HTTP_ADDR}/web/`, started: new Date().toISOString() };
      writeJson(ctx.state, state);
      if (inspected(state, ctx) !== 'running') throw new Error('Cannot verify the new native process identity.');
    } catch (error) {
      // This ChildProcess still belongs to this invocation; never use an old state PID here.
      if (child && child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
      process.removeListener('SIGINT', interrupt);
      process.removeListener('SIGTERM', terminate);
      if (!child || opts.background) fs.closeSync(log);
      throw error;
    }
    if (opts.background) { fs.closeSync(log); child.unref(); }
  });
  console.log(`IronGraph ${version} is starting (PID ${state.pid}).\nData: ${ctx.data}\nPreparing the database and local embedding model; the first download can take several minutes.\nConsole when ready: ${state.url}\nLogs: ${ctx.log}`);
  if (opts.background) {
    // Catch immediate configuration failures without waiting for model installation.
    await delay(150);
    if (inspected(state, ctx) !== 'running') throw new Error(`Database exited during startup. Run npx irongraph logs --data-dir ${JSON.stringify(ctx.data)}.\n${logTail(ctx.log)}`);
    console.log('Running in the background. Use npx irongraph status to check readiness.');
    return;
  }
  let checking = false;
  const timer = setInterval(async () => {
    if (checking) return;
    checking = true;
    if (await consoleReady(state.url)) { console.log(`IronGraph is ready. Open ${state.url}`); clearInterval(timer); }
    checking = false;
  }, 1000);
  const result = await exit;
  clearInterval(timer);
  process.removeListener('SIGINT', interrupt);
  process.removeListener('SIGTERM', terminate);
  await withLock(ctx, async () => {
    const current = readJson(ctx.state);
    if (current?.run === state.run) cleanRun(state, ctx);
  });
  if (result.error) throw result.error;
  process.exitCode = result.code;
}

async function status(ctx) {
  const state = readJson(ctx.state);
  const current = inspected(state, ctx);
  if (current === 'unverified') throw new Error(`Unverified PID ${state.pid}: it does not match the recorded IronGraph process. No process was signaled. State: ${ctx.state}`);
  if (current === 'stopped') {
    const tail = logTail(ctx.log).trim();
    console.log(`IronGraph is stopped.\nData: ${ctx.data}${state ? `\nLast version: ${state.version}` : ''}${tail ? `\nLogs: ${ctx.log}\nRecent log:\n${tail.split('\n').slice(-12).join('\n')}` : ''}`);
    return;
  }
  const ready = await consoleReady(state.url);
  // A process can exit while HTTP readiness is checked.
  if (inspected(state, ctx) !== 'running') { console.log(`IronGraph is stopped.\nData: ${ctx.data}\nLogs: ${ctx.log}`); return; }
  console.log(`IronGraph ${state.version} is ${ready ? 'ready' : 'starting'} (PID ${state.pid}).\nData: ${ctx.data}\n${ready ? 'Console' : 'Console when ready'}: ${state.url}\nLogs: ${ctx.log}`);
}

async function stop(ctx) {
  await withLock(ctx, async () => {
    const state = readJson(ctx.state);
    const current = inspected(state, ctx);
    if (current === 'stopped') { cleanRun(state, ctx); console.log(`IronGraph is stopped. Data preserved: ${ctx.data}`); return; }
    if (current === 'unverified') throw new Error(`Refusing to signal PID ${state.pid}: it does not match the recorded IronGraph process. State: ${ctx.state}`);
    try { process.kill(state.pid, 'SIGTERM'); }
    catch (error) { if (error.code !== 'ESRCH') throw error; }
    console.log(`Stopping IronGraph (PID ${state.pid}); waiting for a clean shutdown.`);
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline) {
      await delay(100);
      if (inspected(state, ctx) !== 'running') { cleanRun(state, ctx); console.log(`IronGraph stopped. Data preserved: ${ctx.data}`); return; }
    }
    throw new Error(`Shutdown is still in progress after 30 seconds. No forced termination was sent. Check npx irongraph status and logs. Data: ${ctx.data}`);
  });
}

async function main(argv = process.argv.slice(2)) {
  const opts = options([...argv]);
  if (['--help', '-h', 'help'].includes(opts.command)) return help();
  if (['--version', '-v', 'version'].includes(opts.command)) { console.log(`irongraph ${version}`); return; }
  const ctx = context(opts);
  if (opts.command === 'start') return start(opts, ctx);
  if (opts.command === 'status') return status(ctx);
  if (opts.command === 'stop') return stop(ctx);
  const tail = logTail(ctx.log);
  console.log(tail || `No logs yet. Start IronGraph for ${ctx.data}.`);
}

if (require.main === module) main().catch((error) => { console.error(`IronGraph: ${error.message}`); process.exitCode = 1; });
module.exports = { address, context, inspected, main, nativePackage, options, processIdentity };
