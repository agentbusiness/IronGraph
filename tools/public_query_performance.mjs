import assert from 'node:assert/strict'
import { createHash, randomUUID } from 'node:crypto'
import { Agent, request as httpRequest } from 'node:http'
import { StringDecoder } from 'node:string_decoder'
import { createRequire } from 'node:module'
import { mkdtemp, readFile, mkdir, writeFile, rm } from 'node:fs/promises'
import { createReadStream, createWriteStream } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve, dirname } from 'node:path'
import { spawn, execFile, execFileSync } from 'node:child_process'
import { performance } from 'node:perf_hooks'
import { fileURLToPath } from 'node:url'
import { Client } from '../bindings/javascript/dist/index.js'
import { queryWorkloads } from './public_query_workloads.mjs'
import { rawBoltPool } from './raw_bolt_client.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const args = process.argv.slice(2)
const option = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback
const config = {
  transport: option('--transport', 'api'),
  api_client: option('--api-client', 'javascript'),
  bolt_client: option('--bolt-client', 'node'),
  label: option('--label', 'current'),
  sizes: option('--sizes', '100,10000,100000,1000000,2000000').split(',').map(Number),
  samples: Number(option('--samples', '5')),
  warmups: Number(option('--warmups', '2')),
  fixture_batch_rows: Number(option('--fixture-batch-rows', '8192')),
  port: Number(option('--port', '19484')),
  binary: resolve(option('--binary', 'target/release/irongraph')),
  native: option('--native', null),
  output: resolve(option('--output', 'performance-results/public-query/current/results.json')),
  operations: option('--operations', '').split(',').filter(Boolean),
  flights: args.includes('--flights'),
  memory_control: args.includes('--memory-control'),
}
assert(['api', 'embedded', 'bolt'].includes(config.transport))
assert(['javascript', 'node', 'raw'].includes(config.api_client))
assert(['node', 'raw'].includes(config.bolt_client))
assert(config.sizes.every(n => Number.isSafeInteger(n) && n >= 100))
const wanted = name => !config.operations.length || config.operations.includes(name)
const digest = async path => {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  return hash.digest('hex')
}
const hardware = {
  cpu: execFileSync('sysctl', ['-n', 'machdep.cpu.brand_string'], { encoding: 'utf8' }).trim(),
  memory_bytes: Number(execFileSync('sysctl', ['-n', 'hw.memsize'], { encoding: 'utf8' })),
  os: process.platform, arch: process.arch,
}
const provenance = {
  harness_sha256: await digest(fileURLToPath(import.meta.url)),
  workload_sha256: await digest(join(root, 'tools/public_query_workloads.mjs')),
  executable_sha256: await digest(config.transport === 'embedded' ? resolve(config.native) : config.binary),
  javascript_client_sha256: await digest(join(root, 'bindings/javascript/dist/index.js')),
  ...(config.transport === 'bolt' && config.bolt_client === 'raw'
    ? { bolt_codec_sha256: await digest(join(root, 'tools/raw_bolt_client.mjs')) } : {}),
  ...((config.transport === 'bolt' && config.bolt_client === 'node') || (config.transport === 'api' && config.api_client === 'node')
    ? { native_client_sha256: await digest(resolve(config.native)) } : {}),
}
const scalar = result => Number(result.rows[0][0].value)

// Independent HTTP protocol control: complete NDJSON decoding with all columns and values.
// Uses no database internals and applies no response/result cap.
function rawApiClient(url) {
  const agent = new Agent({ keepAlive: true })
  return {
    close: () => agent.destroy(),
    query: (cypher, parameters = {}) => new Promise((resolveResult, reject) => {
      const body = JSON.stringify({ request_id: randomUUID(), project_id: null,
        query: cypher, parameters, bookmark: null })
      const request = httpRequest(new URL('/api/query', url), {
        agent, method: 'POST', headers: { Accept: 'application/x-ndjson',
          'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
      }, response => {
        if (response.statusCode !== 200) {
          response.resume()
          reject(Error(`Raw API HTTP ${response.statusCode}`))
          return
        }
        const result = { columns: [], rows: [], summary: {} }
        const decoder = new StringDecoder('utf8')
        const fragments = []
        let summarySeen = false
        const event = line => {
          if (!line.trim()) return
          const item = JSON.parse(line)
          if (item.type === 'error') throw Error(`${item.code}: ${item.message}`)
          if (item.type === 'catalog') { const {type, ...catalog} = item; result.catalog = catalog }
          if (item.type === 'schema') result.columns = item.columns
          if (item.type === 'batch') {
            assert(item.columns.every(column => column.values.length === item.row_count))
            for (let row = 0; row < item.row_count; row++) {
              result.rows.push(item.columns.map(column => column.values[row]))
            }
          }
          if (item.type === 'summary') { const {type, ...summary} = item; result.summary = summary; summarySeen = true }
        }
        const chunk = text => {
          let start = 0
          for (let end; (end = text.indexOf('\n', start)) !== -1; start = end + 1) {
            fragments.push(text.slice(start, end))
            event(fragments.join(''))
            fragments.length = 0
          }
          if (start < text.length) fragments.push(text.slice(start))
        }
        response.on('data', bytes => {
          try { chunk(decoder.write(bytes)) }
          catch (error) { reject(error); response.destroy() }
        })
        response.on('end', () => {
          try {
            chunk(decoder.end())
            if (fragments.length) event(fragments.join(''))
            assert(summarySeen, 'Raw API response ended without its summary')
            resolveResult(result)
          } catch (error) { reject(error) }
        })
        response.on('error', reject)
      })
      request.on('error', reject)
      request.end(body)
    }),
  }
}
const records = []
const memorySamples = []
const lifecycleMemory = []
const systemMemory = []
const footprintSamples = []
let activeDatabasePid = null
async function recordFootprint(nodes, stage) {
  const raw = await new Promise((resolveOutput, reject) => {
    execFile('vmmap', ['-summary', String(activeDatabasePid)],
      (error, stdout) => error ? reject(error) : resolveOutput(stdout))
  })
  const bytes = pattern => {
    const match = raw.match(pattern)
    assert(match, 'Physical footprint missing from vmmap')
    return Number(match[1]) * ({ B: 1, K: 1024, M: 1024 ** 2, G: 1024 ** 3 }[match[2]])
  }
  footprintSamples.push({ nodes, stage, pid: activeDatabasePid,
    physical_footprint_bytes_rounded: bytes(/Physical footprint:\s+([\d.]+)([BKMG])/),
    peak_physical_footprint_bytes_rounded: bytes(/Physical footprint \(peak\):\s+([\d.]+)([BKMG])/), raw })
}
async function recordSystemMemory(nodes, stage) {
  const read = (binary, parameters) => new Promise((resolveOutput, reject) => {
    execFile(binary, parameters, (error, stdout) => error ? reject(error) : resolveOutput(stdout.trim()))
  })
  systemMemory.push({ nodes, stage, vm_stat: await read('vm_stat', []),
    swap_usage: await read('sysctl', ['-n', 'vm.swapusage']) })
}
await mkdir(dirname(config.output), { recursive: true })
const memoryLog = createWriteStream(config.output + '.memory.jsonl')
function recordMemory(row) {
  memorySamples.push(row)
  memoryLog.write(JSON.stringify(row) + '\n')
}
let activeMemory = null

// Sample outside the timed query path. RSS includes the encoder, allocator,
// result buffers and mapped pages; it is not a canonical graph byte count.
async function startMemoryMonitor(databasePid, nodes) {
  const pids = [...new Set([databasePid, process.pid])]
  let stopped = false
  let periodicSampling = true
  let inFlight = null
  const sample = () => {
    if (inFlight) return inFlight
    if (stopped) return Promise.resolve()
    inFlight = new Promise(resolveSample => {
    execFile('ps', ['-o', 'pid=,rss=', '-p', pids.join(',')], (error, stdout) => {
      if (!error) {
        for (const line of stdout.trim().split('\n')) {
          const [pid, rssKiB] = line.trim().split(/\s+/).map(Number)
          if (Number.isFinite(rssKiB)) recordMemory({
            nodes, timestamp_ms: performance.now(), pid, rss_bytes: rssKiB * 1024,
            role: databasePid === process.pid ? 'embedded_database_and_client'
              : pid === databasePid ? 'database' : config.transport === 'bolt' ? 'bolt_client' : 'api_client',
          })
        }
      } else recordMemory({ nodes, timestamp_ms: performance.now(), status: 'sampling_error', error: String(error) })
      resolveSample()
    })
    }).finally(() => { inFlight = null })
    return inFlight
  }
  await sample()
  const timer = setInterval(() => { if (periodicSampling) sample() }, 100)
  return { sample,
    setPeriodicSampling: async enabled => {
      periodicSampling = enabled
      if (inFlight) await inFlight
    },
    stop: async () => { clearInterval(timer); await sample(); stopped = true } }
}

function memorySince(index) {
  const samples = memorySamples.slice(index)
  return [...new Set(samples.filter(row => row.role).map(row => row.role))].map(role => {
    const rows = samples.filter(row => row.role === role)
    return { role, pid: rows[0].pid, samples: rows.length,
      before_rss_bytes: rows[0].rss_bytes, after_rss_bytes: rows.at(-1).rss_bytes,
      peak_rss_bytes: Math.max(...rows.map(row => row.rss_bytes)),
      retained_delta_bytes: rows.at(-1).rss_bytes - rows[0].rss_bytes }
  })
}

async function save() {
  await mkdir(dirname(config.output), { recursive: true })
  await writeFile(config.output, JSON.stringify({
    schema: 'irongraph.public-query-performance.v1',
    config,
    hardware,
    timing_boundary: 'Before the public query call until the complete decoded QueryResult returns; full rows and all properties included',
    fixture: { fanout: 4, edge_stride: 7919, dirty_body_bytes: 2048,
      dirty_stride: 'max(floor(nodes / 100), 1)', value_modulus: 1000, bucket_modulus: 64,
      construction: 'Cypher through the measured public surface; no private graph access' },
    provenance: { ...provenance, generated_utc: new Date().toISOString() },
    measurements: records,
    memory_method: 'Process RSS sampled asynchronously with ps every 100 ms, plus outside-timing checkpoints. API database and client measured separately; embedded includes both. vmmap physical-footprint and region summaries sampled outside query timers; reported sizes are rounded by vmmap and include GPU-owned memory. Encoder, result buffers and allocator retention included. Shorter RSS peaks may be missed; no single memory metric proves graph duplication.',
    memory_samples: memorySamples,
    lifecycle_memory: lifecycleMemory,
    system_memory: systemMemory,
    physical_footprint_samples: footprintSamples,
  }, null, 2))
}

async function measure(nodes, operation, action, check, extra = {}, cleanup = null) {
  if (!wanted(operation)) return
  const durations = []
  const iterationMemory = []
  const memoryStart = memorySamples.length
  await activeMemory.sample()
  await recordFootprint(nodes, operation + ':before')
  let result
  for (let i = 0; i < config.warmups + config.samples; i++) {
    // Release the preceding result before dispatch, as a consuming client does.
    // Otherwise the harness artificially holds two complete result sets alive.
    result = undefined
    const start = performance.now()
    result = await action()
    const elapsed = (performance.now() - start) * 1000
    check(result)
    if (i >= config.warmups) durations.push(elapsed)
    if (cleanup) await cleanup(result)
    await activeMemory.sample()
    iterationMemory.push({ iteration: i, warmup: i < config.warmups,
      processes: memorySince(memorySamples.length - (config.transport === 'embedded' ? 1 : 2)) })
  }
  durations.sort((a, b) => a - b)
  await activeMemory.sample()
  await recordFootprint(nodes, operation + ':after')
  const row = { nodes, edges: nodes * 4, operation, transport: config.transport,
    samples: durations.length, warmups: config.warmups, unit: 'microseconds',
    p50: durations[Math.floor(durations.length / 2)], min: durations[0], max: durations.at(-1),
    result_rows: result.rows.length, status: 'ok',
    memory: memorySince(memoryStart),
    iteration_memory: iterationMemory,
    ...(result.read_overlapped_write === undefined ? {} : { read_overlapped_write: result.read_overlapped_write }),
    ...extra }
  records.push(row)
  console.log(JSON.stringify(row))
  await save()
}

function complete(result, rows, expected) {
  assert.equal(result.rows.length, rows)
  assert.equal(result.summary.truncated, false, 'Incomplete result')
  if (expected !== undefined) assert.equal(scalar(result), expected)
}

async function openSurface() {
  const data = await mkdtemp(join(tmpdir(), 'irongraph-public-query-'))
  if (config.transport === 'embedded') {
    assert(config.native, '--native must identify the measured release native library')
    process.env.NAPI_RS_NATIVE_LIBRARY_PATH = resolve(config.native)
    const { EmbeddedDatabase } = createRequire(import.meta.url)(join(root, 'bindings/node/index.js'))
    const database = await EmbeddedDatabase.open(data, 'cpu', 0, true, null, 'metal', 0)
    return { pid: process.pid, query: (cypher, parameters = {}) => database.query(cypher, null, parameters),
      close: async preserve => { await database.close(); if (!preserve) await rm(data, { recursive: true, force: true }); else console.error('PRESERVED FAILED FIXTURE', data) } }
  }
  const log = await import('node:fs').then(fs => fs.openSync(join(data, 'server.log'), 'w'))
  const child = spawn(config.binary, ['--data-dir', data,
    '--http-addr', `127.0.0.1:${config.port}`, '--bolt-addr', `127.0.0.1:${config.port + 1}`,
    '--stream-addr', `127.0.0.1:${config.port + 2}`, '--queue-addr', `127.0.0.1:${config.port + 3}`,
    '--execution-backend', 'cpu'], {
    env: { ...process.env, IRONGRAPH_MCP_ADDR: `127.0.0.1:${config.port + 4}` },
    stdio: ['ignore', log, log],
  })
  const exited = new Promise(resolveExit => child.once('exit', resolveExit))
  const url = `http://127.0.0.1:${config.port}`
  let client
  if (config.transport === 'bolt' && config.bolt_client === 'raw') {
    client = rawBoltPool(`bolt://127.0.0.1:${config.port + 1}`)
  } else if (config.api_client === 'node' || config.transport === 'bolt') {
    assert(config.native, '--native must identify the measured public Node.js SDK')
    process.env.NAPI_RS_NATIVE_LIBRARY_PATH = resolve(config.native)
    const { Client: NodeClient } = createRequire(import.meta.url)(join(root, 'bindings/node/index.js'))
    client = config.transport === 'bolt'
      ? NodeClient.bolt(`bolt://127.0.0.1:${config.port + 1}`) : NodeClient.api(url)
  } else client = config.api_client === 'raw' ? rawApiClient(url) : new Client(url)
  const start = performance.now()
  try { while (true) {
    if (child.exitCode !== null || child.signalCode !== null) throw Error(`Database exited: ${await readFile(join(data, 'server.log'), 'utf8')}`)
    try {
      const status = await fetch(`${url}/system/startup`)
      if (status.ok) { if ((await status.json()).phase === 'ready') break }
      else if ((await fetch(`${url}/web/`)).ok) break
    } catch {}
    assert(performance.now() - start < 180000, `Startup observer expired; see ${data}/server.log`)
    await new Promise(resolveWait => setTimeout(resolveWait, 100))
  } } catch (error) {
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGINT')
    await exited
    console.error('PRESERVED FAILED STARTUP', data)
    throw error
  }
  const administration = new Client(url)
  const projects = new Map()
  const remoteQuery = async (cypher, parameters = {}) => {
    if (config.transport === 'bolt') {
      const create = cypher.match(/^CREATE PROJECT (\w+)$/)
      const dataset = cypher.match(/^IMPORT DATASET (\w+)$/)
      if (create || dataset) {
        const result = await administration.query({cypher, parameters})
        const name = (create || dataset)[1]
        const selected = await administration.query({cypher: `USE ${name} RETURN 1`})
        projects.set(name, selected.catalog.project_id)
        return result
      }
      const name = cypher.match(/^USE (\w+) /)?.[1] || 'public_bench'
      assert(projects.has(name), `Bolt fixture project was not selected: ${name}`)
      return client.query(cypher, projects.get(name), parameters)
    }
    return config.api_client === 'node'
    ? client.query(cypher, null, parameters) : config.api_client === 'raw'
      ? client.query(cypher, parameters) : client.query({ cypher, parameters })
  }
  return { pid: child.pid, query: remoteQuery,
    close: async preserve => {
      if (config.api_client === 'raw' || (config.transport === 'bolt' && config.bolt_client === 'raw')) client.close()
      child.kill('SIGINT')
      const exitCode = await exited
      assert.equal(exitCode, 0, `Database shutdown failed; preserved ${data}/server.log`)
      if (!preserve) await rm(data, { recursive: true, force: true })
      else console.error('PRESERVED FAILED FIXTURE', data)
    } }
}

async function runFixture(nodes) {
  const surface = await openSurface()
  activeDatabasePid = surface.pid
  activeMemory = null
  let failed = true
  try {
    activeMemory = await startMemoryMonitor(surface.pid, nodes)
    await recordSystemMemory(nodes, 'empty_graph_encoder_ready')
    await recordFootprint(nodes, 'empty_graph_encoder_ready')
    const { query } = surface
    await query('CREATE PROJECT public_bench')
    const q = (statement, parameters) => query(`USE public_bench ${statement}`, parameters)
    const stride = Math.max(Math.floor(nodes / 100), 1)
    for (const [operation, statement] of [
      ['initial_node_ingest', 'UNWIND range($start, $end) AS row CREATE (:Node {value: row % 1000, bucket: row % 64, body: CASE WHEN row % $stride = 0 THEN toString(row) + ":" + $body ELSE null END}) RETURN count(*)'],
      ['initial_edge_ingest', 'UNWIND range($start, $end) AS row UNWIND range(1, 4) AS step MATCH (source:Node) WHERE id(source) = row + 1 MATCH (target:Node) WHERE id(target) = ((row + step * 7919) % $nodes) + 1 CREATE (source)-[:R]->(target) RETURN count(*)'],
    ]) {
      const memoryStart = memorySamples.length
      await activeMemory.sample()
      await recordFootprint(nodes, operation + ':before')
      const start = performance.now()
      for (let offset = 0; offset < nodes; offset += config.fixture_batch_rows) {
        const end = Math.min(nodes - 1, offset + config.fixture_batch_rows - 1)
        const result = await q(statement, { start: offset, end, stride, body: 'x'.repeat(2048), nodes })
        complete(result, 1, (end - offset + 1) * (operation === 'initial_edge_ingest' ? 4 : 1))
      }
      const elapsed = (performance.now() - start) * 1000
      await activeMemory.sample()
      await recordFootprint(nodes, operation + ':after')
      records.push({ operation, nodes, edges: nodes * 4, transport: config.transport,
        samples: 1, warmups: 0, p50: elapsed, memory: memorySince(memoryStart),
        result_rows: Math.ceil(nodes / config.fixture_batch_rows),
        mutated_elements: operation === 'initial_edge_ingest' ? nodes * 4 : nodes,
        unit: 'microseconds', status: 'ok', durability: 'database acknowledgement; WAL active' })
      console.log(JSON.stringify(records.at(-1)))
      await save()
    }
    complete(await q('MATCH (n:Node) RETURN count(n)'), 1, nodes)
    complete(await q('MATCH ()-[r:R]->() RETURN count(r)'), 1, nodes * 4)
    if (config.memory_control) {
      const statement = 'MATCH (n:Node) WHERE id(n) = 43 RETURN n.value'
      for (let iteration = 0; iteration < 4; iteration++) {
        const periodic = iteration % 2 === 0
        await activeMemory.setPeriodicSampling(periodic)
        const durations = []
        const reported = []
        await activeMemory.sample()
        for (let sample = 0; sample < 520; sample++) {
          const started = performance.now()
          const result = await q(statement)
          const elapsed = (performance.now() - started) * 1000
          complete(result, 1, 42)
          if (sample >= 20) {
            durations.push(elapsed)
            reported.push(Number(result.summary.statistics.elapsed_us))
          }
        }
        await activeMemory.sample()
        durations.sort((a, b) => a - b)
        reported.sort((a, b) => a - b)
        records.push({ nodes, operation: 'memory_sampler_control', iteration,
          transport: config.transport, periodic_memory_sampling: periodic,
          samples: 500, warmups: 20, unit: 'microseconds', status: 'ok',
          p50: durations[250], p95: durations[475],
          server_reported_elapsed_us_p50: reported[250],
          timing_boundary: 'Public query call through complete decoded result; server timing is separately reported, not subtracted' })
        console.log(JSON.stringify(records.at(-1)))
      }
      await activeMemory.setPeriodicSampling(true)
      await save()
    }
    if (config.operations.includes('concurrent_generic_create')) {
      await measure(nodes, 'concurrent_generic_create', async () => {
        const results = await Promise.allSettled(Array.from({ length: 128 }, (_, value) =>
          q('UNWIND range(1,512) AS step CREATE (n:BenchGeneric) SET n.value=$value, n.step=step RETURN count(*)', {value})))
        const failures = results.filter(result => result.status === 'rejected')
        assert.equal(failures.length, 0, failures.map(result => String(result.reason)).join('\n'))
        for (const result of results) complete(result.value, 1, 512)
        return q('MATCH (n:BenchGeneric) RETURN count(n)')
      }, result => complete(result, 1, 65536), {}, () => q('MATCH (n:BenchGeneric) DELETE n'))
    }
    for (const [name, statement, rows, value] of queryWorkloads(nodes)) {
      await measure(nodes, name, () => name === 'public_metadata' ? query(statement) : q(statement),
        result => complete(result, rows, value))
    }
    await measure(nodes, 'parallel_full_count_queries', async () => {
      const rows = await Promise.all(Array.from({ length: 8 }, async () => {
        let count = 0
        for (let i = 0; i < 100; i++) { complete(await q('MATCH (n:Node) RETURN count(n)'), 1, nodes); count++ }
        return count
      }))
      return { rows: Array(rows.reduce((a, b) => a + b)).fill(null), summary: { truncated: false } }
    }, result => complete(result, 800), { query_calls: 800 })
    for (const name of ['canonical_batch_insert', 'reader_during_real_write']) {
      await measure(nodes, name, async () => {
        const items = Array.from({ length: name === 'canonical_batch_insert' ? 256 : 4096 }, (_, value) => value)
        let writeFinished = false
        const write = q('UNWIND $items AS value CREATE (:BenchWrite {value: value}) RETURN count(*)', { items })
          .then(result => { complete(result, 1, items.length); writeFinished = true; return result })
        if (name === 'reader_during_real_write') {
          const read = await q('MATCH (n:Node) RETURN count(n)')
          return { ...read, read_overlapped_write: !writeFinished, pending_write: write }
        }
        return write
      }, result => complete(result, 1, name === 'canonical_batch_insert' ? 256 : nodes),
      { elements_per_sample: name === 'canonical_batch_insert' ? 256 : 1 },
      async result => {
        if (result.pending_write) await result.pending_write
        await q('MATCH (n:BenchWrite) DELETE n')
      })
    }
    if (config.flights && nodes === config.sizes[0]) {
      await query('IMPORT DATASET flights')
      for (const [name, statement, rows] of [
        ['flights_nodes', 'MATCH (n) RETURN n', 13859],
        ['flights_route_ids', 'MATCH (source)-[relationship]->(target) RETURN id(source), id(relationship), id(target)', 66770],
        ['flights_full_routes', 'MATCH (source)-[relationship]->(target) RETURN source, relationship, target', 66770],
      ]) await measure(nodes, name, () => query(`USE flights ${statement}`), result => complete(result, rows),
        { dataset: 'flights', dataset_nodes: 13859, dataset_edges: 66770 })
    }
    failed = false
  } catch (error) {
    records.push({ nodes, edges: nodes * 4, operation: 'fixture_or_query_failure',
      transport: config.transport, status: 'error', error: String(error) })
    await save()
    throw error
  } finally {
    const start = memorySamples.length
    try {
      await activeMemory?.sample()
      await recordFootprint(nodes, 'before_close_and_durable_snapshot')
    } catch (error) {
      failed = true
      records.push({ nodes, operation: 'close_memory_failure', status: 'error', error: String(error) })
    }
    try { await surface.close(failed) }
    catch (error) {
      failed = true
      records.push({ nodes, operation: 'close_durability_failure', status: 'error', error: String(error) })
      throw error
    }
    finally {
      await activeMemory?.stop()
      lifecycleMemory.push({ nodes, stage: 'close_and_durable_snapshot', memory: memorySince(start) })
      await recordSystemMemory(nodes, 'after_close')
      await save()
    }
  }
}

for (const nodes of config.sizes) await runFixture(nodes)
await new Promise(resolveEnd => memoryLog.end(resolveEnd))
console.log('PUBLIC QUERY PERFORMANCE COMPLETE')
