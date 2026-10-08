import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { spawn, execFile } from 'node:child_process'
import { createReadStream, createWriteStream } from 'node:fs'
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { promisify } from 'node:util'
import { queryWorkloads } from './public_query_workloads.mjs'

const exec = promisify(execFile)
const args = process.argv.slice(2)
const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback
const config = {
  language: option('--language', 'python'), transport: option('--transport', 'api'),
  sizes: option('--sizes', '100,10000,100000').split(',').map(Number),
  binary: resolve(option('--binary', 'target/release/irongraph')),
  runner: resolve(option('--runner', 'tools/public_sdk_performance.py')),
  sdk_artifact: resolve(option('--sdk-artifact', 'target/release/lib_native.dylib')),
  output: resolve(option('--output', 'performance-results/public-query/python-api/results.json')),
  samples: Number(option('--samples', '5')), warmups: Number(option('--warmups', '2')),
  port: Number(option('--port', '19584')), flights: args.includes('--flights'),
}
assert(['api', 'bolt', 'embedded', 'internal'].includes(config.transport))
assert(['python', 'rust', 'internal'].includes(config.language))
const digest = async path => {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  return hash.digest('hex')
}
const report = {
  schema: 'irongraph.public-sdk-performance.v1', config,
  timing_boundary: 'Inside the actual SDK caller: immediately before query until the complete decoded result returns; no parent IPC included. Full result verification follows each call.',
  measurements: [], memory_samples: [], physical_footprint_samples: [], lifecycle_memory: [],
  provenance: { harness_sha256: await digest(import.meta.filename), runner_sha256: await digest(config.runner),
    workload_sha256: await digest(resolve('tools/public_query_workloads.mjs')),
    sdk_sha256: await digest(config.sdk_artifact), executable_sha256: await digest(config.binary) },
  hardware: { cpu: (await exec('sysctl', ['-n', 'machdep.cpu.brand_string'])).stdout.trim(),
    memory_bytes: Number((await exec('sysctl', ['-n', 'hw.memsize'])).stdout.trim()) },
  memory_method: 'External ps RSS every 100 ms, plus process physical footprints via vmmap at stage boundaries outside timers. The caller waits for these checkpoints. Short peaks may be missed; result buffers, GPU embedding memory and allocator retention are included.',
}
await mkdir(dirname(config.output), { recursive: true })
const rawMemory = createWriteStream(config.output + '.memory.jsonl')
const save = () => writeFile(config.output, JSON.stringify(report, null, 2))

for (const nodes of config.sizes) {
  const data = await mkdtemp(join(tmpdir(), 'irongraph-public-sdk-'))
  const manifest = join(data, 'workloads.json')
  await writeFile(manifest, JSON.stringify({nodes, workloads: queryWorkloads(nodes),
    http_endpoint: `http://127.0.0.1:${config.port}`, flights: config.flights && nodes === config.sizes[0],
    flight_workloads: [
      ['flights_nodes', 'MATCH (n) RETURN n', 13859],
      ['flights_route_ids', 'MATCH (source)-[relationship]->(target) RETURN id(source),id(relationship),id(target)', 66770],
      ['flights_full_routes', 'MATCH (source)-[relationship]->(target) RETURN source,relationship,target', 66770],
    ]}))
  let server, worker, sampling, samplingBusy = false, samplingJob = Promise.resolve(), failed = true
  let closeMemoryStart = 0
  const roles = new Map()
  const stopping = new Set()
  const sample = async () => {
    for (const [pid, role] of roles) {
      if (!pid) continue
      let stdout
      try { ({stdout} = await exec('ps', ['-o', 'rss=,stat=', '-p', String(pid)])) }
      catch (error) {
        if (!roles.has(pid)) continue
        try { process.kill(pid, 0) }
        catch (exitError) {
          if (exitError.code !== 'ESRCH') throw exitError
          roles.delete(pid)
          ;(report.process_exits_during_sampling ??= []).push({nodes, pid, role, utc: new Date().toISOString()})
          continue
        }
        throw error
      }
      // The child can exit while ps runs, before or after its exit callback removes the role.
      if (!roles.has(pid)) continue
      const [rss, state] = stdout.trim().split(/\s+/)
      if (state?.startsWith('Z')) {
        roles.delete(pid)
        ;(report.process_exits_during_sampling ??= []).push({nodes, pid, role, state, utc: new Date().toISOString()})
        continue
      }
      const bytes = Number(rss) * 1024
      if (!(bytes > 0)) {
        try { process.kill(pid, 0) }
        catch (error) {
          if (error.code !== 'ESRCH') throw error
          roles.delete(pid)
          ;(report.process_exits_during_sampling ??= []).push({nodes, pid, role, utc: new Date().toISOString()})
          continue
        }
        if (stopping.has(pid)) {
          // macOS can release RSS before waitpid delivers the requested shutdown's exit.
          // Keep this event explicit; the shutdown below must still exit successfully.
          ;(report.shutdown_sampling_events ??= []).push({nodes, pid, role, state,
            rss_bytes: bytes, utc: new Date().toISOString()})
          continue
        }
      }
      assert(bytes > 0, `Missing live ${role} RSS: ${JSON.stringify({pid, state, stdout})}`)
      const value = {nodes, pid, role, rss_bytes: bytes, utc: new Date().toISOString()}
      report.memory_samples.push(value)
      rawMemory.write(JSON.stringify(value) + '\n')
    }
  }
  const exited = child => new Promise(resolveExit => child.once('exit', (code, signal) => resolveExit({code, signal})))
  let serverExit, workerExit
  try {
    if (['api', 'bolt'].includes(config.transport)) {
      const log = createWriteStream(join(data, 'server.log'))
      server = spawn(config.binary, ['--data-dir', join(data, 'database'), '--execution-backend', 'cpu',
        '--http-addr', `127.0.0.1:${config.port}`, '--bolt-addr', `127.0.0.1:${config.port + 1}`,
        '--stream-addr', `127.0.0.1:${config.port + 2}`, '--queue-addr', `127.0.0.1:${config.port + 3}`],
      {env: {...process.env, IRONGRAPH_MCP_ADDR: `127.0.0.1:${config.port + 4}`}, stdio: ['ignore', 'pipe', 'pipe']})
      server.stdout.pipe(log, {end: false}); server.stderr.pipe(log)
      serverExit = exited(server)
      server.once('exit', () => roles.delete(server.pid))
      roles.set(server.pid, 'database')
      const start = Date.now()
      while (true) {
        assert(server.exitCode === null && server.signalCode === null, `Server exited; ${data}/server.log`)
        try { if ((await (await fetch(`http://127.0.0.1:${config.port}/system/startup`)).json()).phase === 'ready') break } catch {}
        assert(Date.now() - start < 180000, `Startup observer expired; ${data}/server.log`)
        await new Promise(done => setTimeout(done, 100))
      }
    }
    const command = config.language === 'python' ? 'python3' : config.runner
    const runnerArgs = [...(config.language === 'python' ? [config.runner] : []), '--transport', config.transport,
      '--manifest', manifest, '--data-dir', join(data, 'database'), '--endpoint',
      `${config.transport === 'bolt' ? 'bolt' : 'http'}://127.0.0.1:${config.port + (config.transport === 'bolt' ? 1 : 0)}`,
      '--samples', String(config.samples), '--warmups', String(config.warmups)]
    worker = spawn(command, runnerArgs, {stdio: ['pipe', 'pipe', 'pipe'], env: process.env})
    workerExit = exited(worker)
    worker.once('exit', () => roles.delete(worker.pid))
    worker.stderr.pipe(createWriteStream(join(data, 'caller.log')))
    roles.set(worker.pid, server ? `${config.language}_client` : `${config.language}_database_and_client`)
    sampling = setInterval(() => {
      if (samplingBusy) return
      samplingBusy = true
      samplingJob = sample().catch(error => {
        report.memory_samples.push({nodes, status: 'sampling_error', error: String(error)})
      }).finally(() => { samplingBusy = false })
    }, 100)
    const stages = new Map()
    for await (const line of createInterface({input: worker.stdout})) {
      if (!line.startsWith('{')) continue
      const event = JSON.parse(line)
      if (event.event === 'checkpoint') {
        if (event.stage === 'before_close_and_durable_snapshot') closeMemoryStart = report.memory_samples.length
        await samplingJob
        await sample()
        if (event.stage.endsWith(':before')) stages.set(event.stage.slice(0, -7), report.memory_samples.length - roles.size)
        if (!event.stage.includes(':iteration:')) {
          for (const [pid, role] of roles) {
            const {stdout} = await exec('vmmap', ['-summary', String(pid)], {maxBuffer: 8 * 1024 * 1024})
            report.physical_footprint_samples.push({nodes, stage: event.stage, role, pid, raw: stdout})
          }
        }
        worker.stdin.write('continue\n')
      } else if (event.event === 'measurement') {
        const samples = report.memory_samples.slice(stages.get(event.operation))
        const memory = [...roles].map(([pid, role]) => {
          const values = samples.filter(row => row.pid === pid && row.rss_bytes)
          assert(values.length > 0)
          return {role, before_rss_bytes: values[0].rss_bytes, after_rss_bytes: values.at(-1).rss_bytes,
            peak_rss_bytes: Math.max(...values.map(row => row.rss_bytes)), samples: values.length}
        })
        report.measurements.push({...event, language: config.language, transport: config.transport, memory})
        await save()
        console.log(nodes, event.operation, event.p50)
      } else {
        assert(event.event === 'complete', `Unexpected SDK event ${line}`)
        roles.delete(worker.pid)
      }
    }
    assert((await workerExit).code === 0, `SDK runner failed; ${data}/caller.log`)
    failed = false
  } catch (error) {
    report.measurements.push({nodes, status: 'error', error: String(error)})
    throw error
  } finally {
    if (worker && worker.exitCode === null && worker.signalCode === null) { stopping.add(worker.pid); worker.kill('SIGINT'); await workerExit }
    roles.delete(worker?.pid)
    if (server && server.exitCode === null && server.signalCode === null) { stopping.add(server.pid); server.kill('SIGINT') }
    const shutdown = server ? await serverExit : null
    clearInterval(sampling)
    await samplingJob
    report.lifecycle_memory.push({nodes, stage: 'close_and_durable_snapshot', samples: report.memory_samples.slice(closeMemoryStart)})
    if (shutdown && shutdown.code !== 0) {
      failed = true
      report.measurements.push({nodes, status: 'error', error: `Database shutdown failed: ${JSON.stringify(shutdown)}`})
    }
    await save()
    if (failed) console.error('PRESERVED FAILED SDK FIXTURE', data)
    else await rm(data, {recursive: true, force: true})
    assert(!shutdown || shutdown.code === 0, `Database shutdown failed; ${data}/server.log`)
  }
}
await new Promise(done => rawMemory.end(done))
console.log('PUBLIC SDK PERFORMANCE COMPLETE')
