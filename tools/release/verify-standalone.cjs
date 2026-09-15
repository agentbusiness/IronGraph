#!/usr/bin/env node
'use strict'

// Runs against installed release archives, outside the source checkout.
const assert = require('node:assert/strict')
const { execFile } = require('node:child_process')
const { randomUUID } = require('node:crypto')
const fs = require('node:fs/promises')
const net = require('node:net')
const os = require('node:os')
const path = require('node:path')
const { promisify } = require('node:util')
const execute = promisify(execFile)
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds))

async function reservePorts() {
  const sockets = []
  try {
    for (let index = 0; index < 5; index++) {
      const server = net.createServer()
      await new Promise((resolve, reject) => {
        server.once('error', reject)
        server.listen(0, '127.0.0.1', resolve)
      })
      sockets.push(server)
    }
    return sockets.map(server => server.address().port)
  } finally {
    await Promise.all(sockets.map(server => new Promise(resolve => server.close(resolve))))
  }
}

async function main() {
  const [consumerArg, version] = process.argv.slice(2)
  assert(consumerArg && /^\d+\.\d+\.\d+$/.test(version), 'Pass installed consumer directory and version')
  const consumer = path.resolve(consumerArg)
  const cli = path.join(consumer, 'node_modules/irongraph/cli.cjs')
  const manifest = JSON.parse(await fs.readFile(path.join(consumer, 'node_modules/irongraph/package.json'), 'utf8'))
  assert.equal(manifest.version, version)
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'irongraph-standalone-check-'))
  const data = path.join(temporary, 'data')
  const [http, bolt, streams, queues, mcp] = await reservePorts()
  const address = `127.0.0.1:${http}`
  const base = `http://${address}`
  const env = { ...process.env, IRONGRAPH_CLI_HOME: path.join(temporary, 'runtime'),
    IRONGRAPH_HTTP_ADDR: address, IRONGRAPH_BOLT_ADDR: `127.0.0.1:${bolt}`,
    IRONGRAPH_STREAM_ADDR: `127.0.0.1:${streams}`, IRONGRAPH_QUEUE_ADDR: `127.0.0.1:${queues}`,
    IRONGRAPH_MCP_ADDR: `127.0.0.1:${mcp}`, IRONGRAPH_EXECUTION_BACKEND: 'cpu' }
  for (const key of Object.keys(env)) {
    if (key.startsWith('IRONGRAPH_REMOTE_')) delete env[key]
  }
  delete env.IRONGRAPH_MCP_BINARY
  delete env.IRONGRAPH_MCP_URL
  const command = async (...args) => {
    const result = await execute(process.execPath, [cli, ...args], {
      cwd: consumer, env, timeout: 120000, maxBuffer: 8 * 1024 * 1024,
    })
    return result.stdout + result.stderr
  }
  const query = async (statement, parameters = {}) => {
    const response = await fetch(`${base}/api/query`, {
      method: 'POST', signal: AbortSignal.timeout(120000),
      headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson' },
      body: JSON.stringify({ request_id: randomUUID(), project_id: null, query: statement, parameters, bookmark: null }),
    })
    assert.equal(response.status, 200, `Query HTTP status for ${statement}`)
    const events = (await response.text()).trim().split('\n').filter(Boolean).map(line => JSON.parse(line))
    const error = events.find(event => event.type === 'error')
    assert(!error, JSON.stringify(error))
    assert(events.some(event => event.type === 'summary'), 'Missing query summary')
    return events.filter(event => event.type === 'batch').flatMap(event =>
      Array.from({ length: event.row_count }, (_, index) => event.columns.map(column => column.values[index])))
  }
  const ready = async () => {
    // The pinned 2.47 GB embedding model can need longer than five minutes on a cold install.
    const deadline = Date.now() + 600000
    let lastError
    while (Date.now() < deadline) {
      try {
        await query('SHOW PROJECTS')
        return
      } catch (error) { lastError = error; await sleep(1000) }
    }
    throw new Error(`Standalone did not become ready: ${lastError}`)
  }
  let started = false
  try {
    assert.match(await command('--version'), new RegExp(version.replaceAll('.', '\\.')))
    const help = await command('--help')
    for (const action of ['start', 'status', 'logs', 'stop']) assert(help.includes(action))
    const launch = await command('start', '--background', '--data-dir', data, '--http-addr', address)
    started = true
    console.log(launch.trim())
    await ready()
    assert.match(await command('status', '--data-dir', data), /ready/i)
    const htmlResponse = await fetch(`${base}/web/`)
    assert.equal(htmlResponse.status, 200)
    const html = await htmlResponse.text()
    assert.match(html, /<title>IronGraph<\/title>/)
    const assets = [...html.matchAll(/(?:src|href)="([^"]+\.(?:js|css))"/g)].map(match => match[1])
    assert(assets.some(asset => asset.endsWith('.js')), 'Missing bundled console JavaScript')
    for (const asset of assets) {
      const response = await fetch(new URL(asset, `${base}/web/`))
      assert.equal(response.status, 200, `Missing console asset ${asset}`)
      assert((await response.arrayBuffer()).byteLength > 0)
    }
    await query('CREATE PROJECT IF NOT EXISTS standalone_check')
    await query("USE standalone_check MERGE (d:Document {id: 'guide'}) SET d.body = $body, d.embedding = [0.0]",
      { body: 'Graph databases connect nodes and relationships.' })
    await query('USE standalone_check CREATE EMBEDDING INDEX guide_text FOR (d:Document) FROM d.body INTO d.embedding USING MODEL default SIMILARITY COSINE')
    const search = 'USE standalone_check MATCH (d:Document) SEARCH d IN (EMBEDDING INDEX guide_text FOR TEXT $text LIMIT 1) SCORE AS score RETURN d.id, d.body, score'
    const result = await query(search, { text: 'connected graph data' })
    assert.equal(result[0][0].value, 'guide')
    assert.equal(result[0][2].type, 'float')
    for (const statement of ['CREATE TOPIC activity PARTITIONS 2', 'CREATE QUEUE jobs STREAM']) {
      await query(`USE standalone_check ${statement}`)
    }
    assert((await query('USE standalone_check SHOW TOPICS')).length > 0)
    assert((await query('USE standalone_check SHOW QUEUES')).length > 0)
    // A second start must either return the same managed instance or reject it; never admit two.
    try { await command('start', '--background', '--data-dir', data, '--http-addr', address) }
    catch (error) { assert.match(error.stdout + error.stderr, /already|running|start|lock/i) }
    assert.equal((await query('USE standalone_check MATCH (d:Document) RETURN d.id')).length, 1)
    assert((await command('logs', '--data-dir', data)).trim().length > 0)
    const runDirectories = await fs.readdir(path.join(env.IRONGRAPH_CLI_HOME, 'run'))
    assert.equal(runDirectories.length, 1)
    const state = JSON.parse(await fs.readFile(path.join(env.IRONGRAPH_CLI_HOME, 'run', runDirectories[0], 'instance.json'), 'utf8'))
    assert(path.isAbsolute(state.mcpBinary), 'MCP must have a permanent absolute executable path')
    const profile = path.join(temporary, 'host-profile')
    await execute(state.mcpBinary, ['integrations', '--root', profile, 'install', 'zed', '--json'], {
      env: { ...env, IRONGRAPH_MCP_BINARY: state.mcpBinary, IRONGRAPH_MCP_URL: base }, timeout: 10000,
    })
    const hostConfig = JSON.parse(await fs.readFile(path.join(profile, '.config/zed/settings.json'), 'utf8'))
    assert.equal(hostConfig.context_servers.irongraph.command, state.mcpBinary)
    assert.equal(hostConfig.context_servers.irongraph.env.IRONGRAPH_MCP_URL, base)
    await command('stop', '--data-dir', data)
    started = false
    assert.equal((await execute(state.mcpBinary, ['--version'], { env, timeout: 10000 })).stdout.trim(),
      `irongraph-mcp ${version}`, 'Installed host MCP executable must survive database stop')
    assert.match(await command('status', '--data-dir', data), /stopped|not running/i)
    await assert.rejects(fetch(`${base}/web/`, { signal: AbortSignal.timeout(2000) }))
    await command('start', '--background', '--data-dir', data, '--http-addr', address)
    started = true
    await ready()
    assert.equal((await query(search, { text: 'connected graph data' }))[0][1].value,
      'Graph databases connect nodes and relationships.')
    await command('stop', '--data-dir', data)
    started = false
    console.log('standalone installed verification passed')
  } catch (error) {
    try { console.error(await command('logs', '--data-dir', data)) } catch {}
    throw error
  } finally {
    if (started) {
      try { await command('stop', '--data-dir', data) } catch (error) {
        console.error('Standalone cleanup could not stop the managed instance:', error)
      }
    }
    try { await fs.rm(temporary, { recursive: true, force: true }) } catch (error) {
      console.error('Standalone cleanup could not remove its temporary directory:', error)
    }
  }
}

main().catch(error => { console.error(error); process.exitCode = 1 })
