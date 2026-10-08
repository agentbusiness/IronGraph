'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const http = require('node:http')
const os = require('node:os')
const path = require('node:path')
const { Client, EmbeddedDatabase } = require('./index.js')

async function remoteApiSmoke() {
  const vectorNumbers = [0.1, 1.234567, -(2 ** -149), 2 ** -149, 3.4028234663852886e38]
  let requireParallel = false
  const waiting = []
  const server = http.createServer((request, response) => {
    assert.equal(request.method, 'POST')
    assert.equal(request.url, '/api/query')
    let body = ''
    request.setEncoding('utf8')
    request.on('data', (chunk) => { body += chunk })
    request.on('end', () => {
      const query = JSON.parse(body)
      const finish = () => {
      const vector = query.query.startsWith('RETURN vector_fidelity')
      const rowCount = query.query.endsWith('_async_rows') ? 8192 : query.query.endsWith('_rows') ? 17 : 1
      const name = vector ? 'vector' : 'answer'
      const valueType = vector ? 'VECTOR' : 'INTEGER'
      const value = vector ? { type: 'vector', value: vectorNumbers } : { type: 'integer', value: '42' }
      response.writeHead(200, { 'content-type': 'application/x-ndjson' })
      response.end([
        JSON.stringify({ type: 'schema', request_id: query.request_id, columns: [{ name, value_type: valueType, nullable: false }] }),
        JSON.stringify({ type: 'batch', request_id: query.request_id, sequence: 0, row_count: rowCount, columns: [{ name, value_type: valueType, values: Array.from({ length: rowCount }, () => value) }] }),
        JSON.stringify({ type: 'summary', request_id: query.request_id, bookmark: { term: 1, index: 1 }, statistics: { elapsed_ms: 0, elapsed_us: 0, rows: rowCount, nodes: 0, edges: 0, updates: 0 }, truncated: false, truncation_reason: null }),
      ].join('\n') + '\n')
      }
      if (!requireParallel) finish()
      else {
        waiting.push(finish)
        if (waiting.length === 2) waiting.splice(0).forEach(reply => reply())
      }
    })
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  try {
    const address = server.address()
    const client = Client.api(`http://127.0.0.1:${address.port}`)
    const result = await client.query('RETURN 42 AS answer')
    assert.equal(result.rows[0][0].value, '42')
    const vector = await client.query('RETURN vector_fidelity')
    assert.deepEqual(vector.rows[0][0].value, vectorNumbers.map(Math.fround))
    const vectors = await client.query('RETURN vector_fidelity_rows')
    assert.equal(vectors.rows.length, 17)
    for (const row of vectors.rows) assert.deepEqual(row[0].value, vector.rows[0][0].value)
    const originalParse = JSON.parse
    let conversionStarted = false
    let eventLoopRanDuringConversion = false
    try {
      JSON.parse = function (text, ...args) {
        const source = String(text)
        if (!conversionStarted && source.startsWith('[{"type":"vector"')) {
          conversionStarted = true
          setImmediate(() => { eventLoopRanDuringConversion = true })
        }
        return originalParse(text, ...args)
      }
      const streamed = await client.query('RETURN vector_fidelity_async_rows')
      assert.equal(streamed.rows.length, 8192)
      assert.ok(conversionStarted)
      assert.ok(eventLoopRanDuringConversion, 'result conversion blocked the event loop until all rows were decoded')
      for (const row of streamed.rows) assert.deepEqual(row[0].value, vector.rows[0][0].value)
    } finally { JSON.parse = originalParse }
    requireParallel = true
    let watchdog
    try {
      const results = await Promise.race([
        Promise.all([client.query('RETURN 42 AS answer'), client.query('RETURN 42 AS answer')]),
        new Promise((_, reject) => { watchdog = setTimeout(() => reject(Error('one remote client serialized concurrent requests')), 5000) }),
      ])
      assert.ok(results.every(result => result.rows[0][0].value === '42'))
    } finally { clearTimeout(watchdog); waiting.splice(0).forEach(reply => reply()) }
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()))
  }
}

async function main() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'irongraph-node-'))
  try {
    const database = await EmbeddedDatabase.open(directory, 'cpu', 0, false)
    await database.query('CREATE PROJECT app')
    const dirtyBody = 'é漢字\\\"\n\u0000\u2028' + 'large body '.repeat(32768)
    const lossless = await database.query('USE app UNWIND range(1,17) AS row RETURN $body AS body, 9007199254740993 AS integer', null, { body: dirtyBody })
    assert.equal(lossless.rows.length, 17)
    for (const row of lossless.rows) {
      assert.equal(row[0].value, dirtyBody)
      assert.equal(row[1].value, '9007199254740993')
    }
    const largeBody = '漢字'.repeat(256 * 1024)
    const largeRows = await database.query('USE app UNWIND range(1,17) AS row RETURN $body AS body', null, { body: largeBody })
    assert.equal(largeRows.rows.length, 17)
    for (const row of largeRows.rows) assert.equal(row[0].value, largeBody)
    await database.query('USE app CREATE (:Item {value: 42})')
    await Promise.all(Array.from({ length: 128 }, (_, value) =>
      database.query('USE app CREATE (:Parallel {value: $value})', null, { value })))
    const parallel = await database.query('USE app MATCH (n:Parallel) RETURN count(n) AS count')
    assert.equal(parallel.rows[0][0].value, '128')
    await database.query('USE app CREATE (:Document {id: $id, body: $body, embedding: [1.0, 0.0]})', null,
      { id: 'guide', body: 'Complete document body — searchable text' })
    await database.query('USE app CREATE TEXT INDEX document_body FOR (d:Document) ON (d.body)')
    await database.query('USE app MATCH (d:Document) SET d.body = $body', null,
      { body: 'Updated complete document body' })
    const ranked = await database.query("USE app MATCH (d:Document) WHERE d.body CONTAINS 'complete' RETURN d.id, vector.cosine(d.embedding, [1.0, 0.0]) AS score")
    assert.equal(ranked.rows[0][0].value, 'guide')
    assert.ok(Math.abs(ranked.rows[0][1].value - 1.0) < 1e-6)
    await database.query("USE app USE LAYER WORKSPACE WRITE LAYER WORKSPACE CREATE (:Draft {id: 'draft'})")
    assert.equal((await database.query('USE app MATCH (d:Draft) RETURN d')).rows.length, 0)
    assert.equal((await database.query('USE app USE LAYER WORKSPACE MATCH (d:Draft) RETURN d')).rows.length, 1)
    for (const statement of [
      'CREATE TOPIC activity PARTITIONS 2', 'CREATE EXCHANGE routing TYPE TOPIC',
      'CREATE QUEUE jobs STREAM', 'BIND QUEUE jobs TO EXCHANGE routing KEY documents',
    ]) await database.query(`USE app ${statement}`)
    for (const statement of ['SHOW TOPICS', 'SHOW QUEUES', 'SHOW EXCHANGES', 'SHOW INDEXES']) {
      assert.ok((await database.query(`USE app ${statement}`)).rows.length > 0, statement)
    }
    const projectId = (await database.query('USE app RETURN 1')).catalog.project_id
    assert.equal(database.status().ready, true)
    const append = await database.streamAppend({ project_id: projectId, topic: 'activity', partition: 0,
      records: [{ key: [0, 255], headers: { source: [78] }, value: [1, 2, 3], create_time_ms: 1234 }] })
    assert.equal(append.first_offset, 0)
    const fetch = { project_id: projectId, topic: 'activity', partition: 0, offset: 0, max_records: 1, max_bytes: 4096 }
    const page = await database.streamFetch(fetch)
    assert.equal(page.high_watermark, 1)
    assert.deepEqual(page.records[0][1].payload, [1, 2, 3])
    const complete = await database.query('UNWIND [1,2,3] AS n RETURN n', projectId, null, { bookmark: append.bookmark })
    assert.equal(complete.rows.length, 3)
    await assert.rejects(database.query('CREATE (:Rejected)', projectId, null, null, { timeout_ms: 0 }))
    await database.snapshot()
    await database.flush()
    await database.close()

    const reopened = await EmbeddedDatabase.open(directory, 'cpu', 0, false)
    assert.deepEqual((await reopened.streamFetch(fetch)).records, page.records)
    const result = await reopened.query('USE app MATCH (n:Item) RETURN n.value AS value')
    assert.equal(result.rows[0][0].type, 'integer')
    assert.equal(result.rows[0][0].value, '42')
    const document = await reopened.query("USE app MATCH (d:Document {id: 'guide'}) RETURN d.body")
    assert.equal(document.rows[0][0].value, 'Updated complete document body')
    await reopened.query("USE app MATCH (d:Document {id: 'guide'}) DELETE d")
    await reopened.close()
    const deleted = await EmbeddedDatabase.open(directory, 'cpu', 0, false)
    assert.equal((await deleted.query('USE app MATCH (d:Document) RETURN d')).rows.length, 0)
    await deleted.close()
    await remoteApiSmoke()
    if (process.env.IRONGRAPH_QUALIFY_EMBEDDINGS === '1') {
      const semantic = await EmbeddedDatabase.open(path.join(directory, 'semantic'), 'cpu', 0, true, null,
        process.env.IRONGRAPH_QUALIFY_DEVICE || 'cpu', 0)
      try {
        await semantic.query('CREATE PROJECT meaning')
        await semantic.query("USE meaning CREATE (p:Person {name:'Ada'}), (t:Task {title:'Arrange lessons'}), (t)-[:ASSIGNED_TO {description:'guitar music tuition'}]->(p)")
        const search = "USE meaning SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'guitar music tuition' LIMIT 10) SCORE AS score RETURN entity, score"
        const deadline = Date.now() + 120000
        let matches
        do {
          matches = await semantic.query(search)
          if (matches.rows.length === 3) break
          assert.ok(Date.now() < deadline, 'asynchronous semantic embeddings did not finish')
          await new Promise(resolve => setTimeout(resolve, 10))
        } while (true)
        assert.equal(matches.rows.length, 3)
        assert.equal(matches.rows[0][0].type, 'relationship')
        const scores = matches.rows.map((row) => row[1].value)
        assert.deepEqual(scores, [...scores].sort((a, b) => b - a))
        await semantic.query("USE meaning MATCH ()-[r]->() SET r.description = 'contract negotiation'")
        const updatedSearch = search.replace('guitar music tuition', 'contract negotiation').replace('LIMIT 10', 'LIMIT 1')
        let updated
        const updateDeadline = Date.now() + 120000
        do {
          updated = await semantic.query(updatedSearch)
          if (updated.rows[0]?.[0].type === 'relationship') break
          assert.ok(Date.now() < updateDeadline, 'asynchronous relationship embedding did not update')
          await new Promise(resolve => setTimeout(resolve, 10))
        } while (true)
        assert.equal(updated.rows[0][0].type, 'relationship')
        await semantic.snapshot()
      } finally {
        await semantic.close()
      }
    }
  } finally {
    fs.rmSync(directory, { recursive: true, force: true })
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
