'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const http = require('node:http')
const os = require('node:os')
const path = require('node:path')
const { Client, EmbeddedDatabase } = require('./index.js')

async function remoteApiSmoke() {
  const server = http.createServer((request, response) => {
    assert.equal(request.method, 'POST')
    assert.equal(request.url, '/api/query')
    let body = ''
    request.setEncoding('utf8')
    request.on('data', (chunk) => { body += chunk })
    request.on('end', () => {
      const query = JSON.parse(body)
      response.writeHead(200, { 'content-type': 'application/x-ndjson' })
      response.end([
        JSON.stringify({ type: 'schema', request_id: query.request_id, columns: [{ name: 'answer', value_type: 'INTEGER', nullable: false }] }),
        JSON.stringify({ type: 'batch', request_id: query.request_id, sequence: 0, row_count: 1, columns: [{ name: 'answer', value_type: 'INTEGER', values: [{ type: 'integer', value: '42' }] }] }),
        JSON.stringify({ type: 'summary', request_id: query.request_id, bookmark: { term: 1, index: 1 }, statistics: { elapsed_ms: 0, elapsed_us: 0, rows: 1, nodes: 0, edges: 0, updates: 0 }, truncated: false, truncation_reason: null }),
      ].join('\n') + '\n')
    })
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  try {
    const address = server.address()
    const client = Client.api(`http://127.0.0.1:${address.port}`)
    const result = await client.query('RETURN 42 AS answer')
    assert.equal(result.rows[0][0].value, '42')
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()))
  }
}

async function main() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'irongraph-node-'))
  try {
    const database = await EmbeddedDatabase.open(directory, 'cpu', 0, false)
    await database.query('CREATE PROJECT app')
    await database.query('USE app CREATE (:Item {value: 42})')
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
    await assert.rejects(database.query('UNWIND [1,2,3] AS n RETURN n', projectId, null, { bookmark: append.bookmark, limits: { rows: 1 } }), /ResultBudgetExceeded/)
    const bounded = await database.query('UNWIND [1,2,3] AS n RETURN n', projectId, null, { bookmark: append.bookmark, limits: { rows: 3 } })
    assert.equal(bounded.rows.length, 3)
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
      const semantic = await EmbeddedDatabase.open(path.join(directory, 'semantic'), process.env.IRONGRAPH_QUALIFY_DEVICE || 'cpu', 0, true)
      try {
        await semantic.query('CREATE PROJECT meaning')
        await semantic.query("USE meaning CREATE (p:Person {name:'Ada'}), (t:Task {title:'Arrange lessons'}), (t)-[:ASSIGNED_TO {description:'guitar music tuition'}]->(p)")
        const search = "USE meaning SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'guitar music tuition' LIMIT 10) SCORE AS score RETURN entity, score"
        const matches = await semantic.query(search)
        assert.equal(matches.rows.length, 3)
        assert.equal(matches.rows[0][0].type, 'relationship')
        const scores = matches.rows.map((row) => row[1].value)
        assert.deepEqual(scores, [...scores].sort((a, b) => b - a))
        await semantic.query("USE meaning MATCH ()-[r]->() SET r.description = 'contract negotiation'")
        assert.equal((await semantic.query(search.replace('guitar music tuition', 'contract negotiation').replace('LIMIT 10', 'LIMIT 1'))).rows[0][0].type, 'relationship')
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
