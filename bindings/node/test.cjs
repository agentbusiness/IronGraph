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
    await database.snapshot()
    await database.close()

    const reopened = await EmbeddedDatabase.open(directory, 'cpu', 0, false)
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
  } finally {
    fs.rmSync(directory, { recursive: true, force: true })
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
