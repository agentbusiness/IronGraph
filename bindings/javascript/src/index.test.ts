import { afterEach, describe, expect, it, vi } from 'vitest'
import { Client } from './index.js'

afterEach(() => vi.unstubAllGlobals())

describe('Client', () => {
  it('posts only to /api/query and transposes NDJSON column batches', async () => {
    const fetchMock = vi.fn(async (url: URL, init: RequestInit) => {
      expect(url.pathname).toBe('/api/query')
      expect(init.method).toBe('POST')
      expect(JSON.parse(String(init.body))).not.toHaveProperty('limits')
      const body = [
        JSON.stringify({ type: 'schema', columns: [{ name: 'x', value_type: 'INTEGER', nullable: false }] }),
        JSON.stringify({ type: 'batch', row_count: 2, columns: [{ values: [{ type: 'integer', value: '1' }, { type: 'integer', value: '2' }] }] }),
        JSON.stringify({ type: 'summary', bookmark: { term: 0, index: 1 } }),
      ].join('\n')
      return new Response(body, { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('crypto', { randomUUID: () => '00000000-0000-4000-8000-000000000000' })

    const result = await new Client('http://localhost:18484').query({ cypher: 'RETURN 1 AS x' })
    expect(result.rows).toHaveLength(2)
    expect(result.rows[1][0]).toEqual({ type: 'integer', value: '2' })
    expect(fetchMock).toHaveBeenCalledOnce()
  })

  it('rejects plaintext non-loopback endpoints', () => {
    expect(() => new Client('http://example.com')).toThrow(/Plain remote/)
  })

  it('returns a complete value above the former 64 MiB event cap', async () => {
    const length = 64 * 1024 * 1024 + 1
    const encoder = new TextEncoder()
    vi.stubGlobal('fetch', async () => new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(JSON.stringify({ type: 'batch', row_count: 1,
          columns: [{ values: [{ type: 'string', value: 'x'.repeat(length) }] }] })))
        controller.enqueue(encoder.encode('\n{"type":"summary","truncated":false}\n'))
        controller.close()
      },
    })))
    vi.stubGlobal('crypto', { randomUUID: () => '00000000-0000-4000-8000-000000000000' })
    const result = await new Client('http://localhost:18484').query({ cypher: 'RETURN $body' })
    expect(String(result.rows[0][0].value).length).toBe(length)
    expect(result.summary.truncated).toBe(false)
  })

  it('decodes fragmented UTF-8, blank lines and an unterminated final event', async () => {
    const value = '東京 ✈ café\n'.repeat(32768)
    const bytes = new TextEncoder().encode('\r\n' + JSON.stringify({ type: 'batch', row_count: 1,
      columns: [{ values: [{ type: 'string', value }] }] }) + '\n\n{"type":"summary","truncated":false}')
    let offset = 0
    vi.stubGlobal('fetch', async () => new Response(new ReadableStream({
      pull(controller) {
        if (offset === bytes.length) { controller.close(); return }
        const end = Math.min(bytes.length, offset + 257)
        controller.enqueue(bytes.subarray(offset, end))
        offset = end
      },
    })))
    vi.stubGlobal('crypto', { randomUUID: () => '00000000-0000-4000-8000-000000000000' })
    const result = await new Client('http://localhost:18484').query({ cypher: 'RETURN $body' })
    expect(result.rows).toHaveLength(1)
    expect(result.rows[0][0].value).toBe(value)
    expect(result.summary.truncated).toBe(false)
  })

  it.each([
    'MATCH (d:Document) RETURN d.body',
    'CREATE (:Document {body: $body})',
    'MATCH (d:Document) SET d.body = $body',
    'MATCH (d:Document) DELETE d',
    'MATCH (d:Document) SEARCH d IN (EMBEDDING INDEX documents FOR TEXT $body LIMIT 5) SCORE AS score RETURN d, score',
    'CALL graph.wcc() YIELD node, component RETURN node, component',
    'SHOW TOPICS',
    'SHOW QUEUES',
    'SHOW EXCHANGES',
    'SHOW INDEXES',
  ])('transports database capabilities through the canonical endpoint: %s', async (cypher) => {
    const signal = new AbortController().signal
    const fetchMock = vi.fn(async (url: URL, init: RequestInit) => {
      expect(url.pathname).toBe('/api/query')
      expect(init.signal).toBe(signal)
      expect(JSON.parse(String(init.body))).toMatchObject({
        query: cypher,
        project_id: '00000000-0000-4000-8000-000000000001',
        parameters: { body: 'Complete document text' },
      })
      return new Response('{"type":"schema","columns":[]}\n{"type":"summary"}\n', { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('crypto', { randomUUID: () => '00000000-0000-4000-8000-000000000000' })
    await new Client('http://localhost:18484').query({
      cypher, projectId: '00000000-0000-4000-8000-000000000001',
      parameters: { body: 'Complete document text' }, signal,
    })
    expect(fetchMock).toHaveBeenCalledOnce()
  })
})
