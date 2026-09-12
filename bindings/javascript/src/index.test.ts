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
