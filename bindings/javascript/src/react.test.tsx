import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Client } from './index.js'
import { IronGraphProvider, useIronGraphClient } from './react.js'

afterEach(() => vi.unstubAllGlobals())

function Consumer() {
  const client = useIronGraphClient()
  return createElement('span', null, client.constructor.name)
}

describe('IronGraphProvider', () => {
  it('provides one typed API client to React consumers', () => {
    const markup = renderToStaticMarkup(
      createElement(
        IronGraphProvider,
        { baseUrl: 'http://localhost:18484' },
        createElement(Consumer),
      ),
    )
    expect(markup).toBe('<span>Client</span>')
  })

  it('lets React consumers search documents through the same query client', async () => {
    let client: Client | undefined
    function SearchConsumer() {
      client = useIronGraphClient()
      return null
    }
    renderToStaticMarkup(createElement(IronGraphProvider, { baseUrl: 'http://localhost:18484' },
      createElement(SearchConsumer)))
    const fetchMock = vi.fn(async (url: URL, init: RequestInit) => {
      expect(url.pathname).toBe('/api/query')
      expect(JSON.parse(String(init.body))).toMatchObject({
        query: expect.stringContaining('SEARCH d IN'), parameters: { text: 'connected documents' },
      })
      return new Response([
        '{"type":"schema","columns":[{"name":"body","value_type":"STRING","nullable":false}]}',
        '{"type":"batch","row_count":1,"columns":[{"values":[{"type":"string","value":"Complete document body"}]}]}',
        '{"type":"summary"}',
      ].join('\n'), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('crypto', { randomUUID: () => '00000000-0000-4000-8000-000000000000' })
    expect(client).toBeDefined()
    const result = await client!.query({
      cypher: 'USE app MATCH (d:Document) SEARCH d IN (EMBEDDING INDEX documents FOR TEXT $text LIMIT 1) SCORE AS score RETURN d.body AS body',
      parameters: { text: 'connected documents' },
    })
    expect(result.rows).toEqual([[{ type: 'string', value: 'Complete document body' }]])
    expect(fetchMock).toHaveBeenCalledOnce()
  })
})
