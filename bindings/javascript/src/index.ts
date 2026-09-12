export type TypedValue =
  | { type: 'null' }
  | { type: 'boolean'; value: boolean }
  | { type: 'integer'; value: string }
  | { type: 'float'; value: number }
  | { type: 'string'; value: string }
  | { type: string; value?: unknown }

export interface QueryRequest {
  cypher: string
  projectId?: string
  parameters?: Record<string, unknown>
  bookmark?: { term: number; index: number }
  limits?: Partial<QueryLimits>
  signal?: AbortSignal
}

export interface QueryLimits {
  rows: number
  bytes: number
  nodes: number
  edges: number
}

export interface QueryColumn {
  name: string
  value_type: string
  nullable: boolean
}

export interface QueryResult {
  catalog?: Record<string, unknown>
  columns: QueryColumn[]
  rows: TypedValue[][]
  summary: Record<string, unknown>
}

interface StreamEvent {
  type: string
  [key: string]: unknown
}

const MAXIMUM_EVENT_CHARACTERS = 64 * 1024 * 1024

export class IronGraphError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly retryable = false,
    readonly retryAfterMs?: number,
  ) {
    super(message)
    this.name = 'IronGraphError'
  }
}

export class Client {
  private readonly endpoint: URL

  constructor(baseUrl: string | URL) {
    this.endpoint = new URL('/api/query', baseUrl)
    const local = this.endpoint.hostname === 'localhost'
      || this.endpoint.hostname === '127.0.0.1'
      || this.endpoint.hostname === '[::1]'
    if (this.endpoint.protocol !== 'https:' && !(this.endpoint.protocol === 'http:' && local)) {
      throw new IronGraphError(
        'Plain remote connections are forbidden; use HTTPS with a browser-managed client certificate',
        'ConfigurationError',
      )
    }
  }

  async query(request: QueryRequest): Promise<QueryResult> {
    if (!request.cypher.trim()) throw new IronGraphError('Cypher is empty', 'ConfigurationError')
    const response = await fetch(this.endpoint, {
      method: 'POST',
      headers: { Accept: 'application/x-ndjson', 'Content-Type': 'application/json' },
      body: JSON.stringify({
        request_id: crypto.randomUUID(),
        project_id: request.projectId ?? null,
        query: request.cypher,
        parameters: request.parameters ?? {},
        bookmark: request.bookmark ?? null,
        ...(request.limits ? { limits: request.limits } : {}),
      }),
      signal: request.signal,
    })
    if (!response.ok) throw new IronGraphError(`HTTP ${response.status}`, 'HttpError')
    if (!response.body) throw new IronGraphError('Query response has no body', 'ProtocolError')

    const result: QueryResult = { columns: [], rows: [], summary: {} }
    for await (const event of decodeNdjson(response.body)) {
      if (event.type === 'error') {
        throw new IronGraphError(
          String(event.message ?? 'Query failed'),
          String(event.code ?? 'QueryError'),
          Boolean(event.retryable),
          typeof event.retry_after_ms === 'number' ? event.retry_after_ms : undefined,
        )
      }
      if (event.type === 'catalog') {
        const { type: _type, ...catalog } = event
        result.catalog = catalog
      }
      if (event.type === 'schema') result.columns = (event.columns ?? []) as QueryColumn[]
      if (event.type === 'batch') appendBatch(result, event)
      if (event.type === 'summary') {
        const { type: _type, ...summary } = event
        result.summary = summary
      }
    }
    return result
  }
}

function appendBatch(result: QueryResult, event: StreamEvent): void {
  const columns = Array.isArray(event.columns) ? event.columns as Array<{ values?: TypedValue[] }> : []
  const rowCount = Number(event.row_count ?? 0)
  if (!Number.isSafeInteger(rowCount) || rowCount < 0) {
    throw new IronGraphError('Invalid batch row count', 'ProtocolError')
  }
  if (columns.some((column) => !Array.isArray(column.values) || column.values.length !== rowCount)) {
    throw new IronGraphError('Misaligned batch columns', 'ProtocolError')
  }
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    result.rows.push(columns.map((column) => column.values![rowIndex]))
  }
}

async function* decodeNdjson(stream: ReadableStream<Uint8Array>): AsyncGenerator<StreamEvent> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffered = ''
  try {
    while (true) {
      const { value, done } = await reader.read()
      buffered += decoder.decode(value, { stream: !done })
      if (buffered.length > MAXIMUM_EVENT_CHARACTERS && !buffered.includes('\n')) {
        throw new IronGraphError('One query event exceeds 64 MiB', 'ProtocolError')
      }
      const lines = buffered.split('\n')
      buffered = lines.pop() ?? ''
      for (const line of lines) {
        if (line.trim()) yield JSON.parse(line) as StreamEvent
      }
      if (done) break
    }
    if (buffered.trim()) yield JSON.parse(buffered) as StreamEvent
  } finally {
    reader.releaseLock()
  }
}
