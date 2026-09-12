export interface Bookmark {
  term: number
  index: number
}

export type TypedValue =
  | { type: 'null' }
  | { type: 'boolean'; value: boolean }
  | { type: 'integer'; value: string }
  | { type: 'float'; value: number }
  | { type: 'string'; value: string }
  | { type: 'bytes'; value: number[] }
  | { type: 'date'; value: number }
  | { type: 'time'; value: { nanos: number; offset_seconds?: number } }
  | { type: 'date_time'; value: { seconds: number; nanos: number; timezone?: string } }
  | { type: 'duration'; value: { months: number; days: number; seconds: number; nanos: number } }
  | { type: 'vector'; value: number[] }
  | { type: 'node'; value: Record<string, unknown> }
  | { type: 'relationship'; value: Record<string, unknown> }
  | { type: 'path'; value: Record<string, unknown> }
  | { type: 'list'; value: TypedValue[] }
  | { type: 'map'; value: Record<string, TypedValue> }

export interface QueryResult {
  catalog?: Record<string, unknown> | null
  columns: Array<{ name: string; value_type: string; nullable: boolean }>
  rows: TypedValue[][]
  summary: {
    bookmark?: Bookmark | null
    statistics: {
      elapsed_ms: number
      elapsed_us: number
      rows: number
      nodes: number
      edges: number
      updates: number
    }
    truncated: boolean
    truncation_reason?: string | null
  }
}

export declare class Client {
  static api(baseUrl: string): Client
  static bolt(uri: string): Client
  static apiMtls(baseUrl: string, certificate: string, privateKey: string, certificateAuthority: string): Client
  static boltMtls(uri: string, certificate: string, privateKey: string, certificateAuthority: string): Client
  query(cypher: string, projectId?: string | null, parameters?: Record<string, unknown> | null): Promise<QueryResult>
}

export declare class EmbeddedDatabase {
  static open(dataDir: string, device?: 'auto' | 'cpu' | 'metal' | 'cuda' | null, deviceOrdinal?: number | null, loadModels?: boolean | null): Promise<EmbeddedDatabase>
  query(cypher: string, projectId?: string | null, parameters?: Record<string, unknown> | null): Promise<QueryResult>
  snapshot(): Promise<Bookmark>
  close(): Promise<void>
}
