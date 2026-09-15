export interface Bookmark {
  term: number
  index: number
}

export interface QueryOptions {
  bookmark?: Bookmark
  consistency?: 'PUBLISHED'
  limits?: { rows?: number; bytes?: number; nodes?: number; edges?: number }
}
export interface OperationOptions { operation_id?: string; timeout_ms?: number }
export interface ResourceBudgets {
  device_memory_limit_bytes?: number
  device_reserved_bytes?: number
  max_write_bytes?: number
  max_concurrent_operations?: number
  worker_threads?: number
  request_timeout_ms?: number
  startup_timeout_ms?: number
  snapshot_interval_ms?: number
}
export interface StreamRecord {
  key?: number[] | null
  headers?: Record<string, number[]>
  value?: number[] | null
  create_time_ms?: number | null
}
export interface StreamAppend { project_id: string; topic: string; partition: number; records: StreamRecord[] }
export interface StreamFetch { project_id: string; topic: string; partition: number; offset: number; max_records: number; max_bytes: number }
export interface StreamAcknowledgement { bookmark: Bookmark; first_offset: number; record_count: number }
export interface StreamPage {
  records: Array<[number, { id: number; resolved_time_ms: number; ingress: Record<string, unknown>; payload: number[]; checksum: number[] }]>
  high_watermark: number
  next_offset: number
  truncated: boolean
}
export interface RuntimeStatus { data_dir: string; ready: boolean; active_operations: number; max_concurrent_operations: number; worker_threads: number }

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
  query(cypher: string, projectId?: string | null, parameters?: Record<string, unknown> | null, options?: QueryOptions | null): Promise<QueryResult>
}

export declare class EmbeddedDatabase {
  static open(dataDir: string, device?: 'auto' | 'cpu' | 'metal' | 'cuda' | null, deviceOrdinal?: number | null, loadModels?: boolean | null, budgets?: ResourceBudgets | null): Promise<EmbeddedDatabase>
  query(cypher: string, projectId?: string | null, parameters?: Record<string, unknown> | null, options?: QueryOptions | null, operation?: OperationOptions | null): Promise<QueryResult>
  streamAppend(request: StreamAppend, options?: OperationOptions | null): Promise<StreamAcknowledgement>
  streamFetch(request: StreamFetch, options?: OperationOptions | null): Promise<StreamPage>
  status(): RuntimeStatus
  cancel(operationId: string): boolean
  snapshot(): Promise<Bookmark>
  flush(): Promise<void>
  close(): Promise<void>
}
