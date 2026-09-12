export type ThemePreference = 'light' | 'dark' | 'system';

export interface Project {
  id: string;
  name: string;
}

export interface LocalAiIntegration {
  host: string;
  display_name: string;
  integration: string;
  detected: boolean;
  state: 'not-installed' | 'current' | 'activation-required' | 'update-available' | 'repair-required' | 'newer-than-runtime';
  installed_version?: string;
  available_version: string;
  activation_required: boolean;
  activation_instruction?: string;
  install_command: string;
  update_command: string;
}

export interface QueryHistoryEntry {
  id: string;
  projectId: string;
  query: string;
  createdAt: number;
}

export interface ResultColumn {
  name: string;
  valueType?: string;
}

export interface GraphNode {
  id: string;
  labels: string[];
  properties: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relationshipType: string;
  properties: Record<string, unknown>;
}

export interface QueryStatistics {
  elapsed_ms?: number;
  elapsed_us?: number;
  rows?: number;
  nodes?: number;
  edges?: number;
  updates?: number;
  [key: string]: unknown;
}

export interface QueryResult {
  columns: ResultColumn[];
  rows: unknown[][];
  nodes: GraphNode[];
  edges: GraphEdge[];
  bookmark?: string;
  statistics?: QueryStatistics;
  truncated: boolean;
  truncationReason?: string;
}

export type QueryStreamEvent =
  | { type: 'catalog'; catalog: CompletionSchema }
  | { type: 'schema'; columns: ResultColumn[] }
  | { type: 'batch'; rows?: unknown[][]; columns?: Array<ResultColumn & { values: unknown[] }> }
  | { type: 'graph'; nodes?: GraphNode[]; edges?: GraphEdge[] }
  | {
      type: 'summary';
      bookmark?: string;
      statistics?: QueryStatistics;
      truncated?: boolean;
      truncationReason?: string;
    }
  | { type: 'error'; code: string; message: string };

export interface CompletionSchema {
  labels: string[];
  relationshipTypes: string[];
  properties: string[];
  functions: string[];
}
