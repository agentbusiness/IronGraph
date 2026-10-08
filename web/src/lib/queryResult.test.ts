import { describe, expect, it } from 'vitest';
import type { QueryStreamEvent } from '../types';
import { layoutTierFor } from './graphLayout';
import { QueryResultCollector } from './queryResult';

describe('query result normalization', () => {
  it('extracts the complete graph from typed column values', () => {
    const event: QueryStreamEvent = {
      type: 'batch',
      columns: [{
        name: 'path',
        values: [{
          __kind: 'path',
          nodes: [
            { __kind: 'node', id: '1', labels: ['Person'], properties: { name: 'Ada' } },
            { __kind: 'node', id: '2', labels: ['Person'], properties: { name: 'Lin' } },
          ],
          relationships: [{ __kind: 'relationship', id: '3', source: '1', target: '2', relationshipType: 'KNOWS', properties: {} }],
        }],
      }],
    };
    const collected = new QueryResultCollector();
    collected.append(event);
    const result = collected.finish();
    expect(result.nodes.map(({ id }) => id)).toEqual(['1', '2']);
    expect(result.edges).toHaveLength(1);
    expect(result.rows).toHaveLength(1);
  });

  it('keeps results larger than the former 50,000-node browser budget', () => {
    const nodes = Array.from({ length: 57_344 }, (_, index) => ({
      id: String(index),
      labels: ['Account'],
      properties: {},
    }));
    const collected = new QueryResultCollector();
    collected.append({ type: 'graph', nodes, edges: [] });
    const result = collected.finish();
    expect(result.nodes).toHaveLength(57_344);
    expect(result.truncated).toBe(false);
    expect(layoutTierFor(57_344)).toBe('staged');
  });

  it('collects many dirty batches once and keeps complete rows and entity ownership', () => {
    const properties = { body: 'dirty\u0000"漢字'.repeat(512), vector: { __kind: 'vector', value: Array.from({ length: 384 }, (_, i) => i / 7) } };
    const collected = new QueryResultCollector();
    collected.append({ type: 'schema', columns: [{ name: 'source' }, { name: 'relationship' }] });
    const rows: unknown[][] = [];
    for (let batch = 0; batch < 256; batch++) {
      const incoming = Array.from({ length: 256 }, (_, i) => {
        const id = String(batch * 256 + i);
        return [
          { __kind: 'node', id: String(i % 8), labels: ['Airport'], properties },
          { __kind: 'relationship', id, source: String(i % 8), target: String((i + 1) % 8), relationshipType: 'ROUTE', properties },
        ];
      });
      rows.push(...incoming);
      collected.append({ type: 'batch', rows: incoming });
    }
    collected.append({ type: 'summary', bookmark: '1:123', statistics: { rows: 65_536 }, truncated: false });
    const result = collected.finish();
    expect(result.rows).toHaveLength(65_536);
    expect(result.edges).toHaveLength(65_536);
    expect(result.nodes).toHaveLength(8);
    expect(result.rows[0]).toBe(rows[0]);
    expect(result.rows.at(-1)).toBe(rows.at(-1));
    expect(result.nodes[0]?.properties).toBe(properties);
    expect(result.edges.at(-1)?.properties).toBe(properties);
    expect(result.bookmark).toBe('1:123');
    expect(result.truncated).toBe(false);
  });
});
