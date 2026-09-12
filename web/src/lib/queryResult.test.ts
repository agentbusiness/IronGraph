import { describe, expect, it } from 'vitest';
import type { QueryStreamEvent } from '../types';
import { layoutTierFor } from './graphLayout';
import { EMPTY_RESULT, applyQueryEvent } from './queryResult';

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
    const result = applyQueryEvent({ ...EMPTY_RESULT }, event);
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
    const result = applyQueryEvent({ ...EMPTY_RESULT }, { type: 'graph', nodes, edges: [] });
    expect(result.nodes).toHaveLength(57_344);
    expect(result.truncated).toBe(false);
    expect(layoutTierFor(57_344)).toBe('staged');
  });
});
