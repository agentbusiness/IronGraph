import { describe, expect, it } from 'vitest';
import type { GraphEdge } from '../types';
import { adjacentTypes, degreeInResult, presentProperties } from './graphSelection';
import { neighbourhoodQuery, relationshipQuery } from './projectQuery';
import { EMPTY_RESULT, mergeGraphValues } from './queryResult';

const edges: GraphEdge[] = [
  { id: 'e1', source: 'a', target: 'b', relationshipType: 'SENT', properties: {} },
  { id: 'e2', source: 'a', target: 'c', relationshipType: 'SENT', properties: {} },
  { id: 'e3', source: 'd', target: 'a', relationshipType: 'IN_THREAD', properties: {} },
  { id: 'e4', source: 'b', target: 'c', relationshipType: 'SENT', properties: {} },
];

describe('what a selected node is attached to', () => {
  it('counts each relationship type by direction, busiest first', () => {
    const adjacent = adjacentTypes(edges, 'a');

    expect(adjacent).toEqual([
      { relationshipType: 'SENT', out: 2, in: 0 },
      { relationshipType: 'IN_THREAD', out: 0, in: 1 },
    ]);
    expect(degreeInResult(adjacent)).toBe(3);
  });

  it('counts only what the result on screen holds', () => {
    expect(adjacentTypes([], 'a')).toEqual([]);
  });
});

describe('which properties are worth reading', () => {
  it('drops engine bookkeeping and properties the engine answered as null', () => {
    const present = presentProperties({
      name: 'Ada',
      count: 0,
      active: false,
      nickname: null,
      legacyNull: { type: 'null' },
      __irongraph_layer: 'OBSERVED',
    });

    expect(present).toEqual([['name', 'Ada'], ['count', 0], ['active', false]]);
  });
});

describe('the traversal a double-click runs', () => {
  it('seeks by engine identity and returns both endpoints with the relationship', () => {
    const query = neighbourhoodQuery('12');

    expect(query).toContain('WHERE id(source) = 12');
    expect(query).toContain('(source)-[relationship]-(target)');
    expect(query).toContain('RETURN source, relationship, target');
    expect(query).not.toContain('LIMIT');
  });

  it('refuses an identity the engine cannot seek on rather than interpolating it', () => {
    expect(neighbourhoodQuery('cluster:3')).toBeUndefined();
    expect(neighbourhoodQuery('12 OR true')).toBeUndefined();
    expect(neighbourhoodQuery('')).toBeUndefined();
  });

  it('asks the schema list for a relationship type as a graph, not as bare relationships', () => {
    expect(relationshipQuery('IN_THREAD')).toContain('(source)-[relationship:IN_THREAD]->(target)');
    expect(relationshipQuery('IN_THREAD')).toContain('RETURN source, relationship, target');
  });
});

describe('folding a traversal into the result', () => {
  const node = (id: string) => ({ __kind: 'node', id, labels: ['Person'], properties: {} });
  const relationship = { __kind: 'relationship', id: 'r1', source: '1', target: '2', relationshipType: 'KNOWS', properties: {} };

  it('grows the scene and leaves the rows the query produced alone', () => {
    const before = { ...EMPTY_RESULT, rows: [['only row']] };

    const after = mergeGraphValues(before, [node('1'), relationship, node('2')]);

    expect(after.nodes.map(({ id }) => id)).toEqual(['1', '2']);
    expect(after.edges).toHaveLength(1);
    expect(after.rows).toEqual([['only row']]);
  });

  it('returns the same result when the traversal brought back nothing new', () => {
    const before = mergeGraphValues(EMPTY_RESULT, [node('1'), relationship, node('2')]);

    expect(mergeGraphValues(before, [node('1'), node('2'), relationship])).toBe(before);
  });
});
