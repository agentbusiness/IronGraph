import { describe, expect, it } from 'vitest';
import { DEFAULT_GRAPH_QUERY, isProjectCatalogQuery } from '../../lib/projectQuery';

describe('default graph query', () => {
  it('returns both endpoints and their relationship as a multiline result', () => {
    expect(DEFAULT_GRAPH_QUERY.split('\n')).toHaveLength(3);
    expect(DEFAULT_GRAPH_QUERY).toContain('(source)-[relationship]->(target)');
    expect(DEFAULT_GRAPH_QUERY).toContain('RETURN source, relationship, target');
  });
});

describe('project-independent query admission', () => {
  it('allows only project catalog statements before a project exists', () => {
    expect(isProjectCatalogQuery('CREATE PROJECT analytics')).toBe(true);
    expect(isProjectCatalogQuery('  SHOW PROJECTS')).toBe(true);
    expect(isProjectCatalogQuery('ALTER PROJECT analytics RENAME TO archive')).toBe(true);
    expect(isProjectCatalogQuery('DROP PROJECT analytics CASCADE')).toBe(true);
    expect(isProjectCatalogQuery('MATCH (n) RETURN n')).toBe(false);
    expect(isProjectCatalogQuery('CREATE (n)')).toBe(false);
  });
});
