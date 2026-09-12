import { describe, expect, it } from 'vitest';
import { DOC_PAGES, resolveDocLink } from './docsCatalog';

describe('bundled documentation catalog', () => {
  it('indexes the public guide and complete Cypher reference', () => {
    expect(DOC_PAGES.some((page) => page.id === 'getting-started')).toBe(true);
    expect(DOC_PAGES.some((page) => page.id === 'cypher/datasets')).toBe(true);
    expect(DOC_PAGES.length).toBeGreaterThan(50);
  });

  it('puts the beginner path before reference material', () => {
    expect(DOC_PAGES.slice(0, 3).map((page) => page.id)).toEqual(['getting-started', 'installation', 'README']);
    expect(DOC_PAGES.findIndex((page) => page.section === 'Cypher')).toBeGreaterThan(2);
    expect(DOC_PAGES.findIndex((page) => page.section === 'Cypher · clauses'))
      .toBeLessThan(DOC_PAGES.findIndex((page) => page.section === 'Cypher · functions'));
  });

  it('resolves relative Markdown links inside the reader', () => {
    expect(resolveDocLink('cypher/functions/aggregation/avg', '../../datasets.md#trust')).toBe('cypher/datasets');
    expect(resolveDocLink('cypher/procedures/community', './graph-wcc.md')).toBe('cypher/procedures/community/graph-wcc');
    expect(resolveDocLink('getting-started', 'installation.md')).toBe('installation');
  });
});
