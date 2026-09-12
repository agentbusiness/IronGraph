import { describe, expect, it } from 'vitest';
import { DOCUMENT_CREATE_QUERY, DOCUMENT_DELETE_QUERY, DOCUMENT_LIST_QUERY, DOCUMENT_SAVE_QUERY } from './documents';

describe('Cypher-native documents', () => {
  it('keeps complete source text on an ordinary Document node', () => {
    expect(DOCUMENT_CREATE_QUERY).toContain('CREATE (:Document');
    expect(DOCUMENT_CREATE_QUERY).toContain('body: $body');
    expect(DOCUMENT_LIST_QUERY).toContain('document.body AS body');
  });

  it('uses parameterized mutations and explicit detach deletion', () => {
    expect(DOCUMENT_SAVE_QUERY).toContain('{document_id: $document_id}');
    expect(DOCUMENT_SAVE_QUERY).toContain('document.body = $body');
    expect(DOCUMENT_DELETE_QUERY).toContain('DETACH DELETE document');
  });
});
