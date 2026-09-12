import type { Project } from '../types';
import { executeQuery, rowObjects } from './simpleQuery';

export interface GraphDocument {
  documentId: string;
  title: string;
  body: string;
  createdAt: string;
  updatedAt: string;
}

export const DOCUMENT_LIST_QUERY = `MATCH (document:Document)
RETURN document.document_id AS document_id,
       document.title AS title,
       document.body AS body,
       document.created_at AS created_at,
       document.updated_at AS updated_at
ORDER BY document.updated_at DESC, document.title`;

export const DOCUMENT_CREATE_QUERY = `CREATE (:Document {
  document_id: $document_id,
  title: $title,
  body: $body,
  created_at: $created_at,
  updated_at: $updated_at
})`;

export const DOCUMENT_SAVE_QUERY = `MATCH (document:Document {document_id: $document_id})
SET document.title = $title,
    document.body = $body,
    document.updated_at = $updated_at`;

export const DOCUMENT_DELETE_QUERY = `MATCH (document:Document {document_id: $document_id})
DETACH DELETE document`;

function text(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

export async function listDocuments(project: Project, signal?: AbortSignal): Promise<GraphDocument[]> {
  const result = await executeQuery(DOCUMENT_LIST_QUERY, project.id, {}, signal);
  return rowObjects(result).map((row) => ({
    documentId: text(row.document_id),
    title: text(row.title) || 'Untitled document',
    body: text(row.body),
    createdAt: text(row.created_at),
    updatedAt: text(row.updated_at),
  })).filter((document) => document.documentId.length > 0);
}

export async function createDocument(project: Project): Promise<GraphDocument> {
  const now = new Date().toISOString();
  const document: GraphDocument = {
    documentId: crypto.randomUUID(),
    title: 'Untitled document',
    body: '',
    createdAt: now,
    updatedAt: now,
  };
  await executeQuery(DOCUMENT_CREATE_QUERY, project.id, {
    document_id: document.documentId,
    title: document.title,
    body: document.body,
    created_at: now,
    updated_at: now,
  });
  return document;
}

export async function saveDocument(project: Project, document: GraphDocument): Promise<GraphDocument> {
  const saved = { ...document, updatedAt: new Date().toISOString() };
  await executeQuery(DOCUMENT_SAVE_QUERY, project.id, {
    document_id: saved.documentId,
    title: saved.title,
    body: saved.body,
    updated_at: saved.updatedAt,
  });
  return saved;
}

export async function deleteDocument(project: Project, documentId: string): Promise<void> {
  await executeQuery(DOCUMENT_DELETE_QUERY, project.id, { document_id: documentId });
}
