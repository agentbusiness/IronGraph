import { deleteDB, openDB, type DBSchema, type IDBPDatabase } from 'idb';
import type { QueryHistoryEntry } from '../types';
import { MAX_QUERY_HISTORY } from './bounds';

interface BrowserDatabase extends DBSchema {
  queryHistory: {
    key: string;
    value: QueryHistoryEntry;
    indexes: { 'by-project-created': [string, number] };
  };
}

let databasePromise: Promise<IDBPDatabase<BrowserDatabase>> | undefined;

function database(): Promise<IDBPDatabase<BrowserDatabase>> {
  databasePromise ??= openDB<BrowserDatabase>('irongraph-browser', 1, {
    upgrade(db) {
      const history = db.createObjectStore('queryHistory', { keyPath: 'id' });
      history.createIndex('by-project-created', ['projectId', 'createdAt']);
    },
    blocked() { databasePromise = undefined; },
    blocking() { databasePromise = undefined; },
    terminated() { databasePromise = undefined; },
  });
  return databasePromise;
}

export async function verifyBrowserPersistence(): Promise<void> {
  await database();
  try {
    await navigator.storage?.persist?.();
  } catch {
    // Query history remains usable with best-effort browser persistence.
  }
}

export async function listQueryHistory(projectId: string): Promise<QueryHistoryEntry[]> {
  const db = await database();
  const range = IDBKeyRange.bound([projectId, 0], [projectId, Number.MAX_SAFE_INTEGER]);
  const entries = await db.getAllFromIndex('queryHistory', 'by-project-created', range);
  return entries.sort((a, b) => b.createdAt - a.createdAt).slice(0, MAX_QUERY_HISTORY);
}

export async function addQueryHistory(projectId: string, query: string): Promise<QueryHistoryEntry> {
  const db = await database();
  const transaction = db.transaction('queryHistory', 'readwrite');
  const existing = await transaction.store.getAll();
  const repeat = existing.find((entry) => entry.projectId === projectId && entry.query === query);
  const entry: QueryHistoryEntry = {
    id: repeat?.id ?? crypto.randomUUID(),
    projectId,
    query,
    createdAt: Math.max(Date.now(), existing.reduce((latest, item) => Math.max(latest, item.createdAt + 1), 0)),
  };
  await transaction.store.put(entry);
  const mine = [...existing.filter((item) => item.projectId === projectId && item.id !== entry.id), entry];
  if (mine.length > MAX_QUERY_HISTORY) {
    const removable = mine.sort((a, b) => b.createdAt - a.createdAt).slice(MAX_QUERY_HISTORY);
    await Promise.all(removable.map(({ id }) => transaction.store.delete(id)));
  }
  await transaction.done;
  return entry;
}

export async function deleteQueryHistory(id: string): Promise<void> {
  await (await database()).delete('queryHistory', id);
}

export async function clearQueryHistory(projectId: string): Promise<void> {
  const db = await database();
  const transaction = db.transaction('queryHistory', 'readwrite');
  const entries = await transaction.store.getAll();
  await Promise.all(entries.filter((entry) => entry.projectId === projectId).map(({ id }) => transaction.store.delete(id)));
  await transaction.done;
}

export async function resetDatabaseForTests(): Promise<void> {
  const existing = databasePromise;
  databasePromise = undefined;
  if (existing) (await existing).close();
  await deleteDB('irongraph-browser');
}
