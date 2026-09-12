import 'fake-indexeddb/auto';
import { beforeEach, describe, expect, it } from 'vitest';
import type { QueryHistoryEntry } from '../types';
import { addQueryHistory, clearQueryHistory, deleteQueryHistory, listQueryHistory } from './db';
import { historyDays } from './queryHistory';

const DAY = 86_400_000;
const NOW = new Date('2026-08-23T11:00:00').getTime();

function entry(id: string, query: string, createdAt: number): QueryHistoryEntry {
  return { id, projectId: 'p', query, createdAt };
}

describe('history, cut into days', () => {
  it('names today and yesterday, and dates everything older', () => {
    const days = historyDays([
      entry('1', 'a', NOW - 3_600_000),
      entry('2', 'b', NOW - 7_200_000),
      entry('3', 'c', NOW - DAY),
      entry('4', 'd', NOW - DAY * 6),
    ], NOW);

    expect(days.map((day) => day.label)).toEqual(['Today', 'Yesterday', expect.stringMatching(/\w/)]);
    expect(days[0]?.entries).toHaveLength(2);
    expect(days[2]?.label).not.toBe('Yesterday');
  });

  it('groups by calendar day, so late last night is not this morning', () => {
    const lastNight = new Date('2026-08-22T23:50:00').getTime();
    const thisMorning = new Date('2026-08-23T00:10:00').getTime();

    const days = historyDays([entry('1', 'a', thisMorning), entry('2', 'b', lastNight)], NOW);

    expect(days).toHaveLength(2);
    expect(days[0]?.label).toBe('Today');
    expect(days[1]?.label).toBe('Yesterday');
  });

  it('holds an empty history as no days at all', () => {
    expect(historyDays([], NOW)).toEqual([]);
  });
});

describe('what the history rail keeps', () => {
  beforeEach(async () => {
    await clearQueryHistory('one');
    await clearQueryHistory('two');
  });

  it('records a repeated statement once, moved to the top', async () => {
    await addQueryHistory('one', 'MATCH (n) RETURN n');
    await addQueryHistory('one', 'MATCH (m) RETURN m');
    await addQueryHistory('one', 'MATCH (n) RETURN n');

    const kept = await listQueryHistory('one');
    expect(kept.map(({ query }) => query)).toEqual(['MATCH (n) RETURN n', 'MATCH (m) RETURN m']);
  });

  it('keeps one project out of another project’s rail', async () => {
    await addQueryHistory('one', 'MATCH (a) RETURN a');
    await addQueryHistory('two', 'MATCH (b) RETURN b');

    expect((await listQueryHistory('one')).map(({ query }) => query)).toEqual(['MATCH (a) RETURN a']);
    expect((await listQueryHistory('two')).map(({ query }) => query)).toEqual(['MATCH (b) RETURN b']);
  });

  it('forgets one statement, and clears only the project asked for', async () => {
    const kept = await addQueryHistory('one', 'MATCH (a) RETURN a');
    await addQueryHistory('one', 'MATCH (c) RETURN c');
    await addQueryHistory('two', 'MATCH (b) RETURN b');

    await deleteQueryHistory(kept.id);
    expect((await listQueryHistory('one')).map(({ query }) => query)).toEqual(['MATCH (c) RETURN c']);

    await clearQueryHistory('one');
    expect(await listQueryHistory('one')).toEqual([]);
    expect(await listQueryHistory('two')).toHaveLength(1);
  });
});
