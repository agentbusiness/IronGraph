import type { QueryHistoryEntry } from '../types';
import { dayLabel } from './format';

export interface HistoryDay {
  key: string;
  label: string;
  entries: QueryHistoryEntry[];
}

function dayKey(at: number): string {
  return new Date(at).toLocaleDateString(undefined, { year: 'numeric', month: '2-digit', day: '2-digit' });
}

/**
 * History, cut into the days it happened on.
 *
 * Fifty statements in one column read as one undifferentiated stack; the day a question was asked
 * on is the coarsest thing that distinguishes them. Grouping is by local calendar day rather than
 * by elapsed hours, so "today" means the
 * reader's today and a query from 23:50 last night is not filed under this morning.
 *
 * `now` is passed in rather than read here: a function that decides what "today" means from the
 * clock cannot be tested, and the caller already re-renders when the day it cares about changes.
 */
export function historyDays(entries: QueryHistoryEntry[], now: number): HistoryDay[] {
  const today = dayKey(now);
  const yesterday = dayKey(now - 86_400_000);
  const days: HistoryDay[] = [];
  entries.forEach((entry) => {
    const key = dayKey(entry.createdAt);
    const existing = days.find((day) => day.key === key);
    if (existing) {
      existing.entries.push(entry);
      return;
    }
    const label = key === today
      ? 'Today'
      : key === yesterday
        ? 'Yesterday'
        // Anything older is dated by the shared rule, which adds the year only when it is not
        // this one — the same heading the rest of the app writes.
        : dayLabel(entry.createdAt, now);
    days.push({ key, label, entries: [entry] });
  });
  return days;
}
