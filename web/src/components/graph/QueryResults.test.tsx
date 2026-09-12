import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { QueryResult } from '../../types';
import { DEFAULT_VIEW_STATE } from '../../lib/graphView';
import { QueryResults } from './QueryResults';

const result: QueryResult = {
  columns: [],
  rows: [[1], [2]],
  nodes: [],
  edges: [],
  statistics: { elapsed_ms: 17, rows: 2, nodes: 0, edges: 0, updates: 0 },
  truncated: false,
};

describe('query result summary', () => {
  it.each(['3d', '2d', 'table'] as const)('shows server execution time in the %s result bar', (mode) => {
    const { unmount } = render(
      <QueryResults
        result={result}
        running={false}
        mode={mode}
        onModeChange={vi.fn()}
        view={DEFAULT_VIEW_STATE}
        onViewChange={vi.fn()}
        onSelectionChange={vi.fn()}
        onExpandNode={vi.fn()}
      />,
    );

    expect(screen.getByRole('button', { name: '2 rows · 0 nodes · 0 edges · 17 ms' })).toBeInTheDocument();
    unmount();
  });
});
