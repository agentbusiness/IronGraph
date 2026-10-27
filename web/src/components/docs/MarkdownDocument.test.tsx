import { cleanup, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { MarkdownDocument } from './MarkdownDocument';

describe('MarkdownDocument', () => {
  afterEach(cleanup);
  it('keeps escaped signature pipes in one cell and quoted type names in one code span', () => {
    const body = [
      '| Field | Value |',
      '| --- | --- |',
      '| Signature | `RELATIONSHIPS ALL\\|NONE\\|ALL EXCEPT <type>` |',
      '',
      'Quote a type as `` `GENERATED FROM` ``.',
    ].join('\n');
    render(<MarkdownDocument body={body} docId="scope" projects={[]} onProjectsChanged={vi.fn(() => Promise.resolve())} onNavigate={vi.fn()} />);
    const row = screen.getByRole('row', { name: 'Signature RELATIONSHIPS ALL|NONE|ALL EXCEPT <type>' });
    expect(within(row).getAllByRole('cell')).toHaveLength(2);
    expect(screen.getByText('`GENERATED FROM`', { selector: 'code' })).toBeVisible();
  });

  it('renders documented pipe output as a real readable table', () => {
    render(<MarkdownDocument body={`# Search result

Result:

\`\`\`
title | score
------+------
Paper one | 0.91
Paper two | 0.82

2 rows
\`\`\``} docId="example" projects={[]} onProjectsChanged={vi.fn(() => Promise.resolve())} onNavigate={vi.fn()} />);

    const table = screen.getByRole('table');
    expect(within(table).getByRole('columnheader', { name: 'title' })).toBeVisible();
    expect(within(table).getByRole('cell', { name: 'Paper one' })).toBeVisible();
    expect(screen.getByText('2 rows')).toBeVisible();
  });
});
