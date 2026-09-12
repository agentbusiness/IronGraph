import { render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { MarkdownDocument } from './MarkdownDocument';

describe('MarkdownDocument', () => {
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
