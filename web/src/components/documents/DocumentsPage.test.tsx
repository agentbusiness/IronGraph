import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createDocument, listDocuments } from '../../lib/documents';
import type * as DocumentsModule from '../../lib/documents';
import { DocumentsPage } from './DocumentsPage';

vi.mock('../../lib/documents', async (importOriginal) => {
  const original = await importOriginal<typeof DocumentsModule>();
  return {
    ...original,
    listDocuments: vi.fn(),
    createDocument: vi.fn(),
    saveDocument: vi.fn(),
    deleteDocument: vi.fn(),
  };
});

describe('DocumentsPage', () => {
  beforeEach(() => {
    vi.mocked(listDocuments).mockResolvedValue([]);
    vi.mocked(createDocument).mockResolvedValue({
      documentId: 'doc-1',
      title: 'Untitled document',
      body: '',
      createdAt: '2026-08-31T00:00:00Z',
      updatedAt: '2026-08-31T00:00:00Z',
    });
  });

  it('creates immediately and opens the rich editor without exposing Cypher', async () => {
    render(<DocumentsPage project={{ id: 'project-1', name: 'notes' }} />);
    fireEvent.click(screen.getByRole('button', { name: 'New' }));

    await waitFor(() => expect(createDocument).toHaveBeenCalledOnce());
    expect(screen.getByRole('toolbar', { name: 'Formatting' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Bold' })).toBeVisible();
    expect(screen.queryByText(/statement to run|run create|detach delete|CREATE \(:Document/i)).not.toBeInTheDocument();
  });
});
