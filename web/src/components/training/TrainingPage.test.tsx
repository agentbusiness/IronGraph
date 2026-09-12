import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { executeQuery } from '../../lib/simpleQuery';
import { TrainingPage } from './TrainingPage';

vi.mock('../../lib/simpleQuery', () => ({ executeQuery: vi.fn() }));

describe('TrainingPage dataset import', () => {
  beforeEach(() => vi.mocked(executeQuery).mockResolvedValue({ columns: [], rows: [], nodes: [], edges: [], truncated: false }));

  it('offers one import button and keeps Python out of the workflow', async () => {
    const refresh = vi.fn().mockResolvedValue(undefined);
    render(<TrainingPage projects={[]} onProjectsChanged={refresh} />);

    expect(screen.queryByText(/python3|copy command/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Import dataset' }));

    await waitFor(() => expect(executeQuery).toHaveBeenCalledWith('IMPORT DATASET flights'));
    expect(refresh).toHaveBeenCalledOnce();
  });
});
