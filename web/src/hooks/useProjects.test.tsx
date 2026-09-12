import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createProject, fetchProjects } from '../lib/api';
import { reportProjectMissing } from '../lib/projectMissing';
import { useProjects } from './useProjects';

vi.mock('../lib/api', () => ({
  createProject: vi.fn(),
  fetchProjects: vi.fn(),
}));

const mockedCreateProject = vi.mocked(createProject);
const mockedFetchProjects = vi.mocked(fetchProjects);

beforeEach(() => {
  localStorage.clear();
  mockedCreateProject.mockReset();
  mockedFetchProjects.mockReset();
});

afterEach(() => vi.restoreAllMocks());

describe('useProjects', () => {
  it('does not expose a persisted project before the current catalog validates it', async () => {
    localStorage.setItem('irongraph-project', 'project-from-another-database');
    let resolveProjects!: (projects: Array<{ id: string; name: string }>) => void;
    mockedFetchProjects.mockReturnValue(
      new Promise((resolve) => {
        resolveProjects = resolve;
      }),
    );

    const { result } = renderHook(() => useProjects());
    expect(result.current.selectedId).toBeUndefined();

    resolveProjects([]);
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.selectedId).toBeUndefined();
    expect(localStorage.getItem('irongraph-project')).toBeNull();
  });

  it('re-reads the catalog when the selected project is refused as missing', async () => {
    mockedFetchProjects
      .mockResolvedValueOnce([{ id: 'project-old', name: 'Testing' }])
      .mockResolvedValueOnce([{ id: 'project-new', name: 'Testing' }]);

    const { result } = renderHook(() => useProjects());
    await waitFor(() => expect(result.current.selectedId).toBe('project-old'));

    // What a page polling the rebuilt database gets back, once per failed request.
    act(() => {
      reportProjectMissing('project-old');
      reportProjectMissing('project-old');
    });

    await waitFor(() => expect(result.current.selectedId).toBe('project-new'));
    // Twice, not three times: the mount read plus one revalidation, however many polls failed.
    expect(mockedFetchProjects).toHaveBeenCalledTimes(2);
    expect(localStorage.getItem('irongraph-project')).toBe('project-new');
  });

  it('selects and persists the project it just created', async () => {
    mockedFetchProjects
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([{ id: 'project-analytics', name: 'Analytics' }]);
    mockedCreateProject.mockResolvedValue(undefined);

    const { result } = renderHook(() => useProjects());
    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => { await result.current.create(' Analytics '); });

    expect(mockedCreateProject).toHaveBeenCalledWith('Analytics');
    expect(result.current.selected).toEqual({ id: 'project-analytics', name: 'Analytics' });
    expect(localStorage.getItem('irongraph-project')).toBe('project-analytics');
  });
});
