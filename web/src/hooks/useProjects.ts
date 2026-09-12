import { useCallback, useEffect, useRef, useState } from 'react';
import type { Project } from '../types';
import { createProject, fetchProjects } from '../lib/api';
import { errorMessage } from '../lib/format';
import { onProjectMissing } from '../lib/projectMissing';

const STORAGE_KEY = 'irongraph-project';

function storedProject(): string | undefined {
  try { return localStorage.getItem(STORAGE_KEY) ?? undefined; }
  catch { return undefined; }
}

function persistProject(id?: string): void {
  try {
    if (id) localStorage.setItem(STORAGE_KEY, id);
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Selection remains valid for this page lifetime when localStorage is unavailable.
  }
}

export interface ProjectsState {
  projects: Project[];
  selected?: Project;
  selectedId?: string;
  loading: boolean;
  error?: string;
  select: (id: string) => void;
  refresh: () => Promise<void>;
  create: (name: string) => Promise<void>;
}

export function useProjects(): ProjectsState {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>(storedProject);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const abortRef = useRef<AbortController | undefined>(undefined);

  const refresh = useCallback(async (preferredName?: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    try {
      const loaded = await fetchProjects(controller.signal);
      setProjects(loaded);
      setError(undefined);
      setSelectedId((current) => {
        const preferred = preferredName
          ? loaded.find((project) => project.name === preferredName)
          : undefined;
        if (preferred) {
          persistProject(preferred.id);
          return preferred.id;
        }
        if (current && loaded.some(({ id }) => id === current)) return current;
        const fallback = loaded[0]?.id;
        persistProject(fallback);
        return fallback;
      });
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError')) {
        setError(errorMessage(cause));
        throw cause;
      }
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh().catch(() => undefined);
    return () => abortRef.current?.abort();
  }, [refresh]);

  // The list is read once at mount, so a project dropped afterwards — a rebuilt database, a drop
  // from another window — leaves this hook holding an id that still resolves locally and never
  // again on the server. The endpoints are the only ones who know: they answer PROJECT_NOT_FOUND.
  // One such answer for the selected project re-reads the catalogue, which either moves the app
  // onto a project that exists or leaves it with none; either way the polls stop asking for the
  // dead one. Guarded against re-entry so a page mid-poll cannot queue a refresh per request.
  const revalidating = useRef(false);
  useEffect(
    () =>
      onProjectMissing((missing) => {
        if (missing && missing !== selectedId) return;
        if (revalidating.current) return;
        revalidating.current = true;
        void refresh()
          .catch(() => undefined)
          .finally(() => {
            revalidating.current = false;
          });
      }),
    [refresh, selectedId],
  );

  const select = useCallback((id: string) => {
    if (!projects.some((project) => project.id === id)) return;
    persistProject(id);
    setSelectedId(id);
  }, [projects]);

  const create = useCallback(async (name: string) => {
    const trimmed = name.trim();
    await createProject(trimmed);
    await refresh(trimmed);
  }, [refresh]);

  const selected = projects.find(({ id }) => id === selectedId);
  return {
    projects,
    selected,
    // A persisted id is only a preference until SHOW PROJECTS confirms it belongs to this
    // database. Never expose a stale id to polling effects during a clean-database startup.
    selectedId: selected?.id,
    loading,
    error,
    select,
    refresh,
    create,
  };
}
