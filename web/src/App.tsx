import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { StreamConsole } from './components/streams/StreamConsole';
import { Menu } from './design/Menu';
import { Cross } from './design/parts';
import { useProjects } from './hooks/useProjects';
import { useTheme } from './hooks/useTheme';
import type { ThemePreference } from './types';

const GraphPage = lazy(() =>
  import('./components/graph/GraphPage').then((module) => ({ default: module.GraphPage })),
);
const DocumentsPage = lazy(() =>
  import('./components/documents/DocumentsPage').then((module) => ({ default: module.DocumentsPage })),
);
const TrainingPage = lazy(() =>
  import('./components/training/TrainingPage').then((module) => ({ default: module.TrainingPage })),
);
const DocsPage = lazy(() =>
  import('./components/docs/DocsPage').then((module) => ({ default: module.DocsPage })),
);
const SettingsPage = lazy(() =>
  import('./components/settings/SettingsPage').then((module) => ({ default: module.SettingsPage })),
);

type ConsoleView = 'query' | 'streams' | 'documents' | 'training' | 'docs' | 'settings';

const VIEW_PATH: Record<ConsoleView, string> = {
  query: '/web/',
  streams: '/web/streams',
  documents: '/web/documents',
  training: '/web/training',
  docs: '/web/docs',
  settings: '/web/settings',
};

function initialView(): ConsoleView {
  const entry = Object.entries(VIEW_PATH).find(([, path]) => window.location.pathname === path);
  return (entry?.[0] as ConsoleView | undefined) ?? 'query';
}

/** The plate is the dark ground, the page the light one; auto follows the machine. */
const VIEW_LABEL: Record<ThemePreference, string> = { system: 'Auto', dark: 'Plate', light: 'Page' };
const VIEW_ATTR: Record<ThemePreference, string> = { system: 'auto', dark: 'plate', light: 'page' };
const NEXT_VIEW: Record<ThemePreference, ThemePreference> = { system: 'light', light: 'dark', dark: 'system' };

/**
 * The console shell: one band over one screen, all of it on the design's plate.
 *
 * The shell used to be its own stylesheet with its own palette — a teal header over a warm-grey
 * manuscript, two designs in one window. Now `#ig` is the root, so the ground, the inks and the
 * rules reach the band, the query screen and the streams screen from the same tokens, and the
 * plate/page switch turns everything at once.
 */
export default function App() {
  const projects = useProjects();
  const [view, setView] = useState<ConsoleView>(initialView);
  const [theme, setTheme] = useTheme();
  const [creating, setCreating] = useState(false);
  const [projectName, setProjectName] = useState('');
  const [createError, setCreateError] = useState<string>();
  const [submitting, setSubmitting] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onPopState = () => setView(initialView());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const navigate = (next: ConsoleView) => {
    window.history.pushState({}, '', VIEW_PATH[next]);
    setView(next);
  };

  const closeSheet = () => {
    if (submitting) return;
    setCreating(false);
    setProjectName('');
    setCreateError(undefined);
  };

  const createProject = async () => {
    const name = projectName.trim();
    if (!name || submitting) return;
    setSubmitting(true);
    setCreateError(undefined);
    try {
      await projects.create(name);
      setCreating(false);
      setProjectName('');
    } catch (cause) {
      setCreateError(cause instanceof Error ? cause.message : 'Unable to create the project.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main id="ig" data-view={VIEW_ATTR[theme]} className="drawn">
      <svg className="reg tl" viewBox="0 0 11 11" aria-hidden><path d="M5.5 0v11M0 5.5h11" stroke="currentColor" strokeWidth="1" /></svg>
      <svg className="reg tr" viewBox="0 0 11 11" aria-hidden><path d="M5.5 0v11M0 5.5h11" stroke="currentColor" strokeWidth="1" /></svg>
      <svg className="reg bl" viewBox="0 0 11 11" aria-hidden><path d="M5.5 0v11M0 5.5h11" stroke="currentColor" strokeWidth="1" /></svg>
      <svg className="reg br" viewBox="0 0 11 11" aria-hidden><path d="M5.5 0v11M0 5.5h11" stroke="currentColor" strokeWidth="1" /></svg>

      <header className="band">
        <div className="mark">
          <b>IronGraph</b>
          <Cross />
          <i>console</i>
        </div>

        <nav className="seg-set" aria-label="Console screens">
          <button type="button" aria-pressed={view === 'query'} onClick={() => navigate('query')}>
            Query
          </button>
          <button type="button" aria-pressed={view === 'streams'} onClick={() => navigate('streams')}>
            Streams
          </button>
          <button type="button" aria-pressed={view === 'documents'} onClick={() => navigate('documents')}>
            Documents
          </button>
          <button type="button" aria-pressed={view === 'training'} onClick={() => navigate('training')}>
            Training
          </button>
          <button type="button" aria-pressed={view === 'docs'} onClick={() => navigate('docs')}>
            Docs
          </button>
          <button type="button" aria-pressed={view === 'settings'} onClick={() => navigate('settings')}>
            Settings
          </button>
        </nav>

        <div className="grow"></div>

        {/*
          * The catalog's state, said where the projects are named. A hollow pip is the design's
          * "not there": while the list is loading, and when the server did not answer for it.
          */}
        <div className="state" role={projects.loading || projects.error ? 'status' : undefined}>
          <span className={projects.loading || projects.error ? 'pip hollow' : 'pip'}></span>
          {projects.error ? (
            <span className="ap warn">Catalog unreachable</span>
          ) : (
            <span className="ap faint">
              {projects.loading
                ? 'Reading projects…'
                : `${projects.projects.length.toLocaleString()} project${projects.projects.length === 1 ? '' : 's'} · local`}
            </span>
          )}
        </div>

        <Menu
          prefix="Project"
          label={projects.selected?.name ?? 'None'}
          ariaLabel={`Project: ${projects.selected?.name ?? 'none selected'}. Switch project`}
          disabled={projects.projects.length === 0}
          items={projects.projects.map((project) => ({
            key: project.id,
            label: project.name,
            selected: project.id === projects.selectedId,
            onSelect: () => projects.select(project.id),
          }))}
        />
        <button
          className="detent"
          type="button"
          onClick={() => {
            setCreating(true);
            // The sheet exists for this one field, so the field takes focus with it.
            queueMicrotask(() => nameRef.current?.focus());
          }}
          aria-haspopup="dialog"
          aria-expanded={creating}
        >
          New
        </button>
        <button
          className="detent"
          type="button"
          onClick={() => setTheme(NEXT_VIEW[theme])}
          aria-label={`View: ${VIEW_LABEL[theme]}. Change view`}
        >
          {VIEW_LABEL[theme]}
        </button>
      </header>

      {projects.error && (
        <div className="notice contradiction" role="alert">
          <span className="kindmark"></span>
          <span className="ap lbl">Trouble</span>
          <p>The project catalog did not answer: {projects.error}</p>
        </div>
      )}

      <section className="console-leaf" aria-busy={projects.loading}>
        {view === 'query' ? (
          <Suspense fallback={<p className="empty-line">Opening the query screen&hellip;</p>}>
            <GraphPage
              project={projects.selected}
              sidebarOpen={false}
              onSidebarClose={() => undefined}
              onProjectsChanged={projects.refresh}
            />
          </Suspense>
        ) : view === 'streams' ? (
          <StreamConsole project={projects.selected} />
        ) : view === 'documents' ? (
          <Suspense fallback={<p className="empty-line">Opening documents&hellip;</p>}>
            <DocumentsPage project={projects.selected} />
          </Suspense>
        ) : view === 'training' ? (
          <Suspense fallback={<p className="empty-line">Opening training&hellip;</p>}>
            <TrainingPage projects={projects.projects} onProjectsChanged={projects.refresh} />
          </Suspense>
        ) : view === 'docs' ? (
          <Suspense fallback={<p className="empty-line">Opening documentation&hellip;</p>}>
            <DocsPage projects={projects.projects} onProjectsChanged={projects.refresh} />
          </Suspense>
        ) : (
          <Suspense fallback={<p className="empty-line">Opening settings&hellip;</p>}>
            <SettingsPage />
          </Suspense>
        )}
      </section>

      {creating && (
        <div
          className="sheet-scrim"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) closeSheet();
          }}
          onKeyDown={(event) => {
            if (event.key === 'Escape') closeSheet();
          }}
        >
          <form
            className="sheet"
            role="dialog"
            aria-modal="true"
            aria-label="Create project"
            onSubmit={(event) => {
              event.preventDefault();
              void createProject();
            }}
          >
            <div className="sheet-h">
              <Cross />
              <span className="ap">Create project</span>
              <span className="grow"></span>
              <span className="ap faint">its own graph, indexes and streams</span>
            </div>
            <div className="sheet-b">
              <label className="fld">
                <span className="ap faint">Project name</span>
                <input
                  ref={nameRef}
                  type="text"
                  value={projectName}
                  onChange={(event) => setProjectName(event.target.value)}
                  disabled={submitting}
                  autoComplete="off"
                  spellCheck={false}
                />
              </label>
              {createError && <p className="ap warn" role="alert">{createError}</p>}
            </div>
            <div className="sheet-f">
              <span className="grow"></span>
              <button className="detent" type="button" onClick={closeSheet} disabled={submitting}>
                Cancel
              </button>
              <button
                className="detent"
                type="submit"
                aria-pressed={projectName.trim() ? true : undefined}
                disabled={!projectName.trim() || submitting}
              >
                {submitting ? 'Creating…' : 'Create project'}
              </button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}
