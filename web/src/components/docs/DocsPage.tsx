import { useMemo, useState } from 'react';
import type { Project } from '../../types';
import { Apparatus, Caret, Rail, Screen } from '../../design/parts';
import { DOC_PAGES, type DocPage } from './docsCatalog';
import { MarkdownDocument } from './MarkdownDocument';
import { executeQuery } from '../../lib/simpleQuery';
import { errorMessage } from '../../lib/format';
import { BUNDLED_DATASETS, requiredBundledDatasets } from '../../lib/datasets';

function groupPages(pages: DocPage[]): { section: string; pages: DocPage[] }[] {
  const groups = new Map<string, DocPage[]>();
  pages.forEach((page) => groups.set(page.section, [...(groups.get(page.section) ?? []), page]));
  return Array.from(groups, ([section, entries]) => ({ section, pages: entries }));
}

export function DocsPage({ projects, onProjectsChanged }: { projects: Project[]; onProjectsChanged: () => Promise<void> }) {
  const [selectedId, setSelectedId] = useState(DOC_PAGES.find((page) => page.id === 'getting-started')?.id ?? DOC_PAGES[0]?.id ?? '');
  const [query, setQuery] = useState('');
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(['Start']));
  const selected = DOC_PAGES.find((page) => page.id === selectedId) ?? DOC_PAGES[0];
  const filtered = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    if (!needle) return DOC_PAGES;
    return DOC_PAGES.filter((page) => `${page.title}\n${page.section}\n${page.body}`.toLocaleLowerCase().includes(needle));
  }, [query]);
  const groups = useMemo(() => groupPages(filtered), [filtered]);
  const allGroups = useMemo(() => groupPages(DOC_PAGES), []);
  const searching = Boolean(query.trim());

  const navigate = (id: string) => {
    const page = DOC_PAGES.find((entry) => entry.id === id);
    if (page) setExpanded((current) => new Set(current).add(page.section));
    setSelectedId(id);
    document.querySelector('.docs-work')?.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const toggleSection = (section: string) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(section)) next.delete(section);
      else next.add(section);
      return next;
    });
  };

  return (
    <Screen name="docs">
      <nav className="col-i docs-index" aria-label="Documentation index">
        <div className="bar"><label className="field grow"><span className="ap faint" aria-hidden>Find</span><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search documentation" aria-label="Search documentation" /></label></div>
        <div className="doc-count ap faint">{filtered.length} of {DOC_PAGES.length} pages</div>
        <ul className="docs-groups">{groups.map((group) => {
          const open = searching || expanded.has(group.section);
          return <li className="docs-group" key={group.section}>
            <button className="docs-section" type="button" onClick={() => toggleSection(group.section)} aria-expanded={open}>
              <Caret dir={open ? 'down' : 'right'} />
              <span>{group.section}</span>
              <span className="docs-section-count">{group.pages.length}</span>
            </button>
            {open && <ul className="lst docs-list">{group.pages.map((page) => (
              <li key={page.id} className={page.id === selected?.id ? 'on' : undefined}>
                <button type="button" onClick={() => navigate(page.id)} aria-current={page.id === selected?.id ? 'page' : undefined}>
                  <span className="t">{page.title}</span><span className="r"><span className="ap faint">{page.id}</span></span>
                </button>
              </li>
            ))}</ul>}
          </li>;
        })}</ul>
      </nav>
      <Rail screen="docs" cap="Sections" keys={allGroups.map((group, index) => ({ t: String(index + 1).padStart(2, '0'), on: group.section === selected?.section, title: group.section, onSelect: () => toggleSection(group.section) }))} foot="Guide" />
      <article className="col-ii docs-work">
        <svg className="rd-over" aria-hidden></svg>
        {selected ? <div className="docs-prose"><p className="ap origin">{selected.section} · {selected.id}</p>{selected.id === 'cypher/datasets' ? <DatasetImports projects={projects} onProjectsChanged={onProjectsChanged} /> : <DocDatasetImports body={selected.body} projects={projects} onProjectsChanged={onProjectsChanged} />}<MarkdownDocument body={selected.body} docId={selected.id} projects={projects} onProjectsChanged={onProjectsChanged} onNavigate={navigate} /></div> : <p className="empty-line">No documentation page matched.</p>}
      </article>
      <Apparatus heading="Reading" notes={[
        { lbl: 'Local reference', origin: true, p: 'This reader is built from the public documentation shipped with IronGraph.' },
        { lbl: 'Live Cypher', p: 'Every Cypher block can run here when its named project is loaded.' },
        { lbl: 'Dataset rule', p: 'Reference datasets live in projects named after their canonical slug. Existing projects are never re-imported.' },
      ]} />
    </Screen>
  );
}

function DocDatasetImports({ body, projects, onProjectsChanged }: { body: string; projects: Project[]; onProjectsChanged: () => Promise<void> }) {
  const required = requiredBundledDatasets(body);
  const [importing, setImporting] = useState<string>();
  const [error, setError] = useState<string>();
  const missing = required.filter((dataset) => !projects.some((project) => project.name === dataset));
  if (missing.length === 0) return null;
  const run = async (dataset: string) => {
    if (importing) return;
    setImporting(dataset);
    setError(undefined);
    try {
      await executeQuery(`IMPORT DATASET ${dataset}`);
      await onProjectsChanged();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setImporting(undefined);
    }
  };
  return <>{missing.map((dataset) => (
    <section className="import-gate" data-mark="import" key={dataset}>
      <div className="import-state"><span className="pip hollow"></span><span className="ap live">Dataset not loaded</span></div>
      <h2>Import into <code>{dataset}</code></h2>
      <p>This page uses bundled example data. IronGraph imports it into its own project and leaves an existing project untouched.</p>
      <div className="bar bare"><button className="detent" type="button" aria-pressed="true" onClick={() => void run(dataset)} disabled={Boolean(importing)}>{importing === dataset ? 'Importing…' : 'Import dataset'}</button></div>
    </section>
  ))}{error && <div className="notice contradiction" role="alert"><span className="kindmark"></span><span className="ap lbl">Import failed</span><p>{error}</p></div>}</>;
}

function DatasetImports({ projects, onProjectsChanged }: { projects: Project[]; onProjectsChanged: () => Promise<void> }) {
  const [importing, setImporting] = useState<string>();
  const [error, setError] = useState<string>();
  const run = async (dataset: string) => {
    if (importing) return;
    setImporting(dataset);
    setError(undefined);
    try {
      await executeQuery(`IMPORT DATASET ${dataset}`);
      await onProjectsChanged();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setImporting(undefined);
    }
  };
  return (
    <section className="docs-dataset-imports" aria-label="Bundled dataset imports">
      <div className="bar bare"><span className="ap">Bundled datasets</span><span className="grow"></span><span className="ap faint">One project each</span></div>
      <div className="dataset-import-grid">{BUNDLED_DATASETS.map((dataset) => {
        const present = projects.some((project) => project.name === dataset);
        return <div className="dataset-import-row" key={dataset}><span className={present ? 'pip' : 'pip hollow'}></span><b>{dataset}</b><span className="grow"></span>{present ? <span className="ap faint">Imported</span> : <button className="detent" type="button" onClick={() => void run(dataset)} disabled={Boolean(importing)}>{importing === dataset ? 'Importing…' : 'Import'}</button>}</div>;
      })}</div>
      {error && <div className="notice contradiction" role="alert"><span className="kindmark"></span><span className="ap lbl">Import failed</span><p>{error}</p></div>}
    </section>
  );
}
