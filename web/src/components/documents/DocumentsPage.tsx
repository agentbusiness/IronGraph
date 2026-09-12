import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Project } from '../../types';
import {
  createDocument,
  deleteDocument,
  listDocuments,
  saveDocument,
  type GraphDocument,
} from '../../lib/documents';
import { errorMessage } from '../../lib/format';
import { Apparatus, Rail, Screen } from '../../design/parts';
import { MarkdownEditor } from './MarkdownEditor';

interface Props { project?: Project }

function when(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? 'not dated' : date.toLocaleDateString(undefined, {
    day: 'numeric', month: 'short', year: 'numeric',
  });
}

export function DocumentsPage({ project }: Props) {
  const [documents, setDocuments] = useState<GraphDocument[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [draft, setDraft] = useState<GraphDocument>();
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string>();
  const [notice, setNotice] = useState<string>();
  const [deleting, setDeleting] = useState(false);

  const refresh = useCallback(async (preferred?: string) => {
    if (!project) return;
    setLoading(true);
    setError(undefined);
    try {
      const loaded = await listDocuments(project);
      setDocuments(loaded);
      const id = preferred ?? selectedId;
      const selected = loaded.find((document) => document.documentId === id);
      if (selected) {
        setSelectedId(selected.documentId);
        setDraft(selected);
      } else if (!draft) {
        setSelectedId(undefined);
      }
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setLoading(false);
    }
  }, [project, selectedId, draft]);

  useEffect(() => {
    setDocuments([]);
    setSelectedId(undefined);
    setDraft(undefined);
    setQuery('');
    setDeleting(false);
    void refresh();
  // Refresh is intentionally keyed to the project; selecting a document must not reread the list.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project?.id]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    if (!needle) return documents;
    return documents.filter((document) =>
      `${document.title}\n${document.body}`.toLocaleLowerCase().includes(needle));
  }, [documents, query]);

  const original = documents.find((document) => document.documentId === selectedId);
  const dirty = Boolean(draft && original && (draft.title !== original.title || draft.body !== original.body));

  const select = (document: GraphDocument) => {
    setSelectedId(document.documentId);
    setDraft(document);
    setNotice(undefined);
    setDeleting(false);
  };

  const runCreate = async () => {
    if (!project || working) return;
    setWorking(true);
    setError(undefined);
    try {
      const created = await createDocument(project);
      setDocuments((current) => [created, ...current]);
      select(created);
      setNotice('Document created in this project.');
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setWorking(false);
    }
  };

  const runSave = async () => {
    if (!project || !draft || working) return;
    setWorking(true);
    setError(undefined);
    try {
      const saved = await saveDocument(project, draft);
      setDraft(saved);
      setDocuments((current) => [saved, ...current.filter((item) => item.documentId !== saved.documentId)]);
      setNotice('Changes saved.');
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setWorking(false);
    }
  };

  const runDelete = async () => {
    if (!project || !draft || working) return;
    setWorking(true);
    setError(undefined);
    try {
      await deleteDocument(project, draft.documentId);
      setDocuments((current) => current.filter((item) => item.documentId !== draft.documentId));
      setDraft(undefined);
      setSelectedId(undefined);
      setDeleting(false);
      setNotice('Document deleted.');
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setWorking(false);
    }
  };

  const notes = draft ? [
    { lbl: 'Stored document', origin: true, p: 'The complete source text stays with this document in the selected project.' },
    { lbl: 'Project', p: project?.name ?? 'No project' },
    { lbl: 'Updated', p: when(draft.updatedAt) },
  ] : [{ lbl: 'Library', p: 'Create a document or choose one from the index.' }];

  return (
    <Screen name="documents">
      <div className="col-i document-index">
        <div className="bar">
          <button className="detent" type="button" onClick={() => void runCreate()} disabled={!project || working}>{working ? 'Creating…' : 'New'}</button>
          <span className="grow"></span>
          <span className="ap faint">{filtered.length} document{filtered.length === 1 ? '' : 's'}</span>
        </div>
        <div className="bar">
          <label className="field grow">
            <span className="ap faint" aria-hidden>Find</span>
            <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search title or body" aria-label="Search documents" />
          </label>
        </div>
        {!project && <p className="empty-line">Select a project to read its documents.</p>}
        {loading && <p className="empty-line" aria-busy="true">Reading document nodes…</p>}
        {!loading && project && filtered.length === 0 && <p className="empty-line">No documents match this index.</p>}
        <ul className="lst document-list">
          {filtered.map((document) => (
            <li key={document.documentId} className={document.documentId === selectedId ? 'on' : undefined}>
              <button type="button" onClick={() => select(document)} aria-current={document.documentId === selectedId ? 'page' : undefined}>
                <span className="t">{document.title}</span>
                <span className="r"><span className="ap faint">{when(document.updatedAt)}</span><span className="grow"></span><span className="ap origin">Document</span></span>
              </button>
            </li>
          ))}
        </ul>
      </div>
      <Rail screen="documents" cap="Documents" keys={filtered.slice(0, 12).map((document, index) => ({
        t: String(index + 1).padStart(2, '0'),
        on: document.documentId === selectedId,
        title: document.title,
        onSelect: () => select(document),
      }))} foot={dirty ? 'Unsaved' : 'Saved'} />
      <article className="col-ii document-work">
        <svg className="rd-over" aria-hidden></svg>
        {error && <div className="notice contradiction" role="alert"><span className="kindmark"></span><span className="ap lbl">Trouble</span><p>{error}</p></div>}
        {notice && <div className="notice" role="status"><span className="kindmark"></span><span className="ap lbl">Recorded</span><p>{notice}</p></div>}
        {!draft ? (
          <div className="document-empty"><p className="ap faint">Document library</p><h1 className="pagekey">Start a document.</h1><p className="prose">Choose a document from the index, or create a new one and begin writing.</p></div>
        ) : (
          <>
            <header className="document-head">
              <label><span className="ap faint">Title</span><input value={draft.title} onChange={(event) => setDraft({ ...draft, title: event.target.value })} aria-label="Document title" /></label>
              <div className="bar bare">
                <span className="ap faint">{dirty ? 'Unsaved changes' : 'Saved'}</span><span className="grow"></span>
                <button className="detent" type="button" onClick={() => setDeleting((value) => !value)} disabled={working}>Delete</button>
                <button className="detent" type="button" aria-pressed={dirty || undefined} onClick={() => void runSave()} disabled={!dirty || working}>{working ? 'Saving…' : 'Save'}</button>
              </div>
            </header>
            <div className="document-body"><span className="ap faint">Document</span><MarkdownEditor value={draft.body} onChange={(body) => setDraft({ ...draft, body })} onSave={() => void runSave()} /></div>
            {deleting && <section className="notice contradiction delete-proof"><span className="kindmark"></span><span className="ap lbl">Delete document</span><p>This permanently removes “{draft.title}” and its connections. This cannot be undone.</p><div className="acts"><button className="detent" type="button" onClick={() => setDeleting(false)} disabled={working}>Keep document</button><button className="detent warn" type="button" onClick={() => void runDelete()} disabled={working}>{working ? 'Deleting…' : 'Delete document'}</button></div></section>}
          </>
        )}
      </article>
      <Apparatus notes={notes} heading="Document" />
    </Screen>
  );
}
