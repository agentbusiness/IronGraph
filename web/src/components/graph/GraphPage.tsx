import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CompletionSchema, Project, QueryHistoryEntry, QueryResult } from '../../types';
import { columnsToRows, fetchCompletionSchema, fetchSchemaCensus, runQuery, type SchemaCensus } from '../../lib/api';
import { addQueryHistory, clearQueryHistory, deleteQueryHistory, listQueryHistory, verifyBrowserPersistence } from '../../lib/db';
import { errorMessage } from '../../lib/format';
import { historyDays } from '../../lib/queryHistory';
import {
  DEFAULT_GRAPH_QUERY,
  isProjectCatalogQuery,
  neighbourhoodQuery,
} from '../../lib/projectQuery';
import {
  EMPTY_RESULT,
  applyQueryEvent,
  deriveCompletionSchema,
  mergeCompletionSchema,
  mergeGraphValues,
} from '../../lib/queryResult';
import { DEFAULT_VIEW_STATE, type GraphViewState } from '../../lib/graphView';
import type { GraphSelection } from '../../lib/graphSelection';
import { CypherEditor } from './CypherEditor';
import { SchemaCensus as SchemaCensusPanel } from './SchemaCensus';
import { GraphInspector } from './GraphInspector';
import { QueryResults } from './QueryResults';
import { elapsedMicroseconds, formatElapsed, formatTime } from '../../lib/format';
import { Rail, Screen } from '../../design/parts';

/**
 * Write the result out as CSV.
 *
 * Done here rather than server-side because the rows are already in the page — asking the engine
 * to run the query a second time to produce a file would be a second answer to one question, and
 * the two could differ. RFC 4180 quoting: a field containing a quote, a comma or a newline is
 * wrapped, and its quotes doubled.
 */
function exportRows(result: QueryResult, cypher: string): void {
  const cell = (value: unknown): string => {
    // A graph cell is not always a scalar: a node's properties come back as a map, a collect() as
    // a list. `String()` turns both into "[object Object]", so the export would quietly lose
    // exactly the rows worth exporting. Anything that is not a primitive goes out as its JSON.
    const text =
      value === null || value === undefined
        ? ''
        : typeof value === 'string'
          ? value
          : typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint'
            ? String(value)
            : (JSON.stringify(value) ?? '');
    return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  const csv = [
    result.columns.map((column) => cell(column.name)).join(','),
    ...result.rows.map((row) => row.map(cell).join(',')),
  ].join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  // Named after the query, so a folder of exports says what each one asked.
  link.download = `${cypher.trim().slice(0, 40).replace(/[^\w]+/g, '-').replace(/^-|-$/g, '') || 'query'}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

interface Props {
  project?: Project;
  draft?: { revision: number; query: string };
  sidebarOpen: boolean;
  onSidebarClose: () => void;
  onProjectsChanged: () => Promise<void>;
}

const EMPTY_SCHEMA: CompletionSchema = { labels: [], relationshipTypes: [], properties: [], functions: [] };

const LAST_QUERY_KEY = 'irongraph:graph:lastQuery';

function lastQueryStorageKey(projectId?: string): string {
  return `${LAST_QUERY_KEY}:${projectId ?? 'none'}`;
}

/** The last query the user worked on for this project, or the default if none was saved. */
function loadStoredQuery(projectId?: string): string {
  try {
    return localStorage.getItem(lastQueryStorageKey(projectId)) ?? DEFAULT_GRAPH_QUERY;
  } catch {
    return DEFAULT_GRAPH_QUERY;
  }
}

export function GraphPage({ project, draft, onSidebarClose, onProjectsChanged }: Props) {
  const [query, setQuery] = useState(() => loadStoredQuery(project?.id));
  const [history, setHistory] = useState<QueryHistoryEntry[]>([]);
  const [schema, setSchema] = useState<CompletionSchema>(EMPTY_SCHEMA);
  const [result, setResult] = useState<QueryResult>(EMPTY_RESULT);
  const [mode, setMode] = useState<'3d' | '2d' | 'table'>('2d');
  // What the engine reported for the last run, for the rail's foot.
  const elapsed = formatElapsed(elapsedMicroseconds(result.statistics));
  const [running, setRunning] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [error, setError] = useState<string>();
  const [persistenceError, setPersistenceError] = useState<string>();
  const abortRef = useRef<AbortController | undefined>(undefined);
  const runSerial = useRef(0);

  /**
   * The plot's view, its selection and its traversals are owned here.
   *
   * All three are read in more than one column of this screen — the bar over the plot changes the
   * view, the margin reads the selection, and a traversal changes the result the table and the plot
   * both draw. Whatever two columns share belongs to the screen that holds them.
   */
  const [view, setView] = useState<GraphViewState>(DEFAULT_VIEW_STATE);
  const [censusOpen, setCensusOpen] = useState(false);
  const [selection, setSelection] = useState<GraphSelection>();
  const [census, setCensus] = useState<SchemaCensus>();
  const [expandingId, setExpandingId] = useState<string>();
  const [expandError, setExpandError] = useState<string>();
  const [expandNote, setExpandNote] = useState<string>();
  const expandAbortRef = useRef<AbortController | undefined>(undefined);
  /** The result a traversal merges into, without making every result a new callback identity. */
  const resultRef = useRef(result);
  resultRef.current = result;

  const changeView = useCallback((patch: Partial<GraphViewState>) => {
    setView((current) => ({ ...current, ...patch }));
  }, []);

  // Restore the last query for this project on mount and whenever the project changes. Runs before
  // the draft effect below so an explicit draft (e.g. "open in graph") still wins on the same commit.
  useEffect(() => {
    setQuery(loadStoredQuery(project?.id));
  }, [project?.id]);

  useEffect(() => {
    if (draft) setQuery(draft.query);
  }, [draft]);

  // Persist the working query per project across console navigation and reloads.
  useEffect(() => {
    try {
      localStorage.setItem(lastQueryStorageKey(project?.id), query);
    } catch {
      /* localStorage unavailable (private mode / quota) — non-fatal, history still persists. */
    }
  }, [query, project?.id]);

  useEffect(() => () => {
    runSerial.current += 1;
    abortRef.current?.abort();
  }, []);

  useEffect(() => {
    abortRef.current?.abort();
    runSerial.current += 1;
    setRunning(false);
    setResult(EMPTY_RESULT);
    setSchema(EMPTY_SCHEMA);
    setError(undefined);
    setSelection(undefined);
    setCensus(undefined);
    if (!project) {
      setHistory([]);
      setLoadingHistory(false);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    setLoadingHistory(true);
    const historyLoad = verifyBrowserPersistence()
      .then(() => listQueryHistory(project.id))
      .then((entries) => {
        if (!cancelled) {
          setHistory(entries);
          setPersistenceError(undefined);
        }
      })
      .catch((cause: unknown) => {
        if (!cancelled) setPersistenceError(`Browser persistence unavailable: ${errorMessage(cause)}`);
      });
    const catalogLoad = fetchCompletionSchema(project.id, controller.signal)
      .then((catalog) => { if (!cancelled) setSchema(catalog); })
      .catch((cause: unknown) => {
        if (!cancelled && !(cause instanceof DOMException && cause.name === 'AbortError')) {
          setError(`Completion catalog unavailable: ${errorMessage(cause)}`);
        }
      });
    // The catalogue names what exists; the census says how much of each there is. They are separate
    // reads because the catalogue rides along on any query and the census is two counting queries,
    // and neither should hold the other's answer back.
    const censusLoad = fetchSchemaCensus(project.id, controller.signal)
      .then((counts) => { if (!cancelled) setCensus(counts); })
      .catch(() => { /* The schema still lists what exists; it just lists it without counts. */ });
    void Promise.allSettled([historyLoad, catalogLoad, censusLoad]).then(() => { if (!cancelled) setLoadingHistory(false); });
    return () => { cancelled = true; controller.abort(); };
  }, [project]);

  useEffect(() => () => expandAbortRef.current?.abort(), []);

  const refreshHistory = useCallback(async (projectId: string) => {
    setHistory(await listQueryHistory(projectId));
  }, []);

  const forgetOne = useCallback(async (id: string) => {
    setHistory((current) => current.filter((entry) => entry.id !== id));
    try {
      await deleteQueryHistory(id);
    } catch (cause) {
      setPersistenceError(`Browser persistence unavailable: ${errorMessage(cause)}`);
      if (project) await refreshHistory(project.id);
    }
  }, [project, refreshHistory]);

  const forgetAll = useCallback(async () => {
    if (!project) return;
    setHistory([]);
    try {
      await clearQueryHistory(project.id);
    } catch (cause) {
      setPersistenceError(`Browser persistence unavailable: ${errorMessage(cause)}`);
      await refreshHistory(project.id);
    }
  }, [project, refreshHistory]);

  const days = useMemo(() => historyDays(history, Date.now()), [history]);

  const execute = useCallback(async (text: string) => {
    const submitted = text;
    if (!submitted.trim() || running) return;
    if (!project && !isProjectCatalogQuery(submitted)) {
      setError('Select a project, or use New at the top to create one.');
      return;
    }
    const serial = ++runSerial.current;
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setError(undefined);
    // A new answer opens on a plot; which plot is the reader's standing choice to keep.
    setMode((current) => (current === '3d' ? current : '2d'));
    setResult(EMPTY_RESULT);
    // A new result is a new graph: what was selected in the last one, and whatever a traversal was
    // saying about it, describe nodes that are about to leave the canvas.
    expandAbortRef.current?.abort();
    setSelection(undefined);
    setExpandingId(undefined);
    setExpandError(undefined);
    setExpandNote(undefined);
    if (project) {
      try {
        await addQueryHistory(project.id, submitted);
        await refreshHistory(project.id);
      } catch (cause) {
        setPersistenceError(`Browser persistence unavailable: ${errorMessage(cause)}`);
      }
    }

    let mutated = false;
    try {
      for await (const event of runQuery({ projectId: project?.id, query: submitted, signal: controller.signal })) {
        if (serial !== runSerial.current) return;
        if (event.type === 'error') throw new Error(`${event.code}: ${event.message}`);
        if (event.type === 'catalog') {
          setSchema((current) => mergeCompletionSchema(current, event.catalog));
          continue;
        }
        if (event.type === 'summary') {
          const updates = event.statistics?.updates;
          mutated = typeof updates === 'number' && updates > 0;
        }
        setResult((current) => applyQueryEvent(current, event));
      }
      setResult((current) => {
        setSchema((currentSchema) => mergeCompletionSchema(currentSchema, deriveCompletionSchema(current)));
        return current;
      });
      if (isProjectCatalogQuery(submitted)) {
        await onProjectsChanged();
      }
      // A write moves the counts under the schema list, so the census is taken again — but only
      // after a run that changed something. Counting is two scans of the whole store, and paying
      // for them after a read that could not have moved a single count is a tax on every query.
      if (project && mutated) {
        const counts = await fetchSchemaCensus(project.id, controller.signal).catch(() => undefined);
        if (counts && serial === runSerial.current) setCensus(counts);
      }
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError')) setError(errorMessage(cause));
    } finally {
      if (serial === runSerial.current) {
        setRunning(false);
        abortRef.current = undefined;
      }
    }
  }, [onProjectsChanged, project, refreshHistory, running]);

  /**
   * Pull one node's neighbourhood into the plot.
   *
   * This is what a double-click on the canvas runs, and what Expand in the margin runs. It reaches
   * the engine as its own query rather than by rewriting the Cypher above: the editor holds the
   * question the user asked, and a traversal is not an edit to it. Only the scene grows — the rows
   * under the table stay the answer to the query that produced them.
   */
  const expandNode = useCallback(async (id: string) => {
    if (!project || running || expandingId !== undefined) return;
    const query = neighbourhoodQuery(id);
    if (!query) {
      setExpandError('This node has no engine identity to traverse from.');
      return;
    }
    const serial = runSerial.current;
    const controller = new AbortController();
    expandAbortRef.current = controller;
    setExpandingId(id);
    setExpandError(undefined);
    setExpandNote(undefined);
    try {
      const values: unknown[] = [];
      let rows = 0;
      for await (const event of runQuery({ projectId: project.id, query, signal: controller.signal })) {
        if (event.type === 'error') throw new Error(`${event.code}: ${event.message}`);
        if (event.type !== 'batch') continue;
        columnsToRows(event.columns ?? []).forEach((row) => {
          rows += 1;
          row.forEach((value) => values.push(value));
        });
      }
      if (serial !== runSerial.current) return;
      const before = resultRef.current;
      const merged = mergeGraphValues(before, values);
      const nodesAdded = merged.nodes.length - before.nodes.length;
      const edgesAdded = merged.edges.length - before.edges.length;
      if (merged !== before) setResult(merged);
      setExpandNote(nodesAdded === 0 && edgesAdded === 0
        ? rows === 0
          ? 'Nothing is attached to this node.'
          : 'Everything it touches is already on the canvas.'
        : `Pulled in ${nodesAdded.toLocaleString()} node${nodesAdded === 1 ? '' : 's'} and ${edgesAdded.toLocaleString()} relationship${edgesAdded === 1 ? '' : 's'}.`);
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError')) setExpandError(errorMessage(cause));
    } finally {
      expandAbortRef.current = undefined;
      setExpandingId((current) => (current === id ? undefined : current));
    }
  }, [project, running, expandingId]);

  /**
   * What a traversal reported belongs to the node it was run from, so it is cleared when the
   * selection moves to another one — and only then. The renderer also re-reports the same selection
   * when the canvas restyles, and that must not wipe the answer the reader is still reading.
   */
  const selectedIdRef = useRef<string>(undefined);
  const selectFromCanvas = useCallback((next: GraphSelection | undefined) => {
    if (selectedIdRef.current !== next?.id) {
      setExpandNote(undefined);
      setExpandError(undefined);
    }
    selectedIdRef.current = next?.id;
    setSelection(next);
  }, []);

  const cancel = () => {
    runSerial.current += 1;
    abortRef.current?.abort();
    abortRef.current = undefined;
    setRunning(false);
  };

  return (
    <Screen name="graph">
      {/*
        * The sidebar is history, and only history.
        *
        * It carried the schema — every kind of node and relationship, with the history squeezed
        * underneath — against a standing decision that this rail is where past queries live. The
        * census was genuinely useful and has not been deleted; it moved to the editor, which is
        * where a writing aid belongs. What is left here has the whole column: the questions this
        * project has been asked, newest first, grouped by the day they were asked on.
        */}
      <div className="col-i">
        <div className="bar">
          <span className="ap">History</span>
          <span className="ap faint">{history.length.toLocaleString()} kept</span>
          <div className="grow"></div>
          {history.length > 0 && (
            <button className="detent" type="button" onClick={() => void forgetAll()}>Clear</button>
          )}
        </div>
        {history.length === 0 && (
          <p className="empty-line">
            {project && loadingHistory ? 'Reading this project\u2019s history\u2026' : 'Queries you run in this project appear here.'}
          </p>
        )}
        <ul className="lst hist">
          {days.map((day) => (
            <Fragment key={day.key}>
              <li className="daybreak">
                <span className="ap">{day.label}</span>
                <div className="grow"></div>
                <span className="ap faint">{day.entries.length.toLocaleString()}</span>
              </li>
              {day.entries.map((entry) => (
                <li key={entry.id} className="hist-row">
                  <button type="button" onClick={() => { setQuery(entry.query); onSidebarClose(); }}>
                    <span className="t hist-query">{entry.query}</span>
                    <span className="r">
                      {/* The day is already the heading above; what a row adds is the hour. */}
                      <span className="ap faint">{formatTime(entry.createdAt)}</span>
                      <span className="grow"></span>
                    </span>
                  </button>
                  <button
                    className="hist-forget ap faint"
                    type="button"
                    aria-label={`Forget: ${entry.query}`}
                    onClick={() => void forgetOne(entry.id)}
                  >
                    Forget
                  </button>
                </li>
              ))}
            </Fragment>
          ))}
        </ul>
      </div>

      {/*
        * The rail is the result, not the schema.
        *
        * It was a truncated list of label names — "Con, Con, Con" for Concept, Contract and
        * Contest, which distinguishes nothing. The design puts the two views here and how long the
        * query took, so switching view is reachable from the rail as well as the bar.
        */}
      <Rail
        screen="graph"
        cap="Result"
        keys={[
          { t: 'P', title: 'Plot', on: mode === '2d', ticks: [mode === '2d'], onSelect: () => setMode('2d') },
          { t: 'S', title: 'Solid', on: mode === '3d', ticks: [mode === '3d'], onSelect: () => setMode('3d') },
          { t: 'T', title: 'Table', on: mode === 'table', ticks: [mode === 'table'], onSelect: () => setMode('table') },
        ]}
        foot={elapsed ?? '—'}
      />

      <div className="col-ii" style={{ display: 'flex', flexDirection: 'column', paddingRight: '22px' }}>
        {(persistenceError || error) && (
          <div className="notice contradiction" role="alert">
            <span className="kindmark"></span>
            <span className="ap lbl">Trouble</span>
            <p>{persistenceError ?? error}</p>
            <div className="acts">
              <button className="detent" type="button" onClick={() => { setError(undefined); setPersistenceError(undefined); }}>Dismiss</button>
            </div>
          </div>
        )}
        {!project && <p className="empty-line">Use New at the top to create a project.</p>}
        {/*
          * The query path itself is untouched.
          *
          * The editor, the runner, the results and the plot are exactly the components they were:
          * this screen changed the frame around them, not what happens when you press Run.
          */}
        <CypherEditor
          value={query}
          schema={schema}
          running={running}
          onChange={setQuery}
          onRun={(text) => void execute(text)}
          onCancel={cancel}
        />
        {/*
          * The census sits with the editor because that is when it is consulted: while writing a
          * statement, not while reading a result. Shut, it is one line of totals; open, it is the
          * two lists side by side, which the reading column has the width for and the sidebar
          * never did.
          */}
        <SchemaCensusPanel
          schema={schema}
          census={census}
          loading={loadingHistory}
          open={censusOpen}
          onOpenChange={setCensusOpen}
          onPick={(text) => { setQuery(text); setCensusOpen(false); }}
        />
        <QueryResults
          result={result}
          running={running}
          mode={mode}
          onModeChange={setMode}
          view={view}
          onViewChange={changeView}
          selection={selection}
          onSelectionChange={selectFromCanvas}
          onExpandNode={(id) => void expandNode(id)}
          expandingNodeId={expandingId}
        />

        {/*
          * There is no saved-view API for the graph, so that control is deliberately absent.
          */}
        {result.rows.length > 0 && (
          <div className="sign" style={{ marginTop: '14px' }}>
            <button className="detent" type="button" onClick={() => exportRows(result, query)}>
              Export rows
            </button>
            <span className="ap faint">
              {result.rows.length.toLocaleString()} row{result.rows.length === 1 ? '' : 's'} · written to this machine
            </span>
          </div>
        )}
      </div>

      {/*
        * The margin reads the plot.
        *
        * It used to carry one note repeating the row count printed over the plot already. What
        * belongs in a margin is an annotation of the mark beside it, and on this screen that is
        * whatever the reader just clicked: the node, the relationship or the community, with
        * everything it carries.
        */}
      <GraphInspector
        selection={selection}
        nodes={result.nodes}
        edges={result.edges}
        rowCount={result.rows.length}
        expandingNodeId={expandingId}
        expandError={expandError}
        expandNote={expandNote}
        view={view}
        onViewChange={changeView}
        onSelect={selectFromCanvas}
        onExpand={(id) => void expandNode(id)}
      />
    </Screen>
  );
}
