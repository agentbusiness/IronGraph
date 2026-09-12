import { lazy, Suspense, useState } from 'react';
import { pageLoader } from '../../lib/pageChunk';
import type { QueryResult } from '../../types';
import type { GraphViewState } from '../../lib/graphView';
import type { GraphSelection } from '../../lib/graphSelection';
import { elapsedMicroseconds, formatElapsed } from '../../lib/format';
import { Menu } from '../../design/Menu';
import { VirtualTable } from './VirtualTable';

// Both renderers pull in a WebGL stack of their own — Three for 3D, Sigma and Graphology for 2D.
// Loading them on demand keeps the initial bundle to whichever view the user actually opens.
const Graph3D = lazy(pageLoader(() => import('./Graph3D').then((module) => ({ default: module.Graph3D }))));
const Graph2D = lazy(pageLoader(() => import('./Graph2D').then((module) => ({ default: module.Graph2D }))));

type Mode = '3d' | '2d' | 'table';

interface Props {
  result: QueryResult;
  running: boolean;
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  /**
   * The plot's view and its selection are both owned by the page.
   *
   * The bar above the plot changes the view, and the margin beside it reads the selection — two
   * columns of the same screen, neither of them inside the renderer. This component holds the bar;
   * the page holds the state both of them share.
   */
  view: GraphViewState;
  onViewChange: (patch: Partial<GraphViewState>) => void;
  selection?: GraphSelection;
  onSelectionChange: (selection: GraphSelection | undefined) => void;
  onExpandNode: (id: string) => void;
  expandingNodeId?: string;
}

/** What the design's Colour-by pick offers, and what each choice means on a row of the menu. */
const COLOUR_BY: { key: GraphViewState['colorBy']; label: string; note: string }[] = [
  { key: 'label', label: 'Kind', note: 'Primary node label' },
  { key: 'community', label: 'Community', note: 'Louvain, at the resolution set in View' },
  { key: 'degree', label: 'Degree', note: 'How many edges each node has' },
];

export function QueryResults({
  result,
  running,
  mode,
  onModeChange,
  view,
  onViewChange: changeView,
  selection,
  onSelectionChange,
  onExpandNode,
  expandingNodeId,
}: Props) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const statistics = result.statistics;
  const elapsed = formatElapsed(elapsedMicroseconds(statistics));
  // The plot's controls mean nothing over a table. They stay in place and go quiet, so the bar
  // keeps its shape and no control claims to do something it cannot.
  const plotting = mode !== 'table';
  const colour = COLOUR_BY.find((entry) => entry.key === view.colorBy) ?? COLOUR_BY[0]!;
  return (
    <section className="results-panel" aria-label="Query results" aria-busy={running}>
      <div className="bar" style={{ paddingLeft: '0', paddingRight: '0', marginBottom: '12px' }}>
        <div className="seg-set" role="tablist" aria-label="Result view">
          <button role="tab" data-v="plot" aria-selected={mode === '2d'} aria-pressed={mode === '2d'} type="button" onClick={() => onModeChange('2d')}>
            Plot
          </button>
          <button role="tab" data-v="solid" aria-selected={mode === '3d'} aria-pressed={mode === '3d'} type="button" onClick={() => onModeChange('3d')}>
            Solid
          </button>
          <button role="tab" data-v="table" aria-selected={mode === 'table'} aria-pressed={mode === 'table'} type="button" onClick={() => onModeChange('table')}>
            Table
          </button>
        </div>
        <Menu
          prefix="Colour by"
          label={colour.label}
          ariaLabel="Colour nodes by"
          disabled={!plotting}
          items={COLOUR_BY.map((entry) => ({
            key: entry.key,
            label: entry.label,
            note: entry.note,
            selected: entry.key === view.colorBy,
            onSelect: () => changeView({ colorBy: entry.key }),
          }))}
        />
        <button
          className="detent"
          type="button"
          disabled={!plotting}
          aria-pressed={view.labelMode !== 'off'}
          onClick={() => changeView({ labelMode: view.labelMode === 'off' ? 'auto' : 'off' })}
        >
          Labels
        </button>
        <label className="field" style={{ width: '150px' }}>
          <svg viewBox="0 0 10 10" aria-hidden="true">
            <circle cx="4.2" cy="4.2" r="3.2" stroke="currentColor" fill="none" />
            <path d="M6.6 6.6L9 9" stroke="currentColor" />
          </svg>
          <input
            type="search"
            value={view.search}
            disabled={!plotting}
            placeholder="Find nodes"
            aria-label="Find nodes"
            onChange={(event) => changeView({ search: event.target.value })}
          />
        </label>
        <div className="grow"></div>
        <button className="ap faint" type="button" onClick={() => setDetailsOpen((open) => !open)} aria-expanded={detailsOpen}>
          {result.rows.length.toLocaleString()} rows · {result.nodes.length.toLocaleString()} nodes · {result.edges.length.toLocaleString()} edges
          {elapsed ? ` · ${elapsed}` : ''}
        </button>
      </div>
      {result.truncated && (
        <div className="truncation-banner" role="status">Result truncated at the declared browser/server budget. {result.truncationReason}</div>
      )}
      {detailsOpen && (
        <div className="result-details">
          {result.bookmark && <span>Bookmark {result.bookmark}</span>}
          {statistics && Object.entries(statistics).map(([key, value]) => (
            typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
              ? <span key={key}>{key.replaceAll('_', ' ')} {String(value)}</span>
              : null
          ))}
        </div>
      )}
      <div className="result-content" role="tabpanel">
        {mode !== 'table' && result.nodes.length === 0 && result.edges.length === 0 && (
          <div className="empty-line graph-empty">No nodes or relationships in this result.</div>
        )}
        {mode === '3d' && (result.nodes.length > 0 || result.edges.length > 0) && (
          <Suspense fallback={<div className="empty-line">Loading WebGL renderer…</div>}>
            <Graph3D
              nodes={result.nodes}
              edges={result.edges}
              view={view}
              onViewChange={changeView}
              selection={selection}
              onSelectionChange={onSelectionChange}
              onExpandNode={onExpandNode}
              expandingNodeId={expandingNodeId}
            />
          </Suspense>
        )}
        {mode === '2d' && (result.nodes.length > 0 || result.edges.length > 0) && (
          <Suspense fallback={<div className="empty-line">Loading graph renderer…</div>}>
            <Graph2D
              nodes={result.nodes}
              edges={result.edges}
              view={view}
              onViewChange={changeView}
              selection={selection}
              onSelectionChange={onSelectionChange}
              onExpandNode={onExpandNode}
              expandingNodeId={expandingNodeId}
            />
          </Suspense>
        )}
        {mode === 'table' && (
          <VirtualTable
            columns={result.columns}
            rows={result.rows}
            selection={selection}
            onSelect={onSelectionChange}
          />
        )}
      </div>
    </section>
  );
}
