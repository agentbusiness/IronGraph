import type { CommunitySummary } from '../../lib/graphScene';
import type { GraphViewState } from '../../lib/graphView';

interface Props {
  state: GraphViewState;
  onChange: (patch: Partial<GraphViewState>) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  layoutTierLabel: string;
  onFit: () => void;
  betweennessAvailable: boolean;
  focusAvailable: boolean;
  communities: CommunitySummary[];
  communityCount: number;
  modularity: number;
  shownNodes: number;
  shownEdges: number;
  pathStatus?: string;
  searchStatus?: string;
}

interface Option<T> { value: T; label: string; disabled?: boolean; title?: string }

function Segmented<T extends string | number>({ label, value, options, onSelect }: {
  label: string;
  value: T;
  options: Option<T>[];
  onSelect: (value: T) => void;
}) {
  return (
    <div className="graph-control-row">
      <span className="graph-control-label">{label}</span>
      <div className="graph-segmented" role="group" aria-label={label}>
        {options.map((option) => (
          <button
            key={String(option.value)}
            type="button"
            title={option.title}
            disabled={option.disabled}
            aria-pressed={value === option.value}
            className={value === option.value ? 'active' : ''}
            onClick={() => onSelect(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function GraphControls({
  state,
  onChange,
  open,
  onOpenChange,
  layoutTierLabel,
  onFit,
  betweennessAvailable,
  focusAvailable,
  communities,
  communityCount,
  modularity,
  shownNodes,
  shownEdges,
  pathStatus,
  searchStatus,
}: Props) {
  const merged = state.clusterMode === 'merge';
  // Louvain only runs when something on screen reads it, so the resolution slider only appears then.
  const communitiesInUse = merged || state.colorBy === 'community';
  return (
    <>
      <div className="canvas-tools">
        {/* Called with no arguments on purpose: the handler's first parameter is the animation's
            duration, and passing it a click event hands the camera a duration it cannot divide by. */}
        <button type="button" onClick={() => onFit()} aria-label="Fit graph" title="Fit graph">
          Fit
        </button>
        <button
          type="button"
          onClick={() => onOpenChange(!open)}
          aria-label="Graph controls"
          aria-expanded={open}
          title="Graph controls"
          className={open ? 'active' : ''}
        >
          View
        </button>
      </div>

      {open && (
        <div className="graph-control-panel" aria-label="Graph view controls">
          <header>
            <strong>View</strong>
            <button type="button" onClick={() => onOpenChange(false)} aria-label="Close graph controls">Close</button>
          </header>

          {/* Find nodes lives in the bar above the plot; what it matched is reported here. */}
          {searchStatus && <p className="graph-control-note">{searchStatus}</p>}

          <Segmented
            label="Clusters"
            value={state.clusterMode}
            options={[
              { value: 'off', label: 'Expanded', title: 'Show every node' },
              { value: 'merge', label: 'Merged', title: 'Collapse each community into one node with aggregated relationships' },
            ]}
            onSelect={(clusterMode) => onChange({ clusterMode })}
          />

          {communitiesInUse && (
            <>
              <div className="graph-control-row">
                <span className="graph-control-label">Resolution</span>
                <input
                  type="range"
                  min={0.4}
                  max={2.4}
                  step={0.1}
                  value={state.resolution}
                  aria-label="Community resolution"
                  onChange={(event) => onChange({ resolution: Number(event.target.value) })}
                />
                <output>{state.resolution.toFixed(1)}</output>
              </div>
              <p className="graph-control-note">
                {communityCount.toLocaleString()} communities · modularity {modularity.toFixed(3)}
                {merged ? ' · click a cluster to see its members' : ''}
              </p>
            </>
          )}

          {/* Colour by lives in the bar above the plot, where the design puts it. */}

          {state.colorBy === 'community' && !merged && (
            <label className="graph-control-check">
              <input type="checkbox" checked={state.showHulls} onChange={(event) => onChange({ showHulls: event.target.checked })} />
              Draw community regions
            </label>
          )}

          <Segmented
            label="Size by"
            value={state.sizeBy}
            options={[
              { value: 'uniform', label: 'Flat' },
              { value: 'degree', label: 'Degree' },
              { value: 'pagerank', label: 'PageRank' },
              {
                value: 'betweenness',
                label: 'Between',
                disabled: !betweennessAvailable,
                title: betweennessAvailable ? 'Betweenness centrality' : 'Betweenness needs a smaller result to stay interactive',
              },
            ]}
            onSelect={(sizeBy) => onChange({ sizeBy })}
          />

          <Segmented
            label="Focus"
            value={state.focusHops}
            options={[
              { value: 0, label: 'Off' },
              { value: 1, label: '1 hop', disabled: !focusAvailable, title: focusAvailable ? undefined : 'Select a node first' },
              { value: 2, label: '2', disabled: !focusAvailable },
              { value: 3, label: '3', disabled: !focusAvailable },
            ]}
            onSelect={(focusHops) => onChange({ focusHops })}
          />

          <label className="graph-control-check">
            <input
              type="checkbox"
              checked={state.pathMode}
              onChange={(event) => onChange({ pathMode: event.target.checked })}
            />
            Path mode — click two nodes
          </label>
          {pathStatus && <p className="graph-control-note">{pathStatus}</p>}

          {/* Labels is a two-state control in the bar above; `all` is the one setting that has
              no bar equivalent, so it stays reachable here. */}
          <Segmented
            label="All labels"
            value={state.labelMode === 'all' ? 'all' : 'auto'}
            options={[
              { value: 'auto', label: 'As they fit' },
              { value: 'all', label: 'Every node' },
            ]}
            onSelect={(labelMode) => onChange({ labelMode })}
          />

          <div className="graph-control-row graph-control-checks">
            <label className="graph-control-check">
              <input type="checkbox" checked={state.showEdges} onChange={(event) => onChange({ showEdges: event.target.checked })} />
              Edges
            </label>
            <label className="graph-control-check">
              <input type="checkbox" checked={state.curvedEdges} disabled={!state.showEdges} onChange={(event) => onChange({ curvedEdges: event.target.checked })} />
              Curved
            </label>
          </div>

          {communitiesInUse && communities.length > 0 && (
            <div className="graph-legend" aria-label="Community legend">
              {communities.map((community) => (
                <div key={community.community} title={`Anchor: ${community.anchorLabel}`}>
                  <span className="graph-legend-swatch" style={{ background: community.color }} />
                  <span className="graph-legend-name">{community.dominantLabel}</span>
                  <span className="graph-legend-count">{community.nodeCount.toLocaleString()}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="graph-hud-stats" role="status">
        {shownNodes.toLocaleString()} nodes · {shownEdges.toLocaleString()} edges · {layoutTierLabel}
      </div>
    </>
  );
}
