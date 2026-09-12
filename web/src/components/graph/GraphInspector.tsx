import { useState } from 'react';
import type { GraphEdge, GraphNode } from '../../types';
import { formatValue, nodeDisplayLabel } from '../../lib/format';
import {
  adjacentTypes,
  degreeInResult,
  nodeIndex,
  presentProperties,
  type AdjacentType,
  type GraphSelection,
} from '../../lib/graphSelection';
import type { GraphViewState } from '../../lib/graphView';
import { Cross } from '../../design/parts';

interface Props {
  selection?: GraphSelection;
  nodes: GraphNode[];
  edges: GraphEdge[];
  rowCount: number;
  /** The node whose traversal is running, if one is. */
  expandingNodeId?: string;
  expandError?: string;
  /** What the last finished traversal brought back, so a double-click that found nothing says so. */
  expandNote?: string;
  view: GraphViewState;
  onViewChange: (patch: Partial<GraphViewState>) => void;
  onSelect: (selection: GraphSelection | undefined) => void;
  onExpand: (id: string) => void;
}

/** The design draws direction as a rule and a head, not as a character in the text. */
function Arrow({ incoming = false }: { incoming?: boolean }) {
  return (
    <svg className="insp-arrow" viewBox="0 0 14 8" aria-hidden>
      {incoming
        ? <path d="M13.5 4H1M5 1L1 4l4 3" fill="none" stroke="currentColor" strokeWidth="1" />
        : <path d="M0.5 4H13M9 1l4 3-4 3" fill="none" stroke="currentColor" strokeWidth="1" />}
    </svg>
  );
}

function Swatch({ color }: { color?: string }) {
  // Without a colour there is nothing truthful to draw, and an invented one would claim the node is
  // painted in it. The hollow square is the design's own "nothing here yet" mark.
  return <span className={`insp-swatch${color ? '' : ' hollow'}`} style={color ? { background: color } : undefined} aria-hidden />;
}

/** Above this, a property is a passage rather than a value, and is clamped until it is asked for. */
const LONG_VALUE = 220;

function PropertyValue({ value }: { value: unknown }) {
  const [open, setOpen] = useState(false);
  const text = formatValue(value, 4_000);
  const long = text.length > LONG_VALUE;
  return (
    <dd>
      <span className={long && !open ? 'insp-value clamped' : 'insp-value'}>{text}</span>
      {long && (
        <button className="insp-more ap" type="button" onClick={() => setOpen((current) => !current)}>
          {open ? 'Less' : `All ${text.length.toLocaleString()} characters`}
        </button>
      )}
    </dd>
  );
}

function Properties({ properties }: { properties: Record<string, unknown> }) {
  const present = presentProperties(properties);
  if (present.length === 0) {
    return (
      <>
        <Section label="Properties" count={0} />
        <p className="insp-none">None recorded.</p>
      </>
    );
  }
  return (
    <>
      <Section label="Properties" count={present.length} />
      <dl className="insp-props">
        {present.map(([key, value]) => (
          <div key={key}>
            <dt title={key}>{key}</dt>
            <PropertyValue value={value} />
          </div>
        ))}
      </dl>
    </>
  );
}

function Section({ label, count }: { label: string; count?: number }) {
  return (
    <div className="insp-sec">
      <span className="ap">{label}</span>
      <span className="insp-sec-rule" />
      {count !== undefined && <span className="ap faint">{count.toLocaleString()}</span>}
    </div>
  );
}

function Relationships({ adjacent }: { adjacent: AdjacentType[] }) {
  if (adjacent.length === 0) {
    return (
      <>
        <Section label="Relationships here" count={0} />
        <p className="insp-none">Nothing drawn yet. Double-click the node to pull its neighbourhood in.</p>
      </>
    );
  }
  return (
    <>
      <Section label="Relationships here" count={degreeInResult(adjacent)} />
      <ul className="insp-rels">
        {adjacent.map((entry) => (
          <li key={entry.relationshipType}>
            <span className="insp-rel-name" title={entry.relationshipType}>{entry.relationshipType}</span>
            {entry.out > 0 && <span className="insp-rel-count"><Arrow />{entry.out.toLocaleString()}</span>}
            {entry.in > 0 && <span className="insp-rel-count"><Arrow incoming />{entry.in.toLocaleString()}</span>}
          </li>
        ))}
      </ul>
    </>
  );
}

/**
 * What one click on the canvas is about, read in the margin.
 *
 * The design's margin is where a page annotates itself, and a selected node is exactly that: a note
 * about one mark in the plot. It used to be a card floating over the canvas, which covered the
 * graph it was describing and had room for a title and little else. Here it has the column's full
 * height, so a node arrives with its kind, its identity, everything it is attached to in this
 * result, and every property it carries — and the plot stays uncovered underneath.
 */
export function GraphInspector({
  selection,
  nodes,
  edges,
  rowCount,
  expandingNodeId,
  expandError,
  expandNote,
  view,
  onViewChange,
  onSelect,
  onExpand,
}: Props) {
  const byId = nodeIndex(nodes);
  const node = selection?.kind === 'node' ? byId.get(selection.id) : undefined;
  const edge = selection?.kind === 'edge' ? edges.find((candidate) => candidate.id === selection.id) : undefined;
  const cluster = selection?.kind === 'cluster' ? selection : undefined;
  const expanding = expandingNodeId !== undefined;

  const census = (
    <div className="insp-census">
      <span className="ap origin">On this canvas</span>
      <div className="insp-census-row">
        <span className="insp-tally"><b>{nodes.length.toLocaleString()}</b><span className="ap faint">nodes</span></span>
        <span className="insp-tally"><b>{edges.length.toLocaleString()}</b><span className="ap faint">edges</span></span>
        <span className="insp-tally"><b>{rowCount.toLocaleString()}</b><span className="ap faint">rows</span></span>
      </div>
      <p>Read from this machine&rsquo;s graph. Nothing was sent anywhere to answer it.</p>
    </div>
  );

  return (
    <>
      <div className="vrule marg-rule"></div>
      <div className="marg insp">
        <div className="marg-h">
          <Cross />
          <span className="ap">{selection ? 'Selected' : 'Nothing selected'}</span>
          <span className="grow"></span>
          {selection && (
            <button className="insp-clear ap faint" type="button" onClick={() => onSelect(undefined)}>Clear</button>
          )}
        </div>

        {!selection && (
          <>
            <p className="insp-lede">
              Click a node or a relationship to read it here. Double-click a node to pull its
              neighbours and their relationships into the plot.
            </p>
            {census}
          </>
        )}

        {node && selection?.kind === 'node' && (
          <>
            <div className="insp-head">
              <Swatch color={selection.color} />
              <span className="ap">Node</span>
              <span className="insp-id">#{node.id}</span>
            </div>
            <h3 className="insp-title">{nodeDisplayLabel(node)}</h3>
            {node.labels.length > 0 && (
              <div className="insp-labels">
                {node.labels.map((label) => <span className="insp-lb" key={label}>:{label}</span>)}
              </div>
            )}
            <div className="insp-acts">
              <button
                className="detent"
                type="button"
                disabled={expanding}
                onClick={() => onExpand(node.id)}
              >
                {expandingNodeId === node.id ? 'Traversing…' : 'Expand'}
              </button>
              <button
                className="detent"
                type="button"
                aria-pressed={view.focusHops > 0}
                onClick={() => onViewChange({ focusHops: view.focusHops > 0 ? 0 : 1 })}
              >
                Focus
              </button>
            </div>
            {expandError && <p className="insp-trouble" role="alert">{expandError}</p>}
            {!expandError && expandNote && <p className="insp-note">{expandNote}</p>}
            <Relationships adjacent={adjacentTypes(edges, node.id)} />
            <Properties properties={node.properties} />
            {census}
          </>
        )}

        {edge && selection?.kind === 'edge' && (
          <>
            <div className="insp-head">
              <span className="ap">Relationship</span>
              <span className="insp-id">#{edge.id}</span>
            </div>
            <div className="insp-flow">
              <Endpoint
                node={byId.get(edge.source)}
                id={edge.source}
                color={selection.sourceColor}
                onSelect={() => onSelect({ kind: 'node', id: edge.source })}
              />
              <div className="insp-flow-rel">
                <Arrow />
                <span className="insp-rel-name">{edge.relationshipType || 'RELATED'}</span>
              </div>
              <Endpoint
                node={byId.get(edge.target)}
                id={edge.target}
                color={selection.targetColor}
                onSelect={() => onSelect({ kind: 'node', id: edge.target })}
              />
            </div>
            <Properties properties={edge.properties} />
            {census}
          </>
        )}

        {cluster && (
          <>
            <div className="insp-head">
              <Swatch color={cluster.color} />
              <span className="ap">Community</span>
              <span className="insp-id">#{cluster.community}</span>
            </div>
            <h3 className="insp-title">{cluster.memberCount.toLocaleString()} nodes merged</h3>
            <div className="insp-labels">
              <span className="insp-lb">:{cluster.primaryLabel}</span>
            </div>
            <div className="insp-acts">
              <button
                className="detent"
                type="button"
                onClick={() => onViewChange({ clusterMode: 'off', colorBy: 'community' })}
              >
                Break open
              </button>
            </div>
            <Section label="Inside" />
            <dl className="insp-props">
              <div>
                <dt>relationships within</dt>
                <dd><span className="insp-value">{cluster.internalEdges.toLocaleString()}</span></dd>
              </div>
              <div>
                <dt>dominant kind</dt>
                <dd><span className="insp-value">{cluster.primaryLabel}</span></dd>
              </div>
            </dl>
            <Section label="Members" count={cluster.memberCount} />
            <ul className="insp-members">
              {cluster.members.slice(0, 14).map((id) => (
                <li key={id}>
                  <button type="button" onClick={() => onSelect({ kind: 'node', id })}>
                    {nodeDisplayLabel(byId.get(id))}
                  </button>
                </li>
              ))}
              {cluster.memberCount > 14 && (
                <li className="insp-none">and {(cluster.memberCount - 14).toLocaleString()} more</li>
              )}
            </ul>
            {census}
          </>
        )}

        {selection && !node && !edge && !cluster && (
          <>
            <p className="insp-lede">
              What was selected is no longer in this result. Run the query again, or pick another
              node.
            </p>
            {census}
          </>
        )}
      </div>
    </>
  );
}

function Endpoint({ node, id, color, onSelect }: {
  node?: GraphNode;
  id: string;
  color?: string;
  onSelect: () => void;
}) {
  return (
    <button className="insp-end" type="button" onClick={onSelect}>
      <Swatch color={color} />
      <span className="insp-end-name">{node ? nodeDisplayLabel(node) : `#${id}`}</span>
      {/* A label is written the way the schema writes it — `:Item`, not the apparatus's uppercase. */}
      {node?.labels[0]
        ? <span className="insp-lb">:{node.labels[0]}</span>
        : <span className="ap faint">not in this result</span>}
    </button>
  );
}
