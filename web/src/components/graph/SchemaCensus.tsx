import type { CompletionSchema } from '../../types';
import { DEFAULT_READ_LAYERS, type GraphLayer, type SchemaCensus as Census } from '../../lib/api';
import { labelQuery, relationshipQuery } from '../../lib/projectQuery';

interface Props {
  schema: CompletionSchema;
  census?: Census;
  loading: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onPick: (query: string) => void;
}

interface Row {
  name: string;
  count?: number;
  layers?: GraphLayer[];
}

/**
 * The catalogue, ordered by how much of each thing there is.
 *
 * Names arrive in the engine's own order, which is neither alphabetical nor meaningful to a reader.
 * Once the census is in, the census is the order: what the graph mostly holds belongs at the top of
 * the list of what it can hold. A name the catalogue knows and the census counted none of stays,
 * at zero — the vocabulary is real even where the graph is empty of it.
 */
function rows(names: string[], counts?: Map<string, number>, layers?: Map<string, GraphLayer[]>): Row[] {
  const listed: Row[] = names.map((name) => ({
    name,
    count: counts?.get(name) ?? (counts ? 0 : undefined),
    layers: layers?.get(name),
  }));
  if (!counts) return listed;
  return listed.sort((left, right) => (right.count ?? 0) - (left.count ?? 0) || left.name.localeCompare(right.name));
}

/** Whether a plain statement can see this at all, or whether it needs its layer named. */
function outsideDefaultRead(layers?: GraphLayer[]): GraphLayer | undefined {
  if (!layers || layers.length === 0) return undefined;
  if (layers.some((layer) => DEFAULT_READ_LAYERS.includes(layer))) return undefined;
  return layers[0];
}

function Column({ caption, count, rows: listed, render, onPick }: {
  caption: string;
  count: string;
  rows: Row[];
  render: (row: Row) => { text: string; query: string };
  onPick: (query: string) => void;
}) {
  return (
    <div className="census-col">
      <div className="bar tier">
        <span className="ap">{caption}</span>
        <span className="ap faint">{listed.length.toLocaleString()}</span>
        <div className="grow"></div>
        <span className="ap faint">{count}</span>
      </div>
      <ul className="schema">
        {listed.length === 0 && <li className="empty-line">None yet.</li>}
        {listed.map((row) => {
          const { text, query } = render(row);
          const layer = outsideDefaultRead(row.layers);
          return (
            <li key={row.name}>
              <button type="button" onClick={() => onPick(query)}>
                <span className="lb">{text}</span>
                {layer && <span className="census-layer">{layer.toLowerCase()}</span>}
                <span className="grow"></span>
                <span className="ct">{row.count === undefined ? '' : row.count.toLocaleString()}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/**
 * What the graph holds, out of the way until it is asked for.
 *
 * This was the sidebar, and the sidebar belongs to history. The census is a writing aid — it is
 * consulted while composing a statement, not read continuously — so it sits with the editor and
 * stays shut, one line stating the totals, until someone opens it.
 *
 * The totals cover every layer. An unprefixed statement reads OBSERVED and KNOWLEDGE only, so a
 * census that reported only an unprefixed count would omit WORKSPACE rows. Rows outside the
 * default read say which layer they are in, and are written with it.
 */
export function SchemaCensus({ schema, census, loading, open, onOpenChange, onPick }: Props) {
  const empty = schema.labels.length === 0 && schema.relationshipTypes.length === 0;
  const labels = rows(schema.labels, census?.labels, census?.layersOfLabel);
  const types = rows(schema.relationshipTypes, census?.relationshipTypes, census?.layersOfRelationshipType);

  return (
    <section className={`census${open ? ' open' : ''}`} aria-label="The schema">
      <div className="bar census-bar">
        <button
          className="census-toggle ap"
          type="button"
          aria-expanded={open}
          onClick={() => onOpenChange(!open)}
        >
          <span className="census-mark" aria-hidden></span>
          The schema
        </button>
        <span className="ap faint">
          {empty
            ? (loading ? 'reading the catalogue…' : 'nothing in it yet')
            : `${schema.labels.length.toLocaleString()} kinds of node · ${schema.relationshipTypes.length.toLocaleString()} kinds of relationship`}
        </span>
        <div className="grow"></div>
        {census && (
          // "In three layers" is not decoration: the foot band counts the two layers a plain
          // statement reads, so a bare total here would look like it contradicted the status bar
          // rather than covering more than it does.
          <span className="ap faint">
            {census.nodes.toLocaleString()} nodes · {census.edges.toLocaleString()} edges in{' '}
            {census.byLayer.length.toLocaleString()} layers
          </span>
        )}
        <span className="ap faint">{open ? 'tab completes it too' : ''}</span>
      </div>

      {open && !empty && (
        <>
          <div className="census-cols">
            <Column
              caption="Kinds of node"
              count={census ? `${census.nodes.toLocaleString()} nodes` : ''}
              rows={labels}
              onPick={onPick}
              render={(row) => ({ text: `:${row.name}`, query: labelQuery(row.name, row.layers) })}
            />
            <Column
              caption="Kinds of relationship"
              count={census ? `${census.edges.toLocaleString()} edges` : ''}
              rows={types}
              onPick={onPick}
              render={(row) => ({ text: `-[:${row.name}]-`, query: relationshipQuery(row.name, row.layers) })}
            />
          </div>
          {census && (
            <p className="census-layers">
              {census.byLayer.map((layer) => (
                <span key={layer.layer}>
                  <b>{layer.nodes.toLocaleString()}</b> {layer.layer.toLowerCase()}
                </span>
              ))}
              <span className="census-note">
                A statement with no layer named reads observed and knowledge. Rows marked otherwise
                are written with their layer.
              </span>
            </p>
          )}
        </>
      )}
    </section>
  );
}
