import { useVirtualizer } from '@tanstack/react-virtual';
import { useRef } from 'react';
import type { ResultColumn } from '../../types';
import { formatValue } from '../../lib/format';
import type { GraphSelection } from '../../lib/graphSelection';
import { isRecord, stringField } from '../../lib/guards';

interface Props {
  columns: ResultColumn[];
  rows: unknown[][];
  /**
   * The same selection the plot reports, offered from the rows.
   *
   * A node in a cell is the node on the canvas: clicking either reads it in the margin. Without
   * this the table was a dead end — a grid of serialized records with nowhere to go — and reading
   * one entity meant leaving the view that found it.
   */
  selection?: GraphSelection;
  onSelect?: (selection: GraphSelection) => void;
}

const ROW_HEIGHT = 31;
const DEFAULT_COLUMN_WIDTH = 220;

/** Column types whose values read as quantities, set flush right in tabular figures. */
const NUMERIC_TYPES = new Set(['INTEGER', 'FLOAT', 'NUMBER']);

/**
 * What one cell holds, classified for rendering. Entities become controls; scalars become text.
 * The wire marks entities with `__kind`, which is what the classifier reads.
 */
function cellKind(value: unknown): 'node' | 'relationship' | 'null' | 'plain' {
  if (value === null || value === undefined) return 'null';
  if (isRecord(value)) {
    if (value.__kind === 'node') return 'node';
    if (value.__kind === 'relationship') return 'relationship';
    if (value.type === 'null') return 'null';
  }
  return 'plain';
}

function Cell({ value, selection, onSelect }: { value: unknown; selection?: GraphSelection; onSelect?: (selection: GraphSelection) => void }) {
  const kind = cellKind(value);

  if (kind === 'null') return <span className="cell-null">null</span>;

  if ((kind === 'node' || kind === 'relationship') && onSelect && isRecord(value)) {
    const id = stringField(value, 'id') ?? '';
    const selected = selection !== undefined
      && selection.kind === (kind === 'node' ? 'node' : 'edge')
      && selection.id === id;
    const caption = kind === 'node'
      ? (Array.isArray(value.labels) ? value.labels.filter((label): label is string => typeof label === 'string') : [])
          .map((label) => `:${label}`)
          .join('') || ':Node'
      : `:${stringField(value, 'relationshipType') ?? 'RELATED'}`;
    // A node's own name, when it carries one, is what a reader recognises it by.
    const name = kind === 'node' && isRecord(value.properties)
      ? ['name', 'title', 'username', 'handle']
          .map((key) => (value.properties as Record<string, unknown>)[key])
          .find((candidate) => typeof candidate === 'string' && candidate.length > 0) as string | undefined
      : undefined;
    return (
      <button
        className="cell-ent"
        type="button"
        aria-pressed={selected || undefined}
        title={formatValue(value, 2_000)}
        onClick={() => onSelect(kind === 'node' ? { kind: 'node', id } : { kind: 'edge', id })}
      >
        <span className="cell-ent-kind">{caption}</span>
        {name && <span className="cell-ent-name">{name}</span>}
        <span className="cell-ent-id">#{id}</span>
      </button>
    );
  }

  return <>{formatValue(value)}</>;
}

export function VirtualTable({ columns, rows, selection, onSelect }: Props) {
  const parentRef = useRef<HTMLDivElement>(null);
  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 14,
  });
  const columnVirtualizer = useVirtualizer({
    horizontal: true,
    count: columns.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => DEFAULT_COLUMN_WIDTH,
    overscan: 3,
  });
  const virtualColumns = columnVirtualizer.getVirtualItems();
  const width = columnVirtualizer.getTotalSize();

  if (columns.length === 0) return <div className="result-empty">The query returned no columns.</div>;

  return (
    <div
      className="virtual-table"
      ref={parentRef}
      role="table"
      aria-label="Query result table"
      aria-rowcount={rows.length + 1}
      aria-colcount={columns.length}
      tabIndex={0}
    >
      <div className="virtual-table-header" role="row" style={{ width }}>
        {virtualColumns.map((virtualColumn) => {
          const column = columns[virtualColumn.index];
          if (!column) return null;
          return (
            <div
              className={`virtual-cell header-cell${column.valueType && NUMERIC_TYPES.has(column.valueType) ? ' num' : ''}`}
              role="columnheader"
              key={column.name}
              style={{ width: virtualColumn.size, transform: `translateX(${virtualColumn.start}px)` }}
            >
              <span className="hdr-name">{column.name}</span>
              {column.valueType && <span className="hdr-type">{column.valueType}</span>}
            </div>
          );
        })}
      </div>
      <div className="virtual-table-body" style={{ height: rowVirtualizer.getTotalSize(), width }}>
        {rowVirtualizer.getVirtualItems().map((virtualRow) => {
          const row = rows[virtualRow.index] ?? [];
          return (
            <div
              className="virtual-row"
              role="row"
              aria-rowindex={virtualRow.index + 2}
              key={virtualRow.key}
              style={{ height: virtualRow.size, transform: `translateY(${virtualRow.start}px)`, width }}
            >
              {virtualColumns.map((virtualColumn) => {
                const column = columns[virtualColumn.index];
                if (!column) return null;
                const value = row[virtualColumn.index];
                const numeric = column.valueType !== undefined && NUMERIC_TYPES.has(column.valueType);
                return (
                  <div
                    className={`virtual-cell${numeric ? ' num' : ''}`}
                    role="cell"
                    key={column.name}
                    title={cellKind(value) === 'plain' ? formatValue(value, 2_000) : undefined}
                    style={{ width: virtualColumn.size, transform: `translateX(${virtualColumn.start}px)` }}
                  >
                    <Cell value={value} selection={selection} onSelect={onSelect} />
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
