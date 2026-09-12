import type { QueryResult } from '../../types';
import { formatValue } from '../../lib/format';

export function ResultTable({ result }: { result: QueryResult }) {
  if (result.columns.length === 0) {
    return <p className="empty-line">The statement completed without returning columns.</p>;
  }
  return (
    <div className="lesson-result" tabIndex={0} role="region" aria-label="Query result">
      <table className="bt">
        <thead><tr>{result.columns.map((column) => <th key={column.name}>{column.name}</th>)}</tr></thead>
        <tbody>
          {result.rows.map((row, rowIndex) => (
            <tr key={rowIndex}>{result.columns.map((column, columnIndex) => (
              <td key={column.name}>{formatValue(row[columnIndex], 800)}</td>
            ))}</tr>
          ))}
        </tbody>
      </table>
      {result.rows.length === 0 && <p className="empty-line">No rows matched.</p>}
    </div>
  );
}
