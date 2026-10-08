# Time clauses

Three clauses that put time into a query: `AT TIME` selects an event time for declared temporal properties, `HISTORY` expands one property's samples into rows, and `WINDOW` buckets rows by an instant they carry.

They compose. `AT TIME` selects a temporal-property read instant; `HISTORY` turns one property's past into a row stream; `WINDOW` groups any row stream by time, whether its instants came from `HISTORY` or from an ordinary datetime property. Nodes, relationships and ordinary properties remain current.

| Page | Summary | Standard |
| --- | --- | --- |
| [`AT TIME`](./at-time.md) | Reads declared temporal properties at a chosen event time. | extension |
| [`HISTORY`](./history.md) | Expands a temporal property's samples in a time range into one row each. | extension |
| [`WINDOW`](./window.md) | Buckets rows into time windows over any instant the rows carry. | extension |
