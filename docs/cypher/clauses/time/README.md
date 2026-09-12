# Time clauses

Three clauses that put time into a query: `AT TIME` moves the whole query to a past instant, `HISTORY` expands one property's samples into rows, and `WINDOW` buckets rows by an instant they carry.

They compose. `AT TIME` chooses which graph you are looking at; `HISTORY` turns one property's past into a row stream; `WINDOW` groups any row stream by time, whether its instants came from `HISTORY` or from an ordinary datetime property.

| Page | Summary | Standard |
| --- | --- | --- |
| [`AT TIME`](./at-time.md) | Runs the whole query against the graph as it stood at a past instant. | extension |
| [`HISTORY`](./history.md) | Expands a temporal property's samples in a time range into one row each. | extension |
| [`WINDOW`](./window.md) | Buckets rows into time windows over any instant the rows carry. | extension |
