# Clauses

The pieces a query pipeline is built from. Documented here are the clauses IronGraph adds to Cypher rather than the ones it shares with it: the three that put time into a query, the two that choose which layers a query sees and writes to, and the one that ranks rows by similarity.

## [Time clauses](./time/README.md)

Three clauses that put time into a query: `AT TIME` moves the whole query to a past instant, `HISTORY` expands one property's samples into rows, and `WINDOW` buckets rows by an instant they carry.

They compose. `AT TIME` chooses which graph you are looking at; `HISTORY` turns one property's past into a row stream; `WINDOW` groups any row stream by time, whether its instants came from `HISTORY` or from an ordinary datetime property.

| Page | Summary |
| --- | --- |
| [`AT TIME`](./time/at-time.md) | Runs the whole query against the graph as it stood at a past instant. |
| [`HISTORY`](./time/history.md) | Expands a temporal property's samples in a time range into one row each. |
| [`WINDOW`](./time/window.md) | Buckets rows into time windows over any instant the rows carry. |

## [Layer clauses](./layers/README.md)

Every project is partitioned into three layers, and every query chooses which it reads and which it writes. The choice is made once, ahead of the query body, and applies to the whole statement.

The three layers carry fixed meanings.

- `OBSERVED` — facts captured from source activity. What happened.
- `KNOWLEDGE` — curated understanding. What has been concluded.
- `WORKSPACE` — provisional or application working state. What is being tried.

`OBSERVED` and `KNOWLEDGE` together form the default read view. `WORKSPACE` is never in it unless a query asks, which is what keeps scratch data out of results that did not request it.

| Page | Summary |
| --- | --- |
| [`USE LAYER`](./layers/use-layer.md) | Chooses which layers the query reads. |
| [`WRITE LAYER`](./layers/write-layer.md) | Chooses which layer the query's writes land in. |

## [Search](./search/README.md)

One clause and one statement. `CREATE EMBEDDING INDEX` turns a text property into searchable vectors; `SEARCH` uses them to filter and rank rows that a pattern has already bound.

| Page | Summary |
| --- | --- |
| [`CREATE EMBEDDING INDEX`](./search/create-embedding-index.md) | Encodes a text property into vectors with the local model and keeps them searchable. |
| [`SEARCH`](./search/search.md) | Filters and ranks already-bound rows by similarity, binding the score. |
