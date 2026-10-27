# Core clauses

Compose graph reads and writes with these clauses. Each page gives syntax, scope and row behavior,
plus two examples. Select a project with [USE](./use.md) before operating on graph data.

| Clause | Purpose |
| --- | --- |
| [`MATCH / OPTIONAL MATCH`](./match.md) | Read nodes, relationships and paths that satisfy a graph pattern. |
| [`WHERE`](./where.md) | Keep rows whose predicate is true. |
| [`UNWIND`](./unwind.md) | Expand a list to one row per element. |
| [`FOR`](./for.md) | Expand a list to rows using the FOR spelling. |
| [`LET`](./let.md) | Add expression bindings while retaining the current row scope. |
| [`FILTER`](./filter.md) | Filter the current rows without adding a projection. |
| [`CREATE`](./create.md) | Create nodes and directed relationships. |
| [`MERGE`](./merge.md) | Match a pattern or create it when it is absent. |
| [`SET`](./set.md) | Write properties, maps or node labels. |
| [`REMOVE`](./remove.md) | Remove properties or node labels. |
| [`DELETE / DETACH DELETE / NODETACH DELETE`](./delete.md) | Delete graph entities, with explicit handling of incident relationships. |
| [`WITH`](./with.md) | Project rows and set the variable scope for the next clause. |
| [`RETURN`](./return.md) | Choose the result columns and finish the query pipeline. |
| [`ORDER BY`](./order-by.md) | Order projected rows by one or more keys. |
| [`SKIP`](./skip.md) | Discard a specified number of ordered result rows. |
| [`LIMIT`](./limit.md) | Keep at most a specified number of result rows. |
| [`CALL / YIELD`](./call.md) | Run a procedure and bind its declared output columns. |
| [`FINISH`](./finish.md) | End a query without returning graph rows. |
| [`UNION / UNION ALL`](./union.md) | Combine result rows from query branches with the same column names. |
| [Path selectors and modes](./paths.md) | Bound path expansion and choose shortest paths. |
| [`USE`](./use.md) | Select a project. |

