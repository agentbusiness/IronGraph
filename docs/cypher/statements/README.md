# Statements

Statements that change what the database holds or how it is organised, as opposed to clauses that shape a query. Administration is Cypher here: there is no second language for creating a project, declaring an index or making a property remember its past.

## [Temporal schema statements](./temporal/README.md)

Three statements that turn an ordinary property into a remembered one. Declaring a property temporal is what gives `AT TIME` and `HISTORY` something to read; `SET … AT TIME` is how a sample gets an event time of its own; a rollup precomputes windowed aggregates over the history that results.

| Page | Summary |
| --- | --- |
| [`ALTER … SET TEMPORAL`](./temporal/alter-set-temporal.md) | Declares a property temporal, so its values are remembered rather than replaced. |
| [`SET … AT TIME`](./temporal/set-at-time.md) | Writes one history sample stamped with the event time you give it. |
| [`CREATE ROLLUP`](./temporal/create-rollup.md) | Declares windowed aggregates over a temporal property so they are maintained rather than recomputed. |

## [Project statements](./projects/README.md)

A project is a named, isolated graph. Every query names one, and nothing falls through to a default — an application cannot accidentally read or write the wrong graph because it forgot to say which.

| Page | Summary |
| --- | --- |
| [`CREATE PROJECT`](./projects/create-project.md) | Creates a named, isolated graph. |
| [`SHOW PROJECTS`](./projects/show-projects.md) | Lists every project with its stable identity and display name. |
| [`DROP PROJECT`](./projects/drop-project.md) | Removes a project and, with `CASCADE`, everything inside it. |

## [Index statements](./indexes/README.md)

Four index kinds, one statement shape. An equality index answers exact lookups, a range index answers ordered comparisons, a text index answers word matching, and a vector index answers similarity. Declaring the right one is what turns a scan into a lookup; declaring the wrong one costs maintenance and buys nothing.

| Page | Summary |
| --- | --- |
| [`CREATE INDEX`](./indexes/create-index.md) | Declares an access path over a label's property or properties. |
| [`SHOW INDEXES`](./indexes/show-indexes.md) | Lists a project's indexes with their kind, state and diagnostic. |
| [`REBUILD INDEX`](./indexes/rebuild-index.md) | Rebuilds an index from the current graph. |
| [`DROP INDEX`](./indexes/drop-index.md) | Removes an index declaration and everything it maintained. |

## [Constraint statements](./constraints/README.md)

A unique constraint is schema authority, not an access path. It rejects a write that would duplicate a value, which is what makes a property safe to treat as identity.

| Page | Summary |
| --- | --- |
| [`CREATE CONSTRAINT`](./constraints/create-constraint.md) | Requires a property to be unique across a label, and enforces it on write. |
| [`DROP CONSTRAINT`](./constraints/drop-constraint.md) | Removes a uniqueness requirement and the index that enforced it. |
