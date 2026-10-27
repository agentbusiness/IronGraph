# `CREATE INDEX`

> Declares an access path over a label's property or properties.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `CREATE [RANGE\|TEXT\|VECTOR] INDEX <name> [IF NOT EXISTS] FOR (<var>:<Label>) ON (<var>.<property> [, <var>.<property> …])` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`flights`](../../datasets.md#flights) — OpenFlights airports, airlines and routes |

## What it does

`CREATE INDEX` declares an index on one label and one or more of its properties. The bare form builds an equality index; `RANGE`, `TEXT` and `VECTOR` build the other three kinds.

- **Equality** — exact lookup. `MATCH (a:Airport {iata: 'LHR'})`.
- **Range** — ordered comparison. `WHERE a.latitude > 60`.
- **Text** — matching within text.
- **Vector** — similarity over stored vectors.

Several properties may be named, which builds one composite index rather than several.

## How it behaves

The label and every property must already exist in the project's schema. A property that has never been written is not in the catalogue and the statement is rejected with `index property is not declared` — so an index is declared after the first write, not before it.

Index names are unique within a project; re-declaring a name is rejected rather than replacing the index.

Add `IF NOT EXISTS` after the name for repeatable application setup:

```cypher
USE flights CREATE INDEX airport_by_iata IF NOT EXISTS FOR (airport:Airport) ON (airport.iata)
```

If that name already exists, the statement succeeds with zero updates and leaves the existing
definition and index contents unchanged, even if you supply a different definition. It does not
rebuild the index or verify that the existing definition matches. Without the guard, duplicate
creation returns an error.

An index changes cost, never answers. Every query returns the same rows with or without one. That also means an index can be dropped to test whether it was earning its keep.

## When to use it

Declare an equality index on whatever identifies an entity — the property a loader matches on and an application looks up by. On the reference datasets that is what makes bulk relationship loading feasible at all: without one, every relationship written costs a scan.

Add a range index when queries compare rather than match, and a text index when they search inside strings.

## How it differs from its neighbours

A constraint also builds an index, but its purpose is to reject data rather than to speed a lookup. Declare an index when you want a fast path and a constraint when duplicates are a bug.

## Simple example

The four indexes the flights loader declares, as `SHOW INDEXES` reports them. Three access paths and one ordered comparison over coordinates.

```cypher
USE flights
SHOW INDEXES
```

Expected result: the flights access paths are listed with their current state. Automatic semantic indexes may also appear when local embedding is enabled.

## Advanced example

Equality, range, text and composite equality indexes declared on one label in a scratch project,
then read back. Properties are written first so they exist when the declarations run. A VECTOR index
instead requires list-valued numeric vectors compatible with the project's active embedding profile.

Use a new disposable project and run each setup statement separately:

```cypher
CREATE PROJECT index_example
```

```cypher
USE index_example
CREATE (:Book {isbn: '0262033844', title: 'Introduction to Algorithms', year: 2009}),
       (:Book {isbn: '0201896834', title: 'The Art of Computer Programming', year: 1997}),
       (:Book {isbn: '1449373321', title: 'Designing Data-Intensive Applications', year: 2017})
```

```cypher
USE index_example
CREATE INDEX book_by_isbn FOR (book:Book) ON (book.isbn)
```

```cypher
USE index_example
CREATE RANGE INDEX book_by_year FOR (book:Book) ON (book.year)
```

```cypher
USE index_example
CREATE TEXT INDEX book_title_text FOR (book:Book) ON (book.title)
```

```cypher
USE index_example
CREATE INDEX book_by_year_and_title FOR (book:Book) ON (book.year, book.title)
```

```cypher
USE index_example
SHOW INDEXES
```

Expected result: `book_by_isbn`, `book_by_year`, `book_title_text` and `book_by_year_and_title` are listed. Each row also contains the current scope and reconciliation columns; those columns are null for these application indexes.

## Where it earns its place

- Making a bulk load feasible by indexing the property it matches on.
- Turning an ordered comparison into a range scan.
- Supporting text and similarity retrieval.

## Limitations and trade-offs

- The label and properties must already exist; declare after the first write.
- Index names are unique per project; use `IF NOT EXISTS` to leave an existing index unchanged.
- An index is maintained on every write to the property it covers, so an unused one is pure cost.
- Indexes are per project. A second project needs its own.

## See also

- [`SHOW INDEXES`](./show-indexes.md)
- [`DROP INDEX`](./drop-index.md)
- [`CREATE CONSTRAINT`](../constraints/create-constraint.md)
