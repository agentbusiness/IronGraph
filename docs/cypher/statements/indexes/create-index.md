# `CREATE INDEX`

> Declares an access path over a label's property or properties.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `CREATE [RANGE\|TEXT\|VECTOR] INDEX <name> FOR (<var>:<Label>) ON (<var>.<property> [, <var>.<property> …])` |
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

Result:

```
name                | kind     | state  | diagnostic
--------------------+----------+--------+-----------
airline_by_id       | EQUALITY | ONLINE | null      
airport_by_iata     | EQUALITY | ONLINE | null      
airport_by_id       | EQUALITY | ONLINE | null      
airport_by_latitude | RANGE    | ONLINE | null      

4 rows
```

## Advanced example

All four kinds declared on one label in a scratch project, then read back. The properties are written first so that they exist in the schema when the declarations run.

```cypher
USE index_example
SHOW INDEXES
```

Result:

```
name                   | kind     | state  | diagnostic
-----------------------+----------+--------+-----------
book_by_isbn           | EQUALITY | ONLINE | null      
book_by_year           | RANGE    | ONLINE | null      
book_by_year_and_title | EQUALITY | ONLINE | null      
book_title_text        | TEXT     | ONLINE | null      

4 rows
```

## Where it earns its place

- Making a bulk load feasible by indexing the property it matches on.
- Turning an ordered comparison into a range scan.
- Supporting text and similarity retrieval.

## Limitations and trade-offs

- The label and properties must already exist; declare after the first write.
- Index names are unique per project and re-declaring is rejected.
- An index is maintained on every write to the property it covers, so an unused one is pure cost.
- Indexes are per project. A second project needs its own.

## See also

- [`SHOW INDEXES`](./show-indexes.md)
- [`DROP INDEX`](./drop-index.md)
- [`CREATE CONSTRAINT`](../constraints/create-constraint.md)
