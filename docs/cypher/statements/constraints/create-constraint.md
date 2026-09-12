# `CREATE CONSTRAINT`

> Requires a property to be unique across a label, and enforces it on write.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `CREATE CONSTRAINT <name> FOR (<var>:<Label>) REQUIRE <var>.<property> IS UNIQUE` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`flights`](../../datasets.md#flights) — OpenFlights airports, airlines and routes |

## What it does

`CREATE CONSTRAINT` declares that one property must hold a distinct value across every node with a label. A write that would break it is rejected.

The constraint is validated against the data that already exists. If the graph already contains duplicates the declaration fails and nothing changes — a constraint cannot be declared over data that would violate it.

## How it behaves

Enforcement is what distinguishes it from an index. Both make a lookup fast; only a constraint makes a duplicate impossible, which is what lets an application treat the property as identity and use `MERGE` on it without racing.

One property per constraint. There is no composite uniqueness.

## When to use it

Declare one on any property an application treats as identity: a business key, an external identifier, anything a `MERGE` matches on. Without a constraint, uniqueness is a convention the database will not defend.

## How it differs from its neighbours

An equality index makes a lookup fast and permits duplicates. A constraint does both — but fails at declaration time if the data does not already comply, which an index never does.

## Simple example

A constraint over a property whose values are already distinct. Nothing is returned; `SHOW CONSTRAINTS` is how you see it.

```cypher
USE index_example
CREATE CONSTRAINT book_isbn_unique FOR (b:Book) REQUIRE b.isbn IS UNIQUE
```

Result:

```
No rows returned. 0 changes committed.
```

## Advanced example

The constraint refusing a duplicate. The write below names an ISBN that already exists, and is rejected rather than accepted — this example is expected to fail, and the error is the point.

```cypher
USE index_example
CREATE (:Book {isbn: '1449373321', title: 'A second copy', year: 2020})
```

Result:

```
node property uniqueness constraint was violated
```

## Where it earns its place

- Defending a business key the application treats as identity.
- Making `MERGE` on a key safe rather than merely conventional.
- Catching a duplicate at the write that causes it.

## Limitations and trade-offs

- One property per constraint; no composite uniqueness.
- Declaration fails if existing data already violates it.
- Node labels only.

## See also

- [`DROP CONSTRAINT`](./drop-constraint.md)
- [`CREATE INDEX`](../indexes/create-index.md)
