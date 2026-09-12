# `DROP INDEX`

> Removes an index declaration and everything it maintained.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `DROP INDEX [IF EXISTS] <name>` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`flights`](../../datasets.md#flights) — OpenFlights airports, airlines and routes |

## What it does

`DROP INDEX` removes an index. Queries continue to return the same rows, reaching them by scan instead of by lookup. `IF EXISTS` makes the statement idempotent, which is what a teardown or migration script needs.

## How it behaves

Because an index changes cost and not answers, dropping one is safe in the sense that nothing becomes wrong — and unsafe in the sense that something may become far slower. On a large graph the difference between a lookup and a scan is the difference between milliseconds and minutes.

Without `IF EXISTS`, dropping an index that is not there fails with `index does not exist`. With it, the statement succeeds and does nothing.

## When to use it

Drop an index that is not being used, or one whose maintenance cost outweighs what it saves. Use `IF EXISTS` in anything that might run twice.

## How it differs from its neighbours

`REBUILD INDEX` keeps the declaration and recomputes contents; `DROP INDEX` removes both. Dropping a constraint takes `DROP CONSTRAINT`, even though a constraint is index-backed.

## Simple example

The idempotent form. Naming an index that does not exist succeeds and changes nothing, so a teardown script can run twice.

```cypher
USE flights
DROP INDEX IF EXISTS an_index_that_was_never_declared
```

Result:

```
No rows returned. 0 changes committed.
```

## Advanced example

Dropped and re-declared, with the same query run afterwards. The rows are identical to the ones the indexed lookup returned — the index was never part of the answer.

```cypher
USE index_example
MATCH (book:Book {isbn: '1449373321'})
RETURN book.isbn AS isbn, book.title AS title, book.year AS year
```

Result:

```
isbn       | title                                 | year
-----------+---------------------------------------+-----
1449373321 | Designing Data-Intensive Applications | 2017

1 row
```

## Where it earns its place

- Removing an index that is not earning its maintenance cost.
- Teardown and migration scripts that must be re-runnable.
- Measuring what an index was actually buying.

## Limitations and trade-offs

- Without `IF EXISTS`, a missing index is an error.
- Queries stay correct and can become dramatically slower.
- An index backing a constraint is not dropped this way.

## See also

- [`CREATE INDEX`](./create-index.md)
