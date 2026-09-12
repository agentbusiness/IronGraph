# `DROP CONSTRAINT`

> Removes a uniqueness requirement and the index that enforced it.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `DROP CONSTRAINT [IF EXISTS] <name>` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`flights`](../../datasets.md#flights) — OpenFlights airports, airlines and routes |

## What it does

`DROP CONSTRAINT` removes a uniqueness requirement. Writes that were rejected before are accepted afterwards, so this statement changes what the database will store. `IF EXISTS` makes it idempotent.

## How it behaves

Unlike dropping an index, this is not cost-only. Duplicates become possible the moment the constraint is gone, and re-declaring it later will fail if any arrived in the meantime — so dropping a constraint on a live system is a decision about data, not about performance.

## When to use it

Drop one when the uniqueness rule is genuinely no longer true: a key that has become non-unique by design, or a migration that must temporarily hold both old and new values.

## How it differs from its neighbours

`DROP INDEX` removes an access path and cannot change what is storable. This removes a rule, and can.

## Simple example

The idempotent form, on a constraint that does not exist.

```cypher
USE index_example
DROP CONSTRAINT IF EXISTS never_declared
```

Result:

```
No rows returned. 0 changes committed.
```

## Advanced example

The write that the constraint rejected, accepted once it is gone. Both books now carry the same ISBN, which is exactly what the constraint existed to prevent.

```cypher
USE index_example
MATCH (book:Book {isbn: '1449373321'})
RETURN count(book) AS books_sharing_that_isbn,
       collect(book.title) AS titles
```

Result:

```
books_sharing_that_isbn | titles                                                                                                       
------------------------+--------------------------------------------------------------------------------------------------------------
2                       | [{"type":"string","value":"Designing Data-Intensive Applications"},{"type":"string","value":"A second copy"}]

1 row
```

## Where it earns its place

- Retiring a uniqueness rule that no longer holds.
- Migrations that must hold old and new keys at once.

## Limitations and trade-offs

- Changes what the database will accept, unlike dropping an index.
- Re-declaring later fails if duplicates arrived while it was gone.
- Without `IF EXISTS`, a missing constraint is an error.

## See also

- [`CREATE CONSTRAINT`](./create-constraint.md)
