# `revision`

Return the current revision of a node or relationship. Use this function to inspect graph entities in a query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `revision(entity)` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | Self-contained graph records in an existing project |

## What it does

Return the current revision of a node or relationship.

## How it behaves

Use an entity variable. Revisions describe current entity state; they are not a historical version selector or a promise of atomic visibility across multiple records.

A null argument produces null. Values of an incompatible entity kind report an error.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. Each example writes
its own graph records. Run them in a disposable project if you do not want to retain that data.

## Simple example

```cypher
USE worked_example
CREATE (n:ReferenceRevision {name: 'Ada'}) RETURN revision(n) AS result
```

Expected result: one row containing the requested entity information.

## Advanced example

```cypher
USE worked_example
CREATE (a)-[r:REFERENCE_REVISION]->(b) RETURN revision(r) AS result
```

Expected result: one row. Identities and revisions depend on the project's current state.

## Limitations and trade-offs

Entity information is scoped to the selected project and query layers. Reading identity or revision
does not establish snapshot isolation across graph records.

## See also

- [Entity functions](./README.md)
- [Graph patterns](../../clauses/core/match.md)

