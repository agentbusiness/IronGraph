# `endNode`

Return the target node of a relationship. Use this function to inspect graph entities in a query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `endNode(relationship)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained graph records in an existing project |

## What it does

Return the target node of a relationship.

## How it behaves

This follows the relationship's stored direction, regardless of the direction used by a MATCH pattern.

A null argument produces null. Values of an incompatible entity kind report an error.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. Each example writes
its own graph records. Run them in a disposable project if you do not want to retain that data.

## Simple example

```cypher
USE worked_example
CREATE (a {name: 'A'})-[r:REFERENCE_END]->(b {name: 'B'}) RETURN endNode(r).name AS result
```

Result:

```
result
------
B

1 row
```

Expected result: one row containing the requested entity information.

## Advanced example

```cypher
USE worked_example
CREATE (a {name: 'A'})-[r:REFERENCE_END_ID]->(b {name: 'B'}) RETURN endNode(r).name = b.name AS result
```

Result:

```
result
------
true

1 row
```

Expected result: one row. Identities and revisions depend on the project's current state.

## Limitations and trade-offs

Entity information is scoped to the selected project and query layers. Reading identity or revision
does not establish snapshot isolation across graph records.

## See also

- [Entity functions](./README.md)
- [Graph patterns](../../clauses/core/match.md)

