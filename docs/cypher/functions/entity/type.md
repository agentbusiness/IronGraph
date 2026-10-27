# `type`

Return a relationship's type name. Use this function to inspect graph entities in a query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `type(relationship)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained graph records in an existing project |

## What it does

Return a relationship's type name.

## How it behaves

Only relationships are accepted. The result is a case-sensitive string.

A null argument produces null. Values of an incompatible entity kind report an error.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. Each example writes
its own graph records. Run them in a disposable project if you do not want to retain that data.

## Simple example

```cypher
USE worked_example
CREATE (a)-[r:REFERENCE_LINK]->(b) RETURN type(r) AS result
```

Result:

```
result
--------------
REFERENCE_LINK

1 row
```

Expected result: one row containing the requested entity information.

## Advanced example

```cypher
USE worked_example
CREATE (a)-[r:REFERENCE_LINK]->(b) RETURN type(r) = 'REFERENCE_LINK' AS result
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

