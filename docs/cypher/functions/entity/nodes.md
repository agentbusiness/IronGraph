# `nodes`

Return a path's nodes in traversal order. Use this function to inspect graph entities in a query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `nodes(path)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained graph records in an existing project |

## What it does

Return a path's nodes in traversal order.

## How it behaves

Only paths are accepted. A path with one relationship contains two node positions.

A null argument produces null. Values of an incompatible entity kind report an error.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. Each example writes
its own graph records. Run them in a disposable project if you do not want to retain that data.

## Simple example

```cypher
USE worked_example
CREATE (a:ReferencePath {name: 'A'})-[:REFERENCE_PATH]->(b:ReferencePath {name: 'B'}) WITH a, b MATCH p = (a)-[:REFERENCE_PATH]->(b) RETURN [n IN nodes(p) | n.name] AS result
```

Result:

```
result
------
[A, B]

1 row
```

Expected result: one row containing the requested entity information.

## Advanced example

```cypher
USE worked_example
CREATE (a)-[:REFERENCE_NODES]->(b) WITH a, b MATCH p = (a)-[:REFERENCE_NODES]->(b) RETURN size(nodes(p)) AS result
```

Result:

```
result
------
2

1 row
```

Expected result: one row. Identities and revisions depend on the project's current state.

## Limitations and trade-offs

Entity information is scoped to the selected project and query layers. Reading identity or revision
does not establish snapshot isolation across graph records.

## See also

- [Entity functions](./README.md)
- [Graph patterns](../../clauses/core/match.md)

