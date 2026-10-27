# `relationships`

Return a path's relationships in traversal order. Use this function to inspect graph entities in a query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `relationships(path)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained graph records in an existing project |

## What it does

Return a path's relationships in traversal order.

## How it behaves

Only paths are accepted. Repeated node positions do not change the number of relationship positions.

A null argument produces null. Values of an incompatible entity kind report an error.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. Each example writes
its own graph records. Run them in a disposable project if you do not want to retain that data.

## Simple example

```cypher
USE worked_example
CREATE (a)-[:REFERENCE_RELS]->(b) WITH a, b MATCH p = (a)-[:REFERENCE_RELS]->(b) RETURN [r IN relationships(p) | type(r)] AS result
```

Result:

```
result
----------------
[REFERENCE_RELS]

1 row
```

Expected result: one row containing the requested entity information.

## Advanced example

```cypher
USE worked_example
CREATE (a)-[:REFERENCE_RELS_COUNT]->(b) WITH a, b MATCH p = (a)-[:REFERENCE_RELS_COUNT]->(b) RETURN size(relationships(p)) AS result
```

Result:

```
result
------
1

1 row
```

Expected result: one row. Identities and revisions depend on the project's current state.

## Limitations and trade-offs

Entity information is scoped to the selected project and query layers. Reading identity or revision
does not establish snapshot isolation across graph records.

## See also

- [Entity functions](./README.md)
- [Graph patterns](../../clauses/core/match.md)

