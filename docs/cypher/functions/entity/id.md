# `id`

Return the stable identity of a node or relationship within its project. Use this function to inspect graph entities in a query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `id(entity)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained graph records in an existing project |

## What it does

Return the stable identity of a node or relationship within its project.

## How it behaves

Use an entity variable. Node and relationship identities have separate namespaces; pair identity with project and entity kind in applications. Do not infer creation time from its numeric value.

A null argument produces null. Values of an incompatible entity kind report an error.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. Each example writes
its own graph records. Run them in a disposable project if you do not want to retain that data.

## Simple example

```cypher
USE worked_example
CREATE (n:ReferenceIdentity) RETURN id(n) AS result
```

Expected result: one row containing the requested entity information.

## Advanced example

```cypher
USE worked_example
CREATE (a)-[r:REFERENCE_ID]->(b) RETURN id(a) <> id(b) AS result
```

Expected result: one row. Identities and revisions depend on the project's current state.

## Limitations and trade-offs

Entity information is scoped to the selected project and query layers. Reading identity or revision
does not establish snapshot isolation across graph records.

## See also

- [Entity functions](./README.md)
- [Graph patterns](../../clauses/core/match.md)

