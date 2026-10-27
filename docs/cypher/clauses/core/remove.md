# `REMOVE`

Remove properties or node labels. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `REMOVE <entity>.<property> [, ...]` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Remove properties or node labels.

## How it behaves

Removing a property does not delete its entity. Removing a label changes label membership and can change index applicability.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
CREATE (n:ReferenceRemove {name: 'Ada', temporary: true}) REMOVE n.temporary RETURN n.name, n.temporary
```

Result:

```
n.name | n.temporary
-------+------------
Ada    | null

1 row
```

Expected result: the query returns the projected rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
CREATE (n:ReferenceRemove:Temporary) REMOVE n:Temporary RETURN labels(n) AS labels
```

Result:

```
labels
-----------------
[ReferenceRemove]

1 row
```

Expected result: the projected rows reflect the clause behavior described above.

## Limitations and trade-offs

Every graph query selects a project explicitly. Reads remain structurally valid during concurrent
writes, but do not promise snapshot isolation or atomic visibility across several changed records.
Use explicit ordering when result order matters.

## See also

- [Core clauses](./README.md)
- [Expressions and nulls](../../expressions.md)

