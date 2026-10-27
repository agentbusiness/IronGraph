# `SET`

Write properties, maps or node labels. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `SET <entity>.<property> = <value> [, ...]` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Write properties, maps or node labels.

## How it behaves

SET n = map replaces the property map; SET n += map updates the named entries. SET n:Label adds a label. Setting a property to null removes it. SET ... AT TIME writes a declared temporal sample.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
CREATE (n:ReferenceSet) SET n.name = 'Ada' RETURN n.name AS name
```

Result:

```
name
----
Ada

1 row
```

Expected result: the query returns the projected rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
CREATE (n:ReferenceSet {name: 'Ada'}) SET n += {active: true} RETURN n.name, n.active
```

Result:

```
n.name | n.active
-------+---------
Ada    | true

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

