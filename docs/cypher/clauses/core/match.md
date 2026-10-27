# `MATCH / OPTIONAL MATCH`

Read nodes, relationships and paths that satisfy a graph pattern. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `MATCH [<path> =] <pattern> [WHERE <predicate>]` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Read nodes, relationships and paths that satisfy a graph pattern.

## How it behaves

MATCH removes input rows with no match. OPTIONAL MATCH preserves them with null bindings. A WHERE attached to OPTIONAL MATCH is part of that optional pattern. Use labels, properties and relationship direction to express the intended data.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
MATCH (n) RETURN count(n) AS nodes
```

Expected result: the query returns the projected rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
OPTIONAL MATCH (n:ReferenceMissingLabel) RETURN n
```

Expected result: the projected rows reflect the clause behavior described above.

## Limitations and trade-offs

Every graph query selects a project explicitly. Reads remain structurally valid during concurrent
writes, but do not promise snapshot isolation or atomic visibility across several changed records.
Use explicit ordering when result order matters.

## See also

- [Core clauses](./README.md)
- [Expressions and nulls](../../expressions.md)

