# `ORDER BY`

Order projected rows by one or more keys. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `ORDER BY <expression> [ASC\|DESC] [, ...]` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Order projected rows by one or more keys.

## How it behaves

Ascending is the default. Add a stable tie-break key when paging. ORDER BY follows WITH or RETURN; it does not change the graph.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
UNWIND [3, 1, 2] AS value RETURN value ORDER BY value
```

Result:

```
value
-----
1
2
3

3 rows
```

Expected result: the query returns the projected rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
UNWIND [{name: 'B', score: 2}, {name: 'A', score: 2}] AS item RETURN item.name AS name, item.score AS score ORDER BY score DESC, name
```

Result:

```
name | score
-----+------
A    | 2
B    | 2

2 rows
```

Expected result: the projected rows reflect the clause behavior described above.

## Limitations and trade-offs

Every graph query selects a project explicitly. Reads remain structurally valid during concurrent
writes, but do not promise snapshot isolation or atomic visibility across several changed records.
Use explicit ordering when result order matters.

## See also

- [Core clauses](./README.md)
- [Expressions and nulls](../../expressions.md)

