# `FILTER`

Filter the current rows without adding a projection. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `FILTER <predicate>` |
| Relationship to standard Cypher | Standard Cypher, extended by IronGraph |
| Reference dataset | An existing example project |

## What it does

Filter the current rows without adding a projection.

## How it behaves

Only true passes; false and null do not. FILTER is useful after FOR or LET when a standalone filter reads more clearly than WITH ... WHERE.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
UNWIND [1, 2, 3] AS value FILTER value > 1 RETURN value
```

Result:

```
value
-----
2
3

2 rows
```

Expected result: the query returns the projected rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
WITH null AS value FILTER value IS NULL RETURN value
```

Result:

```
value
-----
null

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

