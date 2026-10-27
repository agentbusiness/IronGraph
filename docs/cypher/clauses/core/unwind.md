# `UNWIND`

Expand a list to one row per element. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `UNWIND <list> AS <variable>` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Expand a list to one row per element.

## How it behaves

An empty list or null supplies no rows. Null elements remain individual rows. Apply ORDER BY if row order matters.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
UNWIND [1, 2, 3] AS value RETURN value
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
UNWIND [1, null, 3] AS value RETURN count(*) AS rows, count(value) AS non_null
```

Result:

```
rows | non_null
-----+---------
3    | 2

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

