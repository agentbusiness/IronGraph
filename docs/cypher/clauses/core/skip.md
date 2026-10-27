# `SKIP`

Discard a specified number of ordered result rows. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `SKIP <non-negative integer>` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Discard a specified number of ordered result rows.

## How it behaves

The row count must be a non-negative integer. Use ORDER BY for reproducible pages; skipped rows are not a durable cursor.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
UNWIND [1, 2, 3] AS value RETURN value ORDER BY value SKIP 1
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
UNWIND [1, 2, 3, 4] AS value RETURN value ORDER BY value SKIP 1 LIMIT 2
```

Result:

```
value
-----
2
3

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

