# `LIMIT`

Keep at most a specified number of result rows. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `LIMIT <non-negative integer>` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Keep at most a specified number of result rows.

## How it behaves

LIMIT is explicit query semantics. It does not set a database quota or configure a timeout. A read LIMIT is not a safeguard for an earlier write in the same pipeline.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
UNWIND [3, 1, 2] AS value RETURN value ORDER BY value LIMIT 2
```

Result:

```
value
-----
1
2

2 rows
```

Expected result: the query returns the projected rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
UNWIND [1, 2, 3] AS value WITH value ORDER BY value DESC LIMIT 1 RETURN value
```

Result:

```
value
-----
3

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

