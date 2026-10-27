# `UNION / UNION ALL`

Combine result rows from query branches with the same column names. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `<query> UNION [ALL] <query>` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Combine result rows from query branches with the same column names.

## How it behaves

UNION removes duplicate rows. UNION ALL preserves them. Branch column names must agree. Do not mix UNION and UNION ALL in one query.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN 1 AS value UNION RETURN 1 AS value
```

Result:

```
value
-----
1

1 row
```

Expected result: the query returns the projected rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
RETURN 1 AS value UNION ALL RETURN 1 AS value
```

Result:

```
value
-----
1
1

2 rows
```

Expected result: two rows, both containing value 1.

## Limitations and trade-offs

Every graph query selects a project explicitly. Reads remain structurally valid during concurrent
writes, but do not promise snapshot isolation or atomic visibility across several changed records.
Use explicit ordering when result order matters.

## See also

- [Core clauses](./README.md)
- [Expressions and nulls](../../expressions.md)

