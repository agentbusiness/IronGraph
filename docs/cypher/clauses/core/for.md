# `FOR`

Expand a list to rows using the FOR spelling. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `FOR <variable> IN <list>` |
| Relationship to standard Cypher | Standard Cypher, extended by IronGraph |
| Reference dataset | An existing example project |

## What it does

Expand a list to rows using the FOR spelling.

## How it behaves

FOR has the same row-expansion semantics as UNWIND. This is a query clause, not an imperative loop.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
FOR value IN [1, 2, 3] RETURN value
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
FOR value IN [1, 2, 3] FILTER value > 1 RETURN value
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

