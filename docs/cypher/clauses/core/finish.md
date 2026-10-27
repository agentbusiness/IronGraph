# `FINISH`

End a query without returning graph rows. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `FINISH` |
| Relationship to standard Cypher | Standard Cypher, extended by IronGraph |
| Reference dataset | An existing example project |

## What it does

End a query without returning graph rows.

## How it behaves

FINISH ends the pipeline while preserving the effects of preceding writes. Use RETURN when the caller needs row results.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
WITH 1 AS value FINISH
```

Result:

```
No graph rows returned.
```

Expected result: the query completes without graph rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
CREATE (:ReferenceFinished {name: 'Ada'}) FINISH
```

Result:

```
No graph rows returned.
```

Expected result: the write completes without returning graph rows.

## Limitations and trade-offs

Every graph query selects a project explicitly. Reads remain structurally valid during concurrent
writes, but do not promise snapshot isolation or atomic visibility across several changed records.
Use explicit ordering when result order matters.

## See also

- [Core clauses](./README.md)
- [Expressions and nulls](../../expressions.md)

