# `WHERE`

Keep rows whose predicate is true. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `WHERE <predicate>` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Keep rows whose predicate is true.

## How it behaves

False and null predicates discard rows. WHERE belongs to its preceding MATCH, OPTIONAL MATCH or WITH. Use FILTER for a standalone pipeline filter.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
WITH [1, 2, 3] AS values UNWIND values AS value WITH value WHERE value > 1 RETURN value ORDER BY value
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
WITH null AS value WHERE value IS NULL RETURN value
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

