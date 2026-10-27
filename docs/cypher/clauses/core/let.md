# `LET`

Add expression bindings while retaining the current row scope. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `LET <name> = <expression> [, <name> = <expression> ...]` |
| Relationship to standard Cypher | Standard Cypher, extended by IronGraph |
| Reference dataset | An existing example project |

## What it does

Add expression bindings while retaining the current row scope.

## How it behaves

Bindings preserve existing variables, unlike a WITH projection that drops variables not named. A binding cannot redeclare a variable already in scope.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
WITH 3 AS value LET doubled = value * 2 RETURN value, doubled
```

Result:

```
value | doubled
------+--------
3     | 6

1 row
```

Expected result: the query returns the projected rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
WITH 3 AS value LET doubled = value * 2, text = toString(value) RETURN doubled, text
```

Result:

```
doubled | text
--------+-----
6       | 3

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

