# `replace`

Replace occurrences of a substring. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `replace(text, search, replacement)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Replace occurrences of a substring.

## How it behaves

Matching is case-sensitive and literal; search is not a regular expression.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN replace('graph data', 'data', 'query') AS result
```

Result:

```
result
-----------
graph query

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN replace('a-a-a', '-', '/') AS result
```

Result:

```
result
------
a/a/a

1 row
```

Expected result: one row. Matching is case-sensitive and literal; search is not a regular expression.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [String functions](./README.md)
- [Expressions and null values](../../expressions.md)

