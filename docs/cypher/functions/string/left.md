# `left`

Return at most count characters from the beginning of a string. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `left(text, count)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Return at most count characters from the beginning of a string.

## How it behaves

Count must be a non-negative integer. Character counts are Unicode scalar values, not bytes.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN left('graph', 2) AS result
```

Result:

```
result
------
gr

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN left('graph', 99) AS result
```

Result:

```
result
------
graph

1 row
```

Expected result: one row. Count must be a non-negative integer. Character counts are Unicode scalar values, not bytes.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [String functions](./README.md)
- [Expressions and null values](../../expressions.md)

