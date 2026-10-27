# `substring`

Return a substring beginning at a zero-based character offset. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `substring(text, start [, length])` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Return a substring beginning at a zero-based character offset.

## How it behaves

Start and optional length must be non-negative integers. An offset beyond the end returns an empty string.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN substring('graph', 1, 3) AS result
```

Result:

```
result
------
rap

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN substring('graph', 2) AS result
```

Result:

```
result
------
aph

1 row
```

Expected result: one row. Start and optional length must be non-negative integers. An offset beyond the end returns an empty string.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [String functions](./README.md)
- [Expressions and null values](../../expressions.md)

