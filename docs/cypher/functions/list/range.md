# `range`

Build an inclusive sequence of integers. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `range(start, end [, step])` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Build an inclusive sequence of integers.

## How it behaves

Step defaults to 1 and cannot be zero. An incompatible step direction produces an empty list.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN range(1, 3) AS result
```

Result:

```
result
---------
[1, 2, 3]

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN range(5, 1, -2) AS result
```

Result:

```
result
---------
[5, 3, 1]

1 row
```

Expected result: one row. Step defaults to 1 and cannot be zero. An incompatible step direction produces an empty list.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [List functions](./README.md)
- [Expressions and null values](../../expressions.md)

