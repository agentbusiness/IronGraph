# `abs`

Return the non-negative magnitude of a number. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `abs(number)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Return the non-negative magnitude of a number.

## How it behaves

Integer inputs preserve integer results; the minimum integer cannot be negated and reports overflow.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN abs(-7) AS result
```

Result:

```
result
------
7

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN abs(-3.5) AS result
```

Result:

```
result
------
3.5

1 row
```

Expected result: one row. Integer inputs preserve integer results; the minimum integer cannot be negated and reports overflow.

## Limitations and trade-offs

Floating-point arithmetic can round results. Check the mathematical domain before applying the function to untrusted values.

## See also

- [Numeric functions](./README.md)
- [Expressions and null values](../../expressions.md)

