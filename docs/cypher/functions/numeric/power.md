# `power`

Raise a numeric base to a numeric exponent. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `power(base, exponent)` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | Self-contained values in an existing project |

## What it does

Raise a numeric base to a numeric exponent.

## How it behaves

Both arguments are numeric. Fractional powers use floating-point arithmetic.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN power(2, 3) AS result
```

Result:

```
result
------
8

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN power(9.0, 0.5) AS result
```

Result:

```
result
------
3

1 row
```

Expected result: one row. Both arguments are numeric. Fractional powers use floating-point arithmetic.

## Limitations and trade-offs

Floating-point arithmetic can round results. Check the mathematical domain before applying the function to untrusted values.

## See also

- [Numeric functions](./README.md)
- [Expressions and null values](../../expressions.md)

