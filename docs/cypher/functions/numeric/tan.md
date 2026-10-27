# `tan`

Return the tangent of an angle in radians. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `tan(radians)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Return the tangent of an angle in radians.

## How it behaves

Values near odd multiples of pi/2 are numerically unstable.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN tan(0.0) AS result
```

Result:

```
result
------
0

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN tan(radians(45.0)) AS result
```

Result:

```
result
------
1

1 row
```

Expected result: one row. Values near odd multiples of pi/2 are numerically unstable.

## Limitations and trade-offs

Floating-point arithmetic can round results. Check the mathematical domain before applying the function to untrusted values.

## See also

- [Numeric functions](./README.md)
- [Expressions and null values](../../expressions.md)

