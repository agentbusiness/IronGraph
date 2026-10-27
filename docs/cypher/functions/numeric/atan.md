# `atan`

Return the inverse tangent in radians. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `atan(number)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Return the inverse tangent in radians.

## How it behaves

Use atan2 when you need the quadrant determined by two coordinates.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN atan(0.0) AS result
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
RETURN degrees(atan(1.0)) AS result
```

Result:

```
result
------
45

1 row
```

Expected result: one row. Use atan2 when you need the quadrant determined by two coordinates.

## Limitations and trade-offs

Floating-point arithmetic can round results. Check the mathematical domain before applying the function to untrusted values.

## See also

- [Numeric functions](./README.md)
- [Expressions and null values](../../expressions.md)

