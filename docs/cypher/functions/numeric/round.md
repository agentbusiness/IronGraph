# `round`

Round to the nearest integral value, with halfway cases away from zero. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `round(number)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Round to the nearest integral value, with halfway cases away from zero.

## How it behaves

IronGraph accepts one argument. Precision and rounding-mode arguments are not supported.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN round(1.5) AS result
```

Result:

```
result
------
2

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN round(-1.5) AS result
```

Result:

```
result
------
-2

1 row
```

Expected result: one row. IronGraph accepts one argument. Precision and rounding-mode arguments are not supported.

## Limitations and trade-offs

Floating-point arithmetic can round results. Check the mathematical domain before applying the function to untrusted values.

## See also

- [Numeric functions](./README.md)
- [Expressions and null values](../../expressions.md)

