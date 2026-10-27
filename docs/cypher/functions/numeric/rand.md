# `rand`

Return a new pseudo-random floating-point value in the interval [0, 1). Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `rand()` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Return a new pseudo-random floating-point value in the interval [0, 1).

## How it behaves

The output varies between evaluations. Do not depend on reproducible values or use it as a cryptographic secret.

No arguments are accepted.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN rand() AS result
```

Expected result: one row containing the random value in `result`.

## Advanced example

```cypher
USE worked_example
RETURN rand() >= 0.0 AND rand() < 1.0 AS result
```

Expected result: one row. The output varies between evaluations. Do not depend on reproducible values or use it as a cryptographic secret.

## Limitations and trade-offs

Floating-point arithmetic can round results. Check the mathematical domain before applying the function to untrusted values.

## See also

- [Numeric functions](./README.md)
- [Expressions and null values](../../expressions.md)

