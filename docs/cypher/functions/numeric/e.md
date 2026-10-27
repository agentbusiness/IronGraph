# `e`

Return the base of natural logarithms. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `e()` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Return the base of natural logarithms.

## How it behaves

This function takes no arguments.

No arguments are accepted.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN e() AS result
```

Result:

```
result
--------
2.718282

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN log(e()) AS result
```

Result:

```
result
------
1

1 row
```

Expected result: one row. This function takes no arguments.

## Limitations and trade-offs

Floating-point arithmetic can round results. Check the mathematical domain before applying the function to untrusted values.

## See also

- [Numeric functions](./README.md)
- [Expressions and null values](../../expressions.md)

