# `toInteger`

Convert a boolean, number or numeric string to an integer. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `toInteger(value)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Convert a boolean, number or numeric string to an integer.

## How it behaves

Floating-point input truncates toward zero. Unparseable strings return null; unsupported types are errors.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN toInteger('42') AS result
```

Result:

```
result
------
42

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN toInteger(3.9) AS result
```

Result:

```
result
------
3

1 row
```

Expected result: one row. Floating-point input truncates toward zero. Unparseable strings return null; unsupported types are errors.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [Conversion functions](./README.md)
- [Expressions and null values](../../expressions.md)

