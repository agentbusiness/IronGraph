# `coalesce`

Return the first non-null value, or null when no argument supplies one. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `coalesce(value [, value ...])` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Return the first non-null value, or null when no argument supplies one.

## How it behaves

Arguments may have different types. Do not rely on this function to suppress errors in another argument.

Null arguments are skipped; if all arguments are null the result is null.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN coalesce(null, 'fallback') AS result
```

Result:

```
result
--------
fallback

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN coalesce(null, null, 42) AS result
```

Result:

```
result
------
42

1 row
```

Expected result: one row. Arguments may have different types. Do not rely on this function to suppress errors in another argument.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [Conversion functions](./README.md)
- [Expressions and null values](../../expressions.md)

