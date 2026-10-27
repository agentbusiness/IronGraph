# `toBoolean`

Convert a boolean or a boolean string to a boolean. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `toBoolean(value)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Convert a boolean or a boolean string to a boolean.

## How it behaves

Unrecognized strings return null; unsupported input types are errors.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN toBoolean('true') AS result
```

Result:

```
result
------
true

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN toBoolean('not a boolean') AS result
```

Result:

```
result
------
null

1 row
```

Expected result: one row. Unrecognized strings return null; unsupported input types are errors.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [Conversion functions](./README.md)
- [Expressions and null values](../../expressions.md)

