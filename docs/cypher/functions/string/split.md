# `split`

Split a string into a list using a literal separator. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `split(text, separator)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Split a string into a list using a literal separator.

## How it behaves

Empty fields are preserved. The separator is a string, not a regular expression.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN split('a,b,c', ',') AS result
```

Result:

```
result
---------
[a, b, c]

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN split('a,,b', ',') AS result
```

Result:

```
result
--------
[a, , b]

1 row
```

Expected result: one row. Empty fields are preserved. The separator is a string, not a regular expression.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [String functions](./README.md)
- [Expressions and null values](../../expressions.md)

