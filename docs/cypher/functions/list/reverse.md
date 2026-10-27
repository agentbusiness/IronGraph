# `reverse`

Reverse list elements or string characters. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `reverse(listOrText)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Reverse list elements or string characters.

## How it behaves

Strings reverse by Unicode scalar values. Maps and numeric values are rejected.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN reverse([1, 2, 3]) AS result
```

Result:

```
result
---------
[3, 2, 1]

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN reverse('graph') AS result
```

Result:

```
result
------
hparg

1 row
```

Expected result: one row. Strings reverse by Unicode scalar values. Maps and numeric values are rejected.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [List functions](./README.md)
- [Expressions and null values](../../expressions.md)

