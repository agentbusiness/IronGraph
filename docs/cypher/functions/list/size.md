# `size`

Count list elements, map entries or string characters. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `size(listOrMapOrText)` |
| Relationship to standard Cypher | Standard Cypher, extended by IronGraph |
| Reference dataset | Self-contained values in an existing project |

## What it does

Count list elements, map entries or string characters.

## How it behaves

Null list elements still count. Strings are counted in Unicode scalar values.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN size([1, null, 3]) AS result
```

Result:

```
result
------
3

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN size('café') AS result
```

Result:

```
result
------
4

1 row
```

Expected result: one row. Null list elements still count. Strings are counted in Unicode scalar values.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [List functions](./README.md)
- [Expressions and null values](../../expressions.md)

