# `length`

Count relationships in a path, or entries in a list, map or string. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `length(pathOrListOrMapOrText)` |
| Relationship to standard Cypher | Standard Cypher, extended by IronGraph |
| Reference dataset | Self-contained values in an existing project |

## What it does

Count relationships in a path, or entries in a list, map or string.

## How it behaves

For a path, length counts relationships rather than nodes.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN length([1, 2, 3]) AS result
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
RETURN length('graph') AS result
```

Result:

```
result
------
5

1 row
```

Expected result: one row. For a path, length counts relationships rather than nodes.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [List functions](./README.md)
- [Expressions and null values](../../expressions.md)

