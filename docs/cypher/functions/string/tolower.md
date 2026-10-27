# `toLower`

Convert a string to lowercase. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `toLower(text)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Convert a string to lowercase.

## How it behaves

This supported spelling has the same behavior as lower.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN toLower('GRAPH') AS result
```

Result:

```
result
------
graph

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN toLower('Graph Database') AS result
```

Result:

```
result
--------------
graph database

1 row
```

Expected result: one row. This supported spelling has the same behavior as lower.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [String functions](./README.md)
- [Expressions and null values](../../expressions.md)

