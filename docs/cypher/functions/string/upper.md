# `upper`

Convert a string to uppercase. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `upper(text)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Convert a string to uppercase.

## How it behaves

Unicode case conversion can change the number of characters.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN upper('graph') AS result
```

Result:

```
result
------
GRAPH

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN upper('Graph Database') AS result
```

Result:

```
result
--------------
GRAPH DATABASE

1 row
```

Expected result: one row. Unicode case conversion can change the number of characters.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [String functions](./README.md)
- [Expressions and null values](../../expressions.md)

