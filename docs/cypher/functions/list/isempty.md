# `isEmpty`

Test whether a list, map or string contains no entries or characters. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `isEmpty(listOrMapOrText)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Test whether a list, map or string contains no entries or characters.

## How it behaves

Null is unknown, not empty: isEmpty(null) returns null.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN isEmpty([]) AS result
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
RETURN isEmpty({name: 'Ada'}) AS result
```

Result:

```
result
------
false

1 row
```

Expected result: one row. Null is unknown, not empty: isEmpty(null) returns null.

## Limitations and trade-offs

The function operates on each input row. Use an explicit LIMIT or selective graph pattern when the query should process fewer rows.

## See also

- [List functions](./README.md)
- [Expressions and null values](../../expressions.md)

