# `keys`

Return property names from a map, node or relationship. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `keys(mapOrEntity)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Return property names from a map, node or relationship.

## How it behaves

The result is a list of strings. Do not use property-key order as application ordering.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN keys({name: 'Ada', active: true}) AS result
```

Result:

```
result
--------------
[active, name]

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN keys({}) AS result
```

Result:

```
result
------
[]

1 row
```

Expected result: one row. The result is a list of strings. Do not use property-key order as application ordering.

## Limitations and trade-offs

The result describes the values read by this query; it does not create an index or establish a separate schema.

## See also

- [Entity functions](./README.md)
- [Expressions and null values](../../expressions.md)

