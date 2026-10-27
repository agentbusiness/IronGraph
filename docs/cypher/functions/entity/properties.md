# `properties`

Return the property map of a node or relationship, or a copy of a map value. Use this function when preparing values inside a Cypher query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `properties(mapOrEntity)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Return the property map of a node or relationship, or a copy of a map value.

## How it behaves

Labels, type, endpoints and entity identity are not map properties.

A null argument produces null. Incompatible non-null values report a type error rather than silently changing the operation.

## Prerequisites

Both examples use an existing project named `worked_example`. Create it with
`CREATE PROJECT IF NOT EXISTS worked_example` before running them. No imported dataset or local
embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN properties({name: 'Ada'}) AS result
```

Result:

```
result
----------------------------------------
{"name":{"type":"string","value":"Ada"}}

1 row
```

Expected result: one row containing the function result in `result`.

## Advanced example

```cypher
USE worked_example
RETURN properties({name: 'Ada', active: true}) AS result
```

Result:

```
result
---------------------------------------------------------------------------------
{"active":{"type":"boolean","value":true},"name":{"type":"string","value":"Ada"}}

1 row
```

Expected result: one row. Labels, type, endpoints and entity identity are not map properties.

## Limitations and trade-offs

The result describes the values read by this query; it does not create an index or establish a separate schema.

## See also

- [Entity functions](./README.md)
- [Expressions and null values](../../expressions.md)

