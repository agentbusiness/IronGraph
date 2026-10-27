# Entity functions

Use these scalar functions to transform values within a query. Every page states accepted arguments,
null behavior, limits and two self-contained examples.

| Function | Signature | Purpose |
| --- | --- | --- |
| [`labels`](./labels.md) | `labels(node)` | Return the labels carried by a node. |
| [`type`](./type.md) | `type(relationship)` | Return a relationship's type name. |
| [`id`](./id.md) | `id(entity)` | Return the stable identity of a node or relationship within its project. |
| [`revision`](./revision.md) | `revision(entity)` | Return the current revision of a node or relationship. |
| [`nodes`](./nodes.md) | `nodes(path)` | Return a path's nodes in traversal order. |
| [`relationships`](./relationships.md) | `relationships(path)` | Return a path's relationships in traversal order. |
| [`startNode`](./startnode.md) | `startNode(relationship)` | Return the source node of a relationship. |
| [`endNode`](./endnode.md) | `endNode(relationship)` | Return the target node of a relationship. |
| [`keys`](./keys.md) | `keys(mapOrEntity)` | Return property names from a map, node or relationship. |
| [`properties`](./properties.md) | `properties(mapOrEntity)` | Return the property map of a node or relationship, or a copy of a map value. |

