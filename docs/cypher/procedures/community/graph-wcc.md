# `graph.wcc`

> Weakly connected components: the islands of the graph, ignoring direction.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.wcc() YIELD node, component` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`epinions`](../../datasets.md#epinions) — Epinions directed trust network |

## What it does

`graph.wcc` assigns every node an integer component identifier such that two nodes share an identifier exactly when a path connects them if relationship direction is ignored. It yields one row per node. The identifiers themselves carry no meaning beyond grouping — only equality between them does.

## How it behaves

Weak connectivity is a fact about the graph, not an estimate. Run it twice on unchanged data and the grouping is identical, although the numbering is not something to depend on.

Almost every real network has one component holding the large majority of nodes and a long tail of tiny ones. That shape is the useful output: the size of the largest component tells you how much of the graph is actually one connected object, and everything outside it is unreachable from everything inside it.

## When to use it

Run it early, before anything expensive. Algorithms that assume connectivity produce misleading output when the graph is really several disconnected pieces, and this is the cheapest way to find that out.

## How it differs from its neighbours

`graph.scc` requires a path in *both* directions and therefore cuts the same graph much more finely. `graph.louvain` does not answer a connectivity question at all: it looks for densely connected groups *within* what is already connected.

## Simple example

How the Epinions network divides into disconnected islands.

```cypher
USE epinions
CALL graph.wcc() YIELD node, component
WITH component, count(node) AS members
RETURN members AS component_size, count(*) AS components
ORDER BY component_size DESC
LIMIT 10
```

Result:

```
component_size | components
---------------+-----------
75877          | 1         
2              | 1         

2 rows, 147 ms
```

## Advanced example

The share of the graph held by its largest component, computed in one query. A number close to 100 means connectivity questions can be treated as global; a much lower one means every later algorithm is really being run over several unrelated graphs at once.

```cypher
USE epinions
CALL graph.wcc() YIELD node, component
WITH component, count(node) AS members
RETURN count(*) AS components,
       sum(members) AS nodes,
       max(members) AS largest_component,
       min(members) AS smallest_component,
       round(10000.0 * max(members) / sum(members)) / 100.0 AS percent_in_largest
```

Result:

```
components | nodes | largest_component | smallest_component | percent_in_largest
-----------+-------+-------------------+--------------------+-------------------
2          | 75879 | 75877             | 2                  | 100               

1 row, 150 ms
```

## Where it earns its place

- Checking that a graph is one object before trusting a global measurement.
- Separating a main network from imported fragments and orphans.
- Sizing the reachable universe a later algorithm will actually operate on.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- Component identifiers are grouping labels. Do not store them as stable identity or compare them across runs.
- Direction is discarded. A component says two nodes are connected somehow, not that either can reach the other.

## See also

- [`graph.scc`](./graph-scc.md) for directed connectivity
- [`graph.louvain`](./graph-louvain.md) for density rather than connectivity
