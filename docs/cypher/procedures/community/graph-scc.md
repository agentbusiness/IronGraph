# `graph.scc`

> Strongly connected components: groups where every node can reach every other, following direction.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.scc() YIELD node, component` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`epinions`](../../datasets.md#epinions) — Epinions directed trust network |

## What it does

`graph.scc` assigns every node an integer component identifier such that two nodes share an identifier exactly when each can reach the other by following relationship direction. It yields one row per node. A node with no reciprocal route to anything forms a component of its own.

## How it behaves

The requirement is mutual reachability, which is far stronger than connectivity. In a directed network most nodes end up alone: they can be reached, or they can reach others, but not both. The result is usually one large mutually-reachable core plus a very large number of singletons, and the size of that core is the number worth reading.

That core is the part of the graph where influence can circulate. Outside it, everything flows one way and never returns.

## When to use it

Use it when direction carries obligation or flow and cycles matter: mutual trust, circular dependencies, feedback loops, money moving in a circle. On an undirected graph it degenerates to the same answer as `graph.wcc`.

## How it differs from its neighbours

`graph.wcc` ignores direction and produces far fewer, far larger groups. The gap between the two results is itself informative: it measures how one-way the graph is. `graph.louvain` optimises for density and will happily group nodes with no reciprocal route at all.

## Simple example

The strongly connected components of the Epinions trust network, by size. The long tail of size-one components is the expected shape.

```cypher
USE epinions
CALL graph.scc() YIELD node, component
WITH component, count(node) AS members
RETURN members AS component_size, count(*) AS components
ORDER BY component_size DESC
LIMIT 10
```

Result:

```
component_size | components
---------------+-----------
32223          | 1         
15             | 1         
9              | 2         
8              | 6         
7              | 1         
6              | 5         
5              | 24        
4              | 47        
3              | 164       
2              | 813       

10 rows, 208 ms
```

## Advanced example

How much smaller directed connectivity is than undirected connectivity on the same graph. The ratio between the largest strongly connected component and the largest weakly connected one measures how much of the network's apparent cohesion survives once direction is respected.

```cypher
USE epinions
CALL graph.scc() YIELD node, component
WITH component, count(node) AS members
WITH count(*) AS components, sum(members) AS nodes,
     max(members) AS largest,
     sum(CASE WHEN members = 1 THEN 1 ELSE 0 END) AS singletons
RETURN components, nodes, largest AS largest_strong_component, singletons,
       round(10000.0 * largest / nodes) / 100.0 AS percent_in_largest,
       round(10000.0 * singletons / components) / 100.0 AS percent_singletons
```

Result:

```
components | nodes | largest_strong_component | singletons | percent_in_largest | percent_singletons
-----------+-------+--------------------------+------------+--------------------+-------------------
42176      | 75879 | 32223                    | 41112      | 42.47              | 97.48             

1 row, 218 ms
```

## Where it earns its place

- Finding circular dependencies in a directed graph.
- Isolating the mutually reachable core where influence can circulate.
- Measuring how one-way a network is by comparing against weak components.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- Component identifiers are grouping labels only.
- On an undirected graph the result is the same as `graph.wcc` at higher cost.
- The result is dominated by singletons in most real directed graphs; aggregate rather than listing rows.

## See also

- [`graph.wcc`](./graph-wcc.md) for undirected connectivity
