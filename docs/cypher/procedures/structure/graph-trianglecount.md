# `graph.trianglecount`

> One number: how many closed triangles the whole graph contains.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.trianglecount() YIELD triangleCount` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`social`](../../datasets.md#social) — Facebook combined ego networks |

## What it does

`graph.trianglecount` yields a single row with the total number of triangles in the graph — sets of three nodes each connected to the other two. It is the only procedure here that produces one row rather than one row per node.

## How it behaves

A triangle is the smallest possible evidence of clustering: it means two of a node's neighbours are themselves connected. A graph rich in triangles has genuine communities; a graph with almost none is a tree, a star, or a chain, whatever its size.

The count on its own is hard to interpret, because it grows steeply with degree. It becomes meaningful when compared: against another graph of similar size, against the same graph at another time, or against the relationship count as a crude density ratio.

## When to use it

Use it as a single summary statistic for how clustered a graph is — a quick check before deciding whether community detection has anything to find.

## How it differs from its neighbours

`graph.clusteringcoefficient` computes the same underlying idea per node and normalises it, which makes individual nodes comparable. This procedure gives the graph-wide total and nothing else.

## Simple example

The triangle count of a dense friendship network. Friendship graphs are triangle-rich because friends of friends are frequently friends.

```cypher
USE social
CALL graph.trianglecount() YIELD triangleCount
RETURN triangleCount
```

Result:

```
triangleCount
-------------
1612010      

1 row
```

## Advanced example

Triangle density compared across two graphs of very different character: a friendship network, where mutual connection is the norm, and a citation network, where it is nearly impossible because papers cite backwards in time. Normalising by relationship count makes the two comparable.

```cypher
USE social
CALL graph.trianglecount() YIELD triangleCount
MATCH ()-[relationship]->()
RETURN 'social' AS graph,
       triangleCount AS triangles,
       count(relationship) AS relationships,
       round(1000.0 * triangleCount / count(relationship) * 100) / 100.0
         AS triangles_per_1000_relationships
```

Result:

```
graph  | triangles | relationships | triangles_per_1000_relationships
-------+-----------+---------------+---------------------------------
social | 1612010   | 88234         | 18269.71                        

1 row, 193 ms
```

## Where it earns its place

- A one-number summary of how clustered a graph is.
- Deciding whether community detection is worth running.
- Tracking clustering of one graph over time.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- A bare count is not comparable between graphs of different sizes. Normalise before comparing.
- Triangle counting is quadratic in the degree of the densest nodes and is the most expensive of the structure procedures on a hub-heavy graph.

## See also

- [`graph.clusteringcoefficient`](./graph-clusteringcoefficient.md) for the per-node, normalised form
