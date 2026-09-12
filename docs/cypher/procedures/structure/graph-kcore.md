# `graph.kcore`

> How deep into the graph's densely connected interior each node survives.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.kcore() YIELD node, core` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`social`](../../datasets.md#social) — Facebook combined ego networks |

## What it does

`graph.kcore` yields one row per node with its core number: the largest `k` for which the node belongs to a subgraph where every node has at least `k` neighbours. It is computed by repeatedly removing the least connected nodes and recording when each one falls away.

## How it behaves

The peeling is what makes the number meaningful. A node with a hundred relationships to nodes that have one each is removed early and gets a low core number, because its neighbours do not survive. A node with ten relationships to nodes that also have ten survives deep. Core number therefore measures the density of the region a node sits in, not the node's own count.

The highest core number in a graph is a property of the graph itself, and the nodes that reach it form its densest interior.

## When to use it

Use it to find the resilient centre of a network, or to strip away a periphery cheaply. Filtering to nodes above a core threshold is one of the most effective ways to reduce a large graph before an expensive algorithm, because it removes the sparse edges without breaking the dense middle.

## How it differs from its neighbours

`graph.degree` counts a node's own relationships and is fooled by hubs attached to nothing. `graph.clusteringcoefficient` looks only at a node's immediate neighbours. `graph.kcore` is the one that accounts for the connectedness of the neighbours' neighbours, through peeling.

## Simple example

How the friendship network's population is distributed across core depths.

```cypher
USE social
CALL graph.kcore() YIELD node, core
RETURN core, count(node) AS people
ORDER BY core DESC
LIMIT 10
```

Result:

```
core | people
-----+-------
115  | 158   
114  | 7     
113  | 2     
112  | 3     
111  | 4     
109  | 2     
108  | 1     
107  | 1     
106  | 2     
105  | 1     

10 rows
```

## Advanced example

Where core depth and raw degree disagree. This reports, for each core depth, the range of degrees found there — showing that a high relationship count does not put a node in the dense interior, because a node is only as deep as the company it keeps.

```cypher
USE social
CALL graph.kcore() YIELD node, core
MATCH (node)-[friendship:FRIEND]-()
WITH core, node, count(friendship) AS degree
RETURN core,
       count(node) AS people,
       min(degree) AS lowest_degree,
       round(avg(degree) * 10) / 10.0 AS mean_degree,
       max(degree) AS highest_degree
ORDER BY core DESC
LIMIT 10
```

Result:

```
core | people | lowest_degree | mean_degree | highest_degree
-----+--------+---------------+-------------+---------------
115  | 158    | 130           | 179.7       | 755           
114  | 7      | 124           | 132         | 141           
113  | 2      | 126           | 128.5       | 131           
112  | 3      | 123           | 125         | 128           
111  | 4      | 117           | 126.5       | 141           
109  | 2      | 116           | 120         | 124           
108  | 1      | 113           | 113         | 113           
107  | 1      | 113           | 113         | 113           
106  | 2      | 116           | 116.5       | 117           
105  | 1      | 112           | 112         | 112           

10 rows, 279 ms
```

## Where it earns its place

- Extracting the resilient core of a network.
- Cheaply reducing a large graph before an expensive algorithm.
- Distinguishing genuinely central nodes from hubs attached to a sparse fringe.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- The core number is coarse. Many nodes share a value, so it groups rather than ranks.
- Direction is ignored; a node's core number is computed from its combined relationships.

## See also

- [`graph.degree`](../centrality/graph-degree.md) for the local count it corrects
- [`graph.louvain`](../community/graph-louvain.md) for grouping rather than depth
