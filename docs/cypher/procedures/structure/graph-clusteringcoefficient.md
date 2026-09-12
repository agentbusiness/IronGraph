# `graph.clusteringcoefficient`

> Per node, the share of its neighbours that are connected to each other.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.clusteringcoefficient() YIELD node, coefficient` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`social`](../../datasets.md#social) — Facebook combined ego networks |

## What it does

`graph.clusteringcoefficient` yields one row per node with a coefficient between `0` and `1`: the fraction of the possible connections among that node's neighbours that actually exist. A node whose neighbours all know each other scores `1`; a node at the centre of a star scores `0`.

## How it behaves

Because it is normalised by degree, the coefficient is comparable across nodes of very different sizes — which is exactly what a raw triangle count is not. That normalisation has a sharp edge: a node with one neighbour has no possible connections among its neighbours and scores `0`, which looks identical to a genuinely unclustered hub. Filter by degree before ranking on this value.

High coefficient and high degree together is the rare and interesting combination: a node with many neighbours who nonetheless mostly know each other sits inside a dense community rather than bridging between them.

## When to use it

Use it to distinguish nodes embedded in a tight group from nodes that connect otherwise separate groups. The low-coefficient, high-degree nodes are the bridges, and they are usually the ones worth looking at.

## How it differs from its neighbours

`graph.trianglecount` gives one number for the whole graph and cannot say anything about individual nodes. `graph.kcore` measures cohesion by asking how deep a node survives repeated peeling, which finds dense regions rather than dense neighbourhoods.

## Simple example

The most tightly embedded people in the friendship network, restricted to those with enough neighbours for the coefficient to mean something.

```cypher
USE social
MATCH (person:Person)-[friendship:FRIEND]-()
WITH person, count(friendship) AS degree WHERE degree >= 50
WITH collect(person.person_id) AS dense
CALL graph.clusteringcoefficient() YIELD node, coefficient
WITH dense, node, coefficient WHERE node.person_id IN dense
RETURN node.person_id AS person,
       round(coefficient * 10000) / 10000.0 AS coefficient
ORDER BY coefficient DESC, person
LIMIT 10
```

Result:

```
person | coefficient
-------+------------
2606   | 0.9013     
2554   | 0.8953     
2532   | 0.8916     
2306   | 0.8884     
2579   | 0.8835     
2046   | 0.8807     
2591   | 0.8739     
2418   | 0.8724     
2469   | 0.8715     
1963   | 0.8687     

10 rows, 544 ms
```

## Advanced example

The bridges. These are the people with many connections whose connections do not know each other — the opposite of the previous example, and the structurally significant one: remove a bridge and otherwise separate parts of the network lose their link.

```cypher
USE social
MATCH (person:Person)-[friendship:FRIEND]-()
WITH person, count(friendship) AS degree WHERE degree >= 50
WITH collect(person.person_id) AS dense
CALL graph.clusteringcoefficient() YIELD node, coefficient
WITH dense, node, coefficient WHERE node.person_id IN dense
MATCH (node)-[friendship:FRIEND]-()
WITH node, coefficient, count(friendship) AS connections
RETURN node.person_id AS person, connections,
       round(coefficient * 10000) / 10000.0 AS coefficient,
       round(connections * (1 - coefficient)) AS unclustered_connections
ORDER BY coefficient, person
LIMIT 10
```

Result:

```
person | connections | coefficient | unclustered_connections
-------+-------------+-------------+------------------------
3437   | 547         | 0.0322      | 529                    
0      | 347         | 0.042       | 332                    
1684   | 792         | 0.0448      | 757                    
107    | 1045        | 0.049       | 994                    
3980   | 59          | 0.0853      | 54                     
1912   | 755         | 0.1055      | 675                    
686    | 170         | 0.1156      | 150                    
1505   | 59          | 0.1204      | 52                     
348    | 229         | 0.123       | 201                    
698    | 68          | 0.1313      | 59                     

10 rows, 3279 ms
```

## Where it earns its place

- Separating nodes inside a community from nodes bridging between communities.
- Finding structurally critical nodes whose removal would disconnect groups.
- Comparing local cohesion across nodes of very different degree.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- A node with fewer than two neighbours scores `0` for want of any possible connection, not because it is unclustered. Filter by degree first.
- The coefficient describes a node's immediate neighbourhood only and says nothing about the wider graph.

## See also

- [`graph.trianglecount`](./graph-trianglecount.md) for the graph-wide total
- [`graph.kcore`](./graph-kcore.md) for cohesion by depth
