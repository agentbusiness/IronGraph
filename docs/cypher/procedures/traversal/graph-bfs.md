# `graph.bfs`

> Every node reachable from one source, with its hop distance.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.bfs(source) YIELD node, distance` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`flights`](../../datasets.md#flights) — OpenFlights airports, airlines and routes |

## What it does

`graph.bfs` expands outward from one source node along outgoing relationships, level by level, and yields each node it reaches together with the number of hops taken to reach it. The source itself is yielded at distance `0`. Nodes that cannot be reached are not yielded at all, so the row count is the size of the reachable set rather than the size of the graph.

## How it behaves

Distance counts relationships, not weight. Two airports one flight apart are at distance `1` whether that flight is 90 kilometres or 9,000. Because the expansion is breadth-first, the first time a node is reached is by a shortest hop count, and the distance yielded is final.

Direction is not optional. `graph.bfs` follows relationships in the direction they were created, so in a graph of one-way routes the reachable set from an airport is what you can fly *to*, never what can fly *in*.

## When to use it

Use `graph.bfs` to answer reachability and hop-count questions: what is within two connections of here, how far away is the furthest thing I can still get to, is that node reachable at all. It is the cheapest way to bound a neighbourhood before doing more expensive work on it.

## How it differs from its neighbours

`graph.dfs` visits the same set of nodes and yields a visit order rather than a distance; use it when the shape of the descent matters and the distance does not. `graph.shortestpath` returns one route between two named nodes instead of the whole reachable set. `graph.dijkstra` answers the same question as `graph.bfs` but measures cost with a relationship property instead of hops.

## Simple example

How far the rest of the world is from London Heathrow, counted in flights. The source argument is a bound node or a non-negative stable node identifier. A node that is not visible under the query's selected layers is rejected rather than silently skipped.

```cypher
USE flights
MATCH (origin:Airport {iata: 'LHR'})
CALL graph.bfs(origin) YIELD node, distance
RETURN distance, count(node) AS airports
ORDER BY distance
```

Result:

```
distance | airports
---------+---------
0        | 1       
1        | 170     
2        | 1773    
3        | 924     
4        | 240     
5        | 48      
6        | 8       
7        | 2       

8 rows, 148 ms
```

## Advanced example

The airports that are exactly three flights from Heathrow and cannot be reached in fewer. Because breadth-first distance is final on first arrival, filtering on it is enough to express "no shorter route exists".

```cypher
USE flights
MATCH (origin:Airport {iata: 'LHR'})
CALL graph.bfs(origin) YIELD node, distance
WITH node, distance WHERE distance = 3
RETURN node.iata AS iata, node.name AS airport, node.country AS country
ORDER BY country, iata
LIMIT 10
```

Result:

```
iata | airport                                | country       
-----+----------------------------------------+---------------
BMW  | Bordj Badji Mokhtar Airport            | Algeria       
CBH  | Béchar Boudghene Ben Ali Lotfi Airport | Algeria       
ELG  | El Golea Airport                       | Algeria       
IAM  | In Aménas Airport                      | Algeria       
TMR  | Aguenar – Hadj Bey Akhamok Airport     | Algeria       
PPG  | Pago Pago International Airport        | American Samoa
AXA  | Clayton J Lloyd International Airport  | Anguilla      
AFA  | Suboficial Ay Santiago Germano Airport | Argentina     
BHI  | Comandante Espora Airport              | Argentina     
CPC  | Aviador C. Campos Airport              | Argentina     

10 rows
```

## Where it earns its place

- Bounding a neighbourhood before running something expensive over it.
- Answering "how many connections away" questions without materialising paths.
- Finding the reachable set from a node to test whether the graph is connected in the direction you care about.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- Only outgoing relationships are followed. To expand both ways, keep the reciprocal relationship in the graph.
- Hop distance ignores relationship properties entirely. Reach for `graph.dijkstra` when the cost of an edge matters.

## See also

- [`graph.dfs`](./graph-dfs.md) for visit order rather than distance
- [`graph.dijkstra`](../shortest-path/graph-dijkstra.md) for weighted cost
- [`graph.shortestpath`](../shortest-path/graph-shortestpath.md) for one route
