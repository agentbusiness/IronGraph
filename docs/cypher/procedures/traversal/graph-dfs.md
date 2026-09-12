# `graph.dfs`

> Every node reachable from one source, in depth-first visit order.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.dfs(source) YIELD node, order` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`flights`](../../datasets.md#flights) — OpenFlights airports, airlines and routes |

## What it does

`graph.dfs` explores as far as it can along one branch before backtracking, and yields each reached node with the position at which it was visited. The source is visited at order `0`. As with `graph.bfs`, unreachable nodes are not yielded, so the row count is the size of the reachable set.

## How it behaves

`order` is a visit sequence, not a distance and not a ranking. Two nodes with adjacent order values are adjacent in the descent, which usually means one is the other's neighbour, but a node reached late can still be a direct neighbour of the source. Never read `order` as "how far away".

The order depends on the order relationships are stored in, so it is stable for a given graph state and changes when the graph changes. Treat it as a deterministic traversal trace rather than a property of the data.

## When to use it

Use `graph.dfs` when the question is about the shape of a descent rather than the distance to a node: tracing one dependency chain to its end, or walking a reachable set in an order where a branch is finished before the next begins.

## How it differs from its neighbours

`graph.bfs` visits the identical set of nodes and yields a hop distance, which is the more useful number in almost every reachability question. Prefer `graph.bfs` unless the descent order is specifically what you want.

## Simple example

The first ten airports a depth-first descent from Heathrow visits.

```cypher
USE flights
MATCH (origin:Airport {iata: 'LHR'})
CALL graph.dfs(origin) YIELD node, order
RETURN order, node.iata AS iata, node.name AS airport
ORDER BY order
LIMIT 10
```

Result:

```
order | iata | airport                                
------+------+----------------------------------------
0     | LHR  | London Heathrow Airport                
1     | KEF  | Keflavik International Airport         
2     | GOH  | Godthaab / Nuuk Airport                
3     | UAK  | Narsarsuaq Airport                     
4     | SFJ  | Kangerlussuaq Airport                  
5     | CPH  | Copenhagen Kastrup Airport             
6     | YYZ  | Lester B. Pearson International Airport
7     | YAM  | Sault Ste Marie Airport                
8     | YQT  | Thunder Bay Airport                    
9     | YTS  | Timmins/Victor M. Power                

10 rows
```

## Advanced example

Depth-first order set against breadth-first distance for the same source. The two disagree sharply, which is the point: a node visited late in the descent can be one hop away.

```cypher
USE flights
MATCH (origin:Airport {iata: 'LHR'})
CALL graph.bfs(origin) YIELD node, distance
WITH origin, node AS reached, distance
WHERE distance = 1
WITH origin, collect(reached.iata) AS neighbours
CALL graph.dfs(origin) YIELD node, order
WITH neighbours, node, order WHERE node.iata IN neighbours
RETURN min(order) AS first_neighbour_visited,
       max(order) AS last_neighbour_visited,
       count(node) AS direct_neighbours
```

Result:

```
first_neighbour_visited | last_neighbour_visited | direct_neighbours
------------------------+------------------------+------------------
1                       | 2995                   | 169              

1 row
```

## Where it earns its place

- Tracing one chain of dependencies to its end.
- Producing a deterministic walk of a reachable set for diffing or replay.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- `order` is not a distance and not a rank. Reading it as one is the most common mistake with this procedure.
- Only outgoing relationships are followed.

## See also

- [`graph.bfs`](./graph-bfs.md) for hop distance
