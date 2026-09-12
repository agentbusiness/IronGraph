# `graph.shortestpath`

> One path with the fewest relationships between two nodes, and its length.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.shortestpath(source, target) YIELD path, cost` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`flights`](../../datasets.md#flights) — OpenFlights airports, airlines and routes |

## What it does

`graph.shortestpath` finds a route from `source` to `target` following outgoing relationships and yields it as a path value together with `cost`, the number of relationships on it. When no route exists the procedure yields no rows at all, which is how absence is reported.

## How it behaves

`cost` here is a hop count, not a weight. Several routes may tie on hop count; the procedure yields one of them, chosen deterministically for a given graph state, not all of them.

The yielded `path` is an ordinary Cypher path, so `nodes(path)`, `relationships(path)` and `length(path)` all apply to it and can be projected in the same query. `length(path)` and `cost` agree by construction.

## When to use it

Use it when you need the actual route between two known nodes and every relationship counts the same — connections in a journey, hops in a referral chain, steps in a dependency chain.

## How it differs from its neighbours

`graph.dijkstra` measures cost with a numeric relationship property and reports the cheapest cost to *every* reachable node rather than one route to one node. `graph.bfs` gives the same hop distances as this procedure's `cost` but never materialises a path. Cypher's own `shortestPath` pattern selector expresses the same idea inside a `MATCH`; this procedure is the form that composes with the other algorithms.

## Simple example

The fewest-flight route from Heathrow to Wellington, New Zealand.

```cypher
USE flights
MATCH (origin:Airport {iata: 'LHR'}), (destination:Airport {iata: 'WLG'})
CALL graph.shortestpath(origin, destination) YIELD path, cost
RETURN cost AS flights,
       [airport IN nodes(path) | airport.iata] AS route
```

Result:

```
flights | route                                                                                                                            
--------+----------------------------------------------------------------------------------------------------------------------------------
3       | [{"type":"string","value":"LHR"},{"type":"string","value":"YVR"},{"type":"string","value":"AKL"},{"type":"string","value":"WLG"}]

1 row
```

## Advanced example

The same route, priced. The hop-count route is not the shortest route in kilometres, and putting the two side by side is what makes that visible: summing the `km` property along the returned path gives the distance actually flown by the fewest-flight itinerary.

```cypher
USE flights
MATCH (origin:Airport {iata: 'LHR'}), (destination:Airport {iata: 'WLG'})
CALL graph.shortestpath(origin, destination) YIELD path, cost
UNWIND relationships(path) AS leg
RETURN cost AS flights,
       count(leg) AS legs,
       round(sum(leg.km)) AS kilometres,
       round(max(leg.km)) AS longest_leg_km
```

Result:

```
flights | legs | kilometres | longest_leg_km
--------+------+------------+---------------
3       | 3    | 19420      | 11361         

1 row
```

## Where it earns its place

- Producing a concrete route to show a person, not just a distance.
- Measuring separation between two named entities in hops.
- Feeding a path into further projection with `nodes`, `relationships` and `length`.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- No route means no rows. A query that assumes one row per pair will silently lose the pair instead of reporting it; use `OPTIONAL MATCH`-style reasoning or check the row count.
- Ties are broken deterministically but arbitrarily. Do not read the returned route as "the" route when several are equally short.
- Every relationship costs one. Use `graph.dijkstra` when they should not.

## See also

- [`graph.dijkstra`](./graph-dijkstra.md) for weighted cost
- [`graph.bfs`](../traversal/graph-bfs.md) for distances without paths
