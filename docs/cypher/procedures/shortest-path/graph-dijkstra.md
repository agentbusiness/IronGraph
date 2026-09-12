# `graph.dijkstra`

> The cheapest cost from one source to every reachable node, measured by a numeric relationship property.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.dijkstra(source [, weightProperty]) YIELD node, cost, predecessor` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`flights`](../../datasets.md#flights) — OpenFlights airports, airlines and routes |

## What it does

`graph.dijkstra` computes the cheapest total cost from `source` to every node reachable along outgoing relationships. It yields one row per reachable node with the accumulated `cost` and the `predecessor` node on the cheapest route, which is what lets you rebuild the route itself.

The optional second argument names a relationship property to use as the weight. Omit it and every relationship weighs `1`, which makes the cost a hop count.

## How it behaves

The weight property must be present and numeric on every relationship the search traverses. A missing or non-numeric weight is an error, not a skipped edge — the query fails rather than quietly returning a wrong cheapest cost. That is deliberate: a silently dropped edge changes the answer without changing its shape.

`predecessor` is `null` for the source and for nothing else. Following `predecessor` backwards from any node reconstructs its cheapest route. The procedure yields only reachable nodes, so a node absent from the result has no route from the source at all.

## When to use it

Use it whenever the cost of crossing a relationship differs between relationships: distance, duration, price, latency, risk. It is also the right procedure for one-to-many questions, because a single call prices every destination at once.

## How it differs from its neighbours

`graph.shortestpath` returns one materialised route to one target and counts hops. `graph.dijkstra` returns costs to everything and counts weight — and with the weight argument omitted the two agree on cost while still differing in shape. `graph.bfs` is the unweighted one-to-many form and is cheaper when hops are genuinely what you want.

## Simple example

The ten airports closest to Heathrow by total kilometres flown, rather than by number of flights. `km` is a numeric property on every `ROUTE` relationship in this dataset, computed from the two airports' coordinates.

```cypher
USE flights
MATCH (origin:Airport {iata: 'LHR'})
CALL graph.dijkstra(origin, 'km') YIELD node, cost
WHERE cost > 0
RETURN node.iata AS iata, node.city AS city, round(cost) AS km
ORDER BY cost, iata
LIMIT 10
```

Result:

```
iata | city        | km 
-----+-------------+----
MAN  | Manchester  | 243
LBA  | Leeds       | 278
RTM  | Rotterdam   | 342
CDG  | Paris       | 347
BRU  | Brussels    | 350
ORY  | Paris       | 367
AMS  | Amsterdam   | 370
NCL  | Newcastle   | 405
IOM  | Isle Of Man | 417
EIN  | Eindhoven   | 427

10 rows
```

## Advanced example

Where the cheapest route is not the most direct one. For each destination this compares the cheapest total distance against the great-circle distance from Heathrow, and reports the destinations whose best itinerary is furthest from a straight line — the detour cost of the route network.

```cypher
USE flights
MATCH (origin:Airport {iata: 'LHR'})
CALL graph.dijkstra(origin, 'km') YIELD node, cost, predecessor
WITH origin, node, cost, predecessor WHERE cost > 2000
WITH node, cost, predecessor,
     6371.0088 * 2 * asin(sqrt(
       sin(radians(node.latitude - origin.latitude) / 2)^2 +
       cos(radians(origin.latitude)) * cos(radians(node.latitude)) *
       sin(radians(node.longitude - origin.longitude) / 2)^2)) AS direct
WHERE direct > 0
RETURN node.iata AS iata, node.city AS city,
       round(cost) AS route_km, round(direct) AS direct_km,
       round(100.0 * cost / direct) AS percent_of_direct,
       predecessor.iata AS arrives_from
ORDER BY percent_of_direct DESC, iata
LIMIT 10
```

Result:

```
iata | city         | route_km | direct_km | percent_of_direct | arrives_from
-----+--------------+----------+-----------+-------------------+-------------
OST  | Ostend       | 2643     | 233       | 1135              | PMI         
XCR  | Chalons      | 2609     | 448       | 583               | OPO         
KSF  | Kassel       | 2764     | 682       | 405               | PMI         
DLE  | Dole         | 2593     | 652       | 398               | OPO         
KLV  | Karlovy Vary | 3642     | 949       | 384               | LED         
RMI  | Rimini       | 4741     | 1278      | 371               | DME         
ANG  | Angouleme    | 2181     | 640       | 341               | FSC         
PED  | Pardubice    | 3538     | 1149      | 308               | LED         
EGS  | Egilsstadir  | 5104     | 1729      | 295               | RKV         
XFW  | Hamburg      | 2146     | 733       | 293               | TLS         

10 rows
```

## Where it earns its place

- Pricing every destination from one origin in a single call.
- Rebuilding the cheapest route to any node by following `predecessor`.
- Comparing network cost against an ideal cost to find where a network detours.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- A missing or non-numeric weight on a traversed relationship fails the query.
- Negative weights are not meaningful to this algorithm; costs must be non-negative for the result to be the cheapest route.
- The result has one row per reachable node, which for a well-connected graph is most of it. Filter inside the query rather than in the client.

## See also

- [`graph.shortestpath`](./graph-shortestpath.md) for one route by hop count
- [`graph.bfs`](../traversal/graph-bfs.md) for unweighted one-to-many distance
