# `REBUILD INDEX`

> Rebuilds an index from the current graph.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `REBUILD INDEX <name>` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`flights`](../../datasets.md#flights) — OpenFlights airports, airlines and routes |

## What it does

`REBUILD INDEX` discards an index's derived contents and builds them again from the graph as it now stands. The index's declaration is unchanged; only what it holds is recomputed.

## How it behaves

It does not change any answer. An index is derived state, so a rebuild produces the same query results at possibly different cost.

A rebuild cannot fix a structural failure. A vector index that missed its recall floor will miss it again on the same data, because the measurement is of the data and the parameters rather than of a stale build.

## When to use it

Rebuild after a change in the shape of the data that the incremental path would leave a poor structure for — a bulk load, a large deletion — or when investigating whether an index's contents explain a performance change.

## How it differs from its neighbours

Dropping and re-declaring achieves the same contents and briefly leaves the project without the index. `REBUILD INDEX` keeps the declaration throughout.

## Simple example

Rebuilding a declared index. Nothing is returned; the effect is the rebuild.

```cypher
USE flights
REBUILD INDEX airport_by_iata
```

Result:

```
No rows returned. 0 changes committed.
```

## Advanced example

A rebuild changes no answer. The same lookup is run after the rebuild and returns exactly what it did before, which is the property that makes an index safe to rebuild at any time.

```cypher
USE flights
MATCH (airport:Airport {iata: 'LHR'})
RETURN airport.iata AS iata, airport.name AS name,
       airport.city AS city, airport.country AS country
```

Result:

```
iata | name                    | city   | country       
-----+-------------------------+--------+---------------
LHR  | London Heathrow Airport | London | United Kingdom

1 row, 83 ms
```

## Where it earns its place

- Recovering index quality after a bulk load or large deletion.
- Isolating whether index contents explain a performance change.

## Limitations and trade-offs

- Changes cost, never answers.
- Cannot repair a failure that is a property of the data or the parameters.
- The index is being rebuilt while the statement runs.

## See also

- [`SHOW INDEXES`](./show-indexes.md)
