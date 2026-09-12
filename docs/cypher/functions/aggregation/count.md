# `count`

> How many rows reached this point, or how many carried a value.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `count(expression) \| count(*)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`count(*)` counts rows. `count(expression)` counts the rows where the expression is not null. The two differ exactly by the number of nulls, and that difference is often the measurement you actually want.

`count` is the only aggregate that returns a number rather than null when it sees nothing: over an empty input it returns `0`.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use `count(*)` for volume and `count(expression)` for coverage. Use both together when you need to know how complete a property is across the rows a pattern produced.

## How it differs from its neighbours

`count` is the only aggregate that returns `0` rather than null on empty input, which makes it the safe one to divide *by* only after checking it is not zero. `collect` keeps the values instead of counting them.

## Simple example

Volume and coverage of the rating network in one row.

```cypher
USE trust
MATCH (rater:Account)-[rating:RATED]->(rated:Account)
RETURN count(*) AS ratings,
       count(rating.rating) AS with_a_score,
       count(rating.at) AS with_a_timestamp,
       count(rated.reputation) AS rated_has_reputation
```

Result:

```
ratings | with_a_score | with_a_timestamp | rated_has_reputation
--------+--------------+------------------+---------------------
35592   | 35592        | 35592            | 35592               

1 row, 187 ms
```

## Advanced example

The long tail of participation. Grouping accounts by how many ratings they gave, then counting the groups, turns a per-account count into the shape of the whole population — the two levels of counting that most distribution questions need.

```cypher
USE trust
MATCH (rater:Account)-[rating:RATED]->()
WITH rater, count(rating) AS given
WITH CASE
       WHEN given = 1 THEN '1'
       WHEN given <= 5 THEN '2-5'
       WHEN given <= 20 THEN '6-20'
       WHEN given <= 100 THEN '21-100'
       ELSE 'over 100'
     END AS band, given
RETURN band, count(*) AS accounts, sum(given) AS ratings_given
ORDER BY ratings_given DESC
```

Result:

```
band     | accounts | ratings_given
---------+----------+--------------
21-100   | 304      | 12232        
6-20     | 833      | 8360         
over 100 | 38       | 7709         
2-5      | 1846     | 5498         
1        | 1793     | 1793         

5 rows, 185 ms
```

## Where it earns its place

- Measuring how many rows a pattern actually produced.
- Measuring property coverage as the gap between `count(*)` and `count(x)`.
- Counting distinct participants with `count(DISTINCT ...)`.

## Limitations and trade-offs

- `count(*)` counts pattern matches, not distinct entities. A node matched by several paths is counted several times.
- `count(DISTINCT ...)` must retain the distinct values it has seen, so it costs more than a plain count on a high-cardinality expression.

## See also

- [`collect`](./collect.md) to keep the values rather than count them
- [`sum`](./sum.md) to total them
