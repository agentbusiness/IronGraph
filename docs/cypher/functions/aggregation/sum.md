# `sum`

> The total of the numeric values in the rows that reached this point.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `sum(expression)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`sum` adds up numeric values, skipping nulls. Over an input with no numeric values it returns null.

On signed data — a rating from -10 to +10, a balance, a delta — a sum is a net position, and a net position near zero can mean either no activity or a great deal of activity in both directions. Pair it with `count` whenever the sign varies.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use it for quantities that genuinely add: totals, net positions, accumulated weight along a path. Do not use it for rates or ratios, which do not.

## How it differs from its neighbours

`avg` divides the same total by the count and so hides volume; `sum` keeps volume and hides typicality. Reporting both costs nothing and prevents the most common misreading of either.

## Simple example

Net trust in the network, alongside the volume that produced it.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
RETURN sum(rating.rating) AS net_trust,
       count(rating) AS ratings,
       round(avg(rating.rating) * 10000) / 10000.0 AS mean_rating
```

Result:

```
net_trust | ratings | mean_rating
----------+---------+------------
36020     | 35592   | 1.012      

1 row, 183 ms
```

## Advanced example

Accounts whose net trust is near zero for opposite reasons. Splitting the sum into its positive and negative halves separates an account nobody has an opinion about from one the network actively disagrees over — a distinction the net figure alone destroys.

```cypher
USE trust
MATCH ()-[rating:RATED]->(rated:Account)
WITH rated,
     sum(rating.rating) AS net,
     sum(CASE WHEN rating.rating > 0 THEN rating.rating ELSE 0 END) AS positive,
     sum(CASE WHEN rating.rating < 0 THEN -rating.rating ELSE 0 END) AS negative,
     count(rating) AS ratings
WHERE ratings >= 20 AND net > -5 AND net < 5
RETURN rated.account_id AS account, ratings, net, positive, negative,
       round(100.0 * negative / (positive + negative) * 10) / 10.0 AS percent_negative
ORDER BY negative DESC, account
LIMIT 10
```

Result:

```
account | ratings | net | positive | negative | percent_negative
--------+---------+-----+----------+----------+-----------------

0 rows, 232 ms
```

## Where it earns its place

- Totals and net positions over signed measures.
- Accumulating a weight along matched relationships.
- Splitting a total into signed halves with `CASE` to expose disagreement.

## Limitations and trade-offs

- A sum over signed values conceals volume. Report `count` beside it.
- Summing rates, ratios or percentages produces a number with no meaning.
- Floating-point sums depend on row order for their last digits; round before comparing for equality.

## See also

- [`avg`](./avg.md) for the same total per row
- [`count`](./count.md) for the volume behind it
