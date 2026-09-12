# `variance`

> Sample variance: the squared spread, before the square root.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `variance(expression)` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`variance` returns the sample variance of the non-null numeric values — the mean squared deviation from the mean, divided by one less than the count. It is `stdev` squared, in squared units.

Standard Cypher offers only the standard deviation. Exposing the variance directly matters because variances combine and standard deviations do not: adding two variances is meaningful, adding two standard deviations is not.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use it when the value feeds further arithmetic — pooling dispersions across groups, weighting them, or decomposing total variation into within-group and between-group parts. Report `stdev` when a person is going to read the number.

## How it differs from its neighbours

`stdev` is the square root of this and is in the original units, which makes it the one to display. `variancep` uses the population divisor. The choice between variance and standard deviation is about what happens next to the number, not about what it measures.

## Simple example

Variance and standard deviation of the same values, related by a square root.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
RETURN count(rating) AS ratings,
       round(variance(rating.rating) * 10000) / 10000.0 AS sample_variance,
       round(variancep(rating.rating) * 10000) / 10000.0 AS population_variance,
       round(stdev(rating.rating) * 10000) / 10000.0 AS sample_stdev,
       round(sqrt(variance(rating.rating)) * 10000) / 10000.0 AS sqrt_of_variance
```

Result:

```
ratings | sample_variance | population_variance | sample_stdev | sqrt_of_variance
--------+-----------------+---------------------+--------------+-----------------
35592   | 12.6885         | 12.6882             | 3.5621       | 3.5621          

1 row, 211 ms
```

## Advanced example

Decomposing total variation into the part explained by which account is being rated and the part that remains inside each account's own ratings. This is the calculation variances exist for: weighting each group's variance by its size and pooling them is only valid in squared units.

```cypher
USE trust
MATCH ()-[rating:RATED]->(rated:Account)
WITH rated, count(rating) AS ratings,
     avg(rating.rating) AS group_mean,
     variancep(rating.rating) AS group_variance
WHERE ratings >= 10
WITH sum(ratings) AS total_ratings,
     count(*) AS accounts,
     sum(ratings * group_variance) AS weighted_within,
     variancep(group_mean) AS between_group_variance,
     avg(group_mean) AS grand_mean
RETURN accounts, total_ratings,
       round(grand_mean * 1000) / 1000.0 AS grand_mean,
       round(weighted_within / total_ratings * 10000) / 10000.0 AS within_group_variance,
       round(between_group_variance * 10000) / 10000.0 AS between_group_variance,
       round(100.0 * between_group_variance /
             (between_group_variance + weighted_within / total_ratings) * 10) / 10.0
         AS percent_explained_by_account
```

Result:

```
accounts | total_ratings | grand_mean | within_group_variance | between_group_variance | percent_explained_by_account
---------+---------------+------------+-----------------------+------------------------+-----------------------------
741      | 23200         | 0.998      | 7.9802                | 4.9183                 | 38.1                        

1 row, 203 ms
```

## Where it earns its place

- Pooling dispersion across groups, which requires squared units.
- Decomposing total variation into within-group and between-group parts.
- Feeding a dispersion into further arithmetic without a round trip through a square root.

## Limitations and trade-offs

- Squared units. A variance of rating points is in rating points squared and should not be shown to a reader as though it were a rating.
- Undefined for a single row.
- Squaring amplifies outliers even more than the standard deviation does.

## See also

- [`variancep`](./variancep.md) for the population form
- [`stdev`](./stdev.md) for the readable form
