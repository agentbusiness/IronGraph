# `variancep`

> Population variance: squared spread when the rows are the whole population.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `variancep(expression)` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`variancep` returns the population variance — the mean squared deviation from the mean, dividing by the count. It is `stdevp` squared and, like `stdevp`, is defined for a single row, where it is `0`.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use it when the group is complete and the result feeds further arithmetic. It is the correct form inside a variance decomposition, where each group's variance describes that group entirely rather than sampling it.

## How it differs from its neighbours

`variance` applies the sample correction. `stdevp` is the square root of this and is what to display. As with the standard deviations, the choice is a claim about what the rows represent.

## Simple example

Population variance per rating year, with the count that produced it.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
WINDOW TUMBLING duration('P365D') ON rating.at AS year
WITH year, rating
RETURN year.start AS window_start,
       count(rating) AS ratings,
       round(avg(rating.rating) * 1000) / 1000.0 AS mean,
       round(variancep(rating.rating) * 1000) / 1000.0 AS population_variance,
       round(stdevp(rating.rating) * 1000) / 1000.0 AS population_stdev
ORDER BY window_start
```

Result:

```
window_start        | ratings | mean  | population_variance | population_stdev
--------------------+---------+-------+---------------------+-----------------
1261440000000000000 | 111     | 2.973 | 6.098               | 2.469           
1292976000000000000 | 7644    | 1.719 | 5.14                | 2.267           
1324512000000000000 | 9236    | 1.194 | 10.477              | 3.237           
1356048000000000000 | 13104   | 0.5   | 18.061              | 4.25            
1387584000000000000 | 4359    | 0.855 | 12.718              | 3.566           
1419120000000000000 | 1081    | 1.079 | 12.92               | 3.594           
1450656000000000000 | 57      | 1.263 | 11.913              | 3.452           

7 rows, 261 ms
```

## Advanced example

Which raters are consistent and which are erratic. Each rater's complete output is a population, so `variancep` describes it exactly; comparing each rater's variance against the network's overall variance says whether they discriminate more or less than the network does as a whole.

```cypher
USE trust
MATCH ()-[all_ratings:RATED]->()
WITH variancep(all_ratings.rating) AS network_variance
MATCH (rater:Account)-[rating:RATED]->()
WITH network_variance, rater, count(rating) AS given,
     avg(rating.rating) AS mean,
     variancep(rating.rating) AS rater_variance
WHERE given >= 40
RETURN rater.account_id AS account, given AS ratings_given,
       round(mean * 100) / 100.0 AS mean_given,
       round(rater_variance * 1000) / 1000.0 AS rater_variance,
       round(network_variance * 1000) / 1000.0 AS network_variance,
       round(rater_variance / network_variance * 1000) / 1000.0 AS variance_ratio
ORDER BY variance_ratio, account
LIMIT 10
```

Result:

```
account | ratings_given | mean_given | rater_variance | network_variance | variance_ratio
--------+---------------+------------+----------------+------------------+---------------
3129    | 212           | 1          | 0              | 12.688           | 0             
2877    | 65            | -1         | 0.092          | 12.688           | 0.007         
4779    | 43            | 1.12       | 0.149          | 12.688           | 0.012         
3735    | 133           | 1.12       | 0.196          | 12.688           | 0.015         
1162    | 70            | 1.13       | 0.283          | 12.688           | 0.022         
257     | 93            | 4.52       | 0.594          | 12.688           | 0.047         
2404    | 65            | 1.31       | 0.613          | 12.688           | 0.048         
3820    | 51            | 1.59       | 0.987          | 12.688           | 0.078         
1053    | 47            | 1.68       | 1.026          | 12.688           | 0.081         
1281    | 56            | 1.04       | 1.07           | 12.688           | 0.084         

10 rows, 369 ms
```

## Where it earns its place

- Group variances inside a decomposition.
- Comparing one group's dispersion against the whole graph's.
- Complete populations where the sample correction would be wrong.

## Limitations and trade-offs

- Squared units, and not a number to display directly.
- Returns `0` for a single row.

## See also

- [`variance`](./variance.md) for the sample form
