# `stdev`

> Sample standard deviation: typical distance from the mean, in the original units.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `stdev(expression)` |
| Relationship to standard Cypher | Standard Cypher, extended by IronGraph |
| Reference dataset | [`trust`](../../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`stdev` returns the sample standard deviation of the non-null numeric values in the group — the square root of the sample variance, divided by one less than the count. It is expressed in the same units as the input, which is what makes it directly comparable to the mean.

Use the sample form when the rows are a sample of some larger process, which is the usual case for observational data.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use it whenever you report a mean over data that might not be tightly clustered. A mean without a spread is an assertion that the group is homogeneous, and `stdev` is the cheapest way to check that assertion.

## How it differs from its neighbours

`stdevp` divides by the count rather than the count less one, and is correct when the rows are the entire population rather than a sample of it. The two differ noticeably on small groups and negligibly on large ones. `variance` is the same quantity before the square root, in squared units.

## Simple example

Spread of ratings overall. A standard deviation larger than the mean says the ratings are not clustered around it at all.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
RETURN count(rating) AS ratings,
       round(avg(rating.rating) * 1000) / 1000.0 AS mean,
       round(stdev(rating.rating) * 1000) / 1000.0 AS sample_stdev,
       round(stdevp(rating.rating) * 1000) / 1000.0 AS population_stdev
```

Result:

```
ratings | mean  | sample_stdev | population_stdev
--------+-------+--------------+-----------------
35592   | 1.012 | 3.562        | 3.562           

1 row, 196 ms
```

## Advanced example

The accounts the network cannot agree about. A high mean with a low spread is a solid reputation; the same mean with a high spread is a contested one. Dividing the spread by the mean gives a coefficient of variation that makes accounts with different reputations comparable.

```cypher
USE trust
MATCH ()-[rating:RATED]->(rated:Account)
WITH rated, count(rating) AS ratings,
     avg(rating.rating) AS mean,
     stdev(rating.rating) AS spread
WHERE ratings >= 25 AND mean > 0.5
RETURN rated.account_id AS account, ratings,
       round(mean * 100) / 100.0 AS mean_rating,
       round(spread * 100) / 100.0 AS spread,
       round(spread / mean * 100) / 100.0 AS coefficient_of_variation
ORDER BY coefficient_of_variation DESC, account
LIMIT 10
```

Result:

```
account | ratings | mean_rating | spread | coefficient_of_variation
--------+---------+-------------+--------+-------------------------
4694    | 80      | 0.51        | 4.04   | 7.88                    
905     | 264     | 0.61        | 3.97   | 6.51                    
1810    | 311     | 0.74        | 4.5    | 6.08                    
2028    | 279     | 0.72        | 4.4    | 6.07                    
481     | 37      | 0.97        | 5.18   | 5.33                    
2187    | 26      | 0.62        | 3.14   | 5.1                     
4559    | 82      | 0.72        | 3.63   | 5.05                    
2173    | 54      | 0.78        | 3.86   | 4.97                    
2388    | 136     | 0.73        | 3.35   | 4.6                     
2322    | 31      | 0.65        | 2.92   | 4.52                    

10 rows, 207 ms
```

## Where it earns its place

- Qualifying a mean with how much the values actually vary.
- Finding contested entities: same average, much wider spread.
- Expressing a value as a number of deviations from its group's mean.

## Limitations and trade-offs

- Undefined for a single row; a group of one has no sample spread.
- Assumes the values are a sample. Use `stdevp` for a complete population.
- Like the mean, it is pulled by extremes. On heavily skewed data an interquartile range built from `percentilecont` describes spread better.

## See also

- [`stdevp`](./stdevp.md) for the population form
- [`variance`](./variance.md) for the squared form
