# `stdevp`

> Population standard deviation: spread when the rows are everything, not a sample.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `stdevp(expression)` |
| Relationship to standard Cypher | Standard Cypher, extended by IronGraph |
| Reference dataset | [`trust`](../../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`stdevp` returns the population standard deviation: the square root of the mean squared deviation, dividing by the count rather than the count less one. Unlike `stdev` it is defined for a single row, where it is `0`.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use it when the rows in the group are the complete population you are describing — every rating an account ever received, every relationship in the graph — rather than a sample drawn from something larger.

## How it differs from its neighbours

The only difference from `stdev` is the divisor. `stdevp` is always the smaller of the two, and the gap matters only on small groups. Choosing between them is a statement about what the rows represent, not about the arithmetic.

## Simple example

Where the sample and population forms diverge. The gap is large on small groups and vanishes on large ones.

```cypher
USE trust
MATCH ()-[rating:RATED]->(rated:Account)
WITH rated, count(rating) AS ratings,
     stdev(rating.rating) AS sample,
     stdevp(rating.rating) AS population
WHERE ratings >= 2
WITH CASE
       WHEN ratings <= 3 THEN '2-3'
       WHEN ratings <= 10 THEN '4-10'
       WHEN ratings <= 50 THEN '11-50'
       ELSE 'over 50'
     END AS band, sample, population
RETURN band, count(*) AS accounts,
       round(avg(sample) * 10000) / 10000.0 AS mean_sample_stdev,
       round(avg(population) * 10000) / 10000.0 AS mean_population_stdev,
       round(avg(sample) - avg(population) * 10000) / 10000.0 AS gap
ORDER BY gap DESC
```

Result:

```
band    | accounts | mean_sample_stdev | mean_population_stdev | gap    
--------+----------+-------------------+-----------------------+--------
2-3     | 1607     | 1.5294            | 1.1498                | -1.1496
4-10    | 1158     | 2.2698            | 2.0536                | -2.0534
11-50   | 560      | 2.3201            | 2.2524                | -2.2522
over 50 | 106      | 2.5892            | 2.5732                | -2.573 

4 rows, 205 ms
```

## Advanced example

Standardising a rating against the population it belongs to. Each account's complete set of received ratings is a population, so `stdevp` is the correct divisor, and the resulting z-score says how unusual the harshest rating each account received was relative to its own record.

```cypher
USE trust
MATCH ()-[rating:RATED]->(rated:Account)
WITH rated, count(rating) AS ratings,
     avg(rating.rating) AS mean,
     stdevp(rating.rating) AS spread,
     min(rating.rating) AS harshest
WHERE ratings >= 25 AND spread > 0
RETURN rated.account_id AS account, ratings,
       round(mean * 100) / 100.0 AS mean_rating,
       harshest,
       round((harshest - mean) / spread * 100) / 100.0 AS harshest_z_score
ORDER BY harshest_z_score, account
LIMIT 10
```

Result:

```
account | ratings | mean_rating | harshest | harshest_z_score
--------+---------+-------------+----------+-----------------
546     | 144     | 1.38        | -10      | -7.34           
13      | 191     | 1.79        | -10      | -7.18           
5227    | 63      | 1.35        | -10      | -6.01           
1744    | 69      | 1.33        | -10      | -5.91           
1612    | 41      | 1.22        | -10      | -5.85           
2942    | 123     | 1.96        | -10      | -5.67           
1348    | 34      | 0.94        | -10      | -5.56           
2296    | 145     | 1.62        | -10      | -5.48           
2198    | 85      | 1.94        | -10      | -5.47           
96      | 37      | 1.22        | -10      | -5.42           

10 rows, 224 ms
```

## Where it earns its place

- Describing a complete population rather than a sample of one.
- Standardising values into z-scores within their own group.
- Groups small enough that the sample correction would distort the result.

## Limitations and trade-offs

- Returns `0` for a single row, which is arithmetically right and easy to misread as "no variation observed".
- Understates spread if the rows really are a sample.

## See also

- [`stdev`](./stdev.md) for the sample form
