# `percentilecont`

> The value at a percentile, interpolating between the two rows that surround it.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `percentilecont(expression, percentile)` |
| Relationship to standard Cypher | Standard Cypher, extended by IronGraph |
| Reference dataset | [`trust`](../../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`percentilecont` orders the non-null numeric values and returns the value at the requested position, interpolating linearly when the position falls between two rows. `percentilecont(x, 0.5)` is the median; `0` and `1` give the minimum and maximum.

Because it interpolates, the result need not be a value that appears in the data — which is correct for a continuous measure and wrong for a discrete one.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use it on continuous quantities: durations, distances, prices, scores. It is the right tool for describing skewed data, where the mean is dragged away from anything typical and a set of percentiles describes the shape honestly.

## How it differs from its neighbours

`percentiledisc` returns an actual value from the data instead of interpolating, which is what you want when the value is a category or a count. `avg` gives the centre of mass; a percentile gives a position, and on skewed data the two say very different things.

## Simple example

The distribution of ratings as five positions. Reading them together shows a shape the mean alone conceals.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
RETURN count(rating) AS ratings,
       percentilecont(rating.rating, 0.05) AS p05,
       percentilecont(rating.rating, 0.25) AS p25,
       percentilecont(rating.rating, 0.5) AS median,
       percentilecont(rating.rating, 0.75) AS p75,
       percentilecont(rating.rating, 0.95) AS p95,
       round(avg(rating.rating) * 1000) / 1000.0 AS mean
```

Result:

```
ratings | p05 | p25 | median | p75 | p95 | mean 
--------+-----+-----+--------+-----+-----+------
35592   | -10 | 1   | 1      | 2   | 5   | 1.012

1 row, 236 ms
```

## Advanced example

How the distribution moved over the life of the network. Each year gets its own quartiles and interquartile range, which shows whether the network's ratings became more generous, more polarised, or simply more numerous.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
WINDOW TUMBLING duration('P365D') ON rating.at AS year
WITH year, rating
RETURN year.start AS window_start,
       count(rating) AS ratings,
       percentilecont(rating.rating, 0.25) AS q1,
       percentilecont(rating.rating, 0.5) AS median,
       percentilecont(rating.rating, 0.75) AS q3,
       percentilecont(rating.rating, 0.75)
         - percentilecont(rating.rating, 0.25) AS interquartile_range
ORDER BY window_start
```

Result:

```
window_start        | ratings | q1 | median | q3 | interquartile_range
--------------------+---------+----+--------+----+--------------------
1261440000000000000 | 111     | 1  | 2      | 4  | 3                  
1292976000000000000 | 7644    | 1  | 1      | 2  | 1                  
1324512000000000000 | 9236    | 1  | 1      | 2  | 1                  
1356048000000000000 | 13104   | 1  | 1      | 2  | 1                  
1387584000000000000 | 4359    | 1  | 1      | 2  | 1                  
1419120000000000000 | 1081    | 1  | 1      | 2  | 1                  
1450656000000000000 | 57      | 1  | 1      | 2  | 1                  

7 rows, 282 ms
```

## Where it earns its place

- Describing skewed distributions honestly with a set of positions.
- Interquartile range as a spread measure that ignores extremes.
- Service-level style thresholds on continuous measures.

## Limitations and trade-offs

- The percentile argument is a fraction between `0` and `1`, not a number out of one hundred.
- The interpolated result may not exist in the data. On a discrete measure use `percentiledisc`.
- Computing a percentile requires ordering the group, so it costs more than `avg` over the same rows.

## See also

- [`percentiledisc`](./percentiledisc.md) for a value drawn from the data
- [`avg`](../avg.md) for the centre of mass
