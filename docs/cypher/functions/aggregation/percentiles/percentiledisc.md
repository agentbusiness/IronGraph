# `percentiledisc`

> The value at a percentile, always one that actually occurs in the data.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `percentiledisc(expression, percentile)` |
| Relationship to standard Cypher | Standard Cypher, extended by IronGraph |
| Reference dataset | [`trust`](../../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`percentiledisc` orders the non-null numeric values and returns the first one at or past the requested position. It never interpolates, so the result is always a value present in the group.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use it when a value between two observations would be meaningless: counts, ordinal ratings, category codes, anything where 'two and a half' is not a thing that can exist. On this dataset ratings are whole numbers from -10 to +10, so the discrete form is the honest one.

## How it differs from its neighbours

`percentilecont` interpolates and so can return a value that never occurred. On a large group of continuous values the two agree closely; on a small group of discrete values they differ visibly, and the discrete one is right.

## Simple example

The same percentiles under both definitions. Where they disagree, `percentilecont` has invented a rating nobody gave.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
RETURN percentiledisc(rating.rating, 0.25) AS q1_discrete,
       percentilecont(rating.rating, 0.25) AS q1_continuous,
       percentiledisc(rating.rating, 0.5) AS median_discrete,
       percentilecont(rating.rating, 0.5) AS median_continuous,
       percentiledisc(rating.rating, 0.9) AS p90_discrete,
       percentilecont(rating.rating, 0.9) AS p90_continuous
```

Result:

```
q1_discrete | q1_continuous | median_discrete | median_continuous | p90_discrete | p90_continuous
------------+---------------+-----------------+-------------------+--------------+---------------
1           | 1             | 1               | 1                 | 4            | 4             

1 row, 219 ms
```

## Advanced example

A per-account rating profile expressed only in ratings that were actually given. Because every reported figure occurs in the data, the row can be read as a description of real behaviour rather than of a fitted distribution.

```cypher
USE trust
MATCH ()-[rating:RATED]->(rated:Account)
WITH rated, count(rating) AS ratings,
     percentiledisc(rating.rating, 0.1) AS p10,
     percentiledisc(rating.rating, 0.5) AS median,
     percentiledisc(rating.rating, 0.9) AS p90
WHERE ratings >= 40
RETURN rated.account_id AS account, ratings, p10, median, p90,
       p90 - p10 AS span
ORDER BY span DESC, account
LIMIT 10
```

Result:

```
account | ratings | p10 | median | p90 | span
--------+---------+-----+--------+-----+-----
25      | 113     | -10 | 2      | 10  | 20  
2017    | 45      | -10 | -10    | 10  | 20  
1363    | 44      | -10 | 1      | 8   | 18  
62      | 52      | -10 | 1      | 5   | 15  
135     | 93      | -10 | 1      | 5   | 15  
1543    | 51      | -10 | 1      | 5   | 15  
1810    | 311     | -10 | 1      | 5   | 15  
2028    | 279     | -10 | 1      | 5   | 15  
2498    | 45      | -10 | -10    | 5   | 15  
832     | 92      | -10 | 2      | 4   | 14  

10 rows, 229 ms
```

## Where it earns its place

- Percentiles over counts, ordinal scales and category codes.
- Reports where every figure must be a value that genuinely occurred.
- Small groups, where interpolation would invent precision.

## Limitations and trade-offs

- The percentile argument is a fraction between `0` and `1`.
- On a small group the result jumps between observed values as the percentile moves; it does not vary smoothly.
- Requires ordering the group, like `percentilecont`.

## See also

- [`percentilecont`](./percentilecont.md) for the interpolating form
