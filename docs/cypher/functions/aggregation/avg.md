# `avg`

> The arithmetic mean of the numeric values that reached this point.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `avg(expression)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`avg` returns the sum of the non-null numeric values divided by how many there were, always as a float. Over no numeric values it returns null rather than zero, which keeps an empty group distinguishable from a group averaging zero.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use it when you want a typical value and the data is roughly symmetric. On skewed data — which most graph measurements are — the mean sits away from anything typical, and `median` describes the population better.

## How it differs from its neighbours

`median`, spelled `percentilecont(x, 0.5)`, is unmoved by extremes; `avg` is dragged by every one of them. Where the two disagree, the data is skewed, and the size of the disagreement is a useful measurement in itself.

## Simple example

Mean and median rating, side by side.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
RETURN count(rating) AS ratings,
       round(avg(rating.rating) * 10000) / 10000.0 AS mean,
       percentilecont(rating.rating, 0.5) AS median,
       min(rating.rating) AS lowest,
       max(rating.rating) AS highest
```

Result:

```
ratings | mean  | median | lowest | highest
--------+-------+--------+--------+--------
35592   | 1.012 | 1      | -10    | 10     

1 row, 209 ms
```

## Advanced example

Mean against median per year. The gap between them measures the skew of each year's ratings: where the mean sits well below the median, a minority of strongly negative ratings is pulling the average away from the typical experience.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
WINDOW TUMBLING duration('P365D') ON rating.at AS year
WITH year, rating
RETURN year.start AS window_start,
       count(rating) AS ratings,
       round(avg(rating.rating) * 1000) / 1000.0 AS mean,
       percentilecont(rating.rating, 0.5) AS median,
       round(avg(rating.rating) - percentilecont(rating.rating, 0.5) * 1000) / 1000.0 AS skew
ORDER BY window_start
```

Result:

```
window_start        | ratings | mean  | median | skew  
--------------------+---------+-------+--------+-------
1261440000000000000 | 111     | 2.973 | 2      | -1.997
1292976000000000000 | 7644    | 1.719 | 1      | -0.998
1324512000000000000 | 9236    | 1.194 | 1      | -0.999
1356048000000000000 | 13104   | 0.5   | 1      | -1    
1387584000000000000 | 4359    | 0.855 | 1      | -0.999
1419120000000000000 | 1081    | 1.079 | 1      | -0.999
1450656000000000000 | 57      | 1.263 | 1      | -0.999

7 rows, 271 ms
```

## Where it earns its place

- A typical value over roughly symmetric data.
- Comparing groups on a common scale.
- Detecting skew by differencing against the median.

## Limitations and trade-offs

- Extremes move the mean without limit. One outlier in a small group dominates it.
- A mean over a handful of rows is not a measurement. Report `count` beside it.
- Averaging values that are themselves averages weights the groups wrongly.

## See also

- [`percentilecont`](../aggregation/percentilecont.md) for the median
- [`stdev`](./stdev.md) for how spread the values are
