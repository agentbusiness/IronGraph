# `WINDOW`

> Buckets rows into time windows over any instant the rows carry.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `WINDOW TUMBLING <width> ON <instant> AS <alias>  \|  WINDOW HOPPING <width> EVERY <step> ON <instant> AS <alias>` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`WINDOW` assigns each row to one or more time buckets based on an instant the row carries, and binds an alias exposing `.start` and `.end` as epoch nanoseconds. Those become ordinary grouping columns, so the aggregate that follows is grouped per window.

`TUMBLING` windows are adjacent and non-overlapping: a row lands in exactly one. `HOPPING` windows are as wide as `TUMBLING` ones but start every `EVERY` interval, so they overlap and a row lands in several — which is how a moving average is expressed.

## How it behaves

The instant can come from anywhere: a `datetime` property on a matched relationship, a `HISTORY` sample's `.time`, or any expression producing an instant. `WINDOW` does not read temporal history itself and does not require a declared temporal property. That is what makes it usable on ordinary event-bearing data.

`.start` and `.end` are epoch nanoseconds. They are useful as-is for ordering and differencing, and should be converted with `datetime.fromepoch` for display.

Optional modifiers refine the grid. `ALIGN TO` fixes the boundary the windows are measured from, so buckets line up with a business day rather than the epoch. `TIME ZONE` names the zone the alignment is interpreted in. `EMIT EMPTY` keeps windows with no rows, which matters for a chart that must not silently close a gap.

## When to use it

Use it for any per-period rollup: activity per month, a moving average, a rate over time, or a comparison of the same measure across successive periods. It replaces bucketing the timestamp by hand with arithmetic, and unlike hand bucketing it can overlap.

## How it differs from its neighbours

`TUMBLING` and `HOPPING` differ only in overlap: with `EVERY` equal to the width, a hopping window is a tumbling one. Bucketing by hand with `date.truncate` is equivalent to a tumbling window aligned to the calendar, but cannot express overlap at all.

A rollup declared with `CREATE ROLLUP` computes the same shape ahead of time for a temporal property; `WINDOW` computes it per query over any rows.

## Simple example

Rating activity per calendar year of the network's life, bucketed on the timestamp each rating carries.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
WINDOW TUMBLING duration('P365D') ON rating.at AS year
WITH year, rating
RETURN datetime.fromepoch(year.start / 1000000000, 0) AS window_start,
       count(rating) AS ratings,
       round(avg(rating.rating) * 1000) / 1000.0 AS mean_rating
ORDER BY year.start
```

Result:

```
window_start         | ratings | mean_rating
---------------------+---------+------------
2009-12-22T00:00:00Z | 111     | 2.973      
2010-12-22T00:00:00Z | 7644    | 1.719      
2011-12-22T00:00:00Z | 9236    | 1.194      
2012-12-21T00:00:00Z | 13104   | 0.5        
2013-12-21T00:00:00Z | 4359    | 0.855      
2014-12-21T00:00:00Z | 1081    | 1.079      
2015-12-21T00:00:00Z | 57      | 1.263      

7 rows, 249 ms
```

## Advanced example

A three-month moving view of the network, stepped monthly. Overlapping windows smooth the month-to-month noise: each row summarises the quarter ending at its start plus two months, and successive rows share two thirds of their data.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
WINDOW HOPPING duration('P90D') EVERY duration('P30D')
  ON rating.at AS quarter
WITH quarter, rating
RETURN datetime.fromepoch(quarter.start / 1000000000, 0) AS window_start,
       count(rating) AS ratings,
       round(avg(rating.rating) * 100) / 100.0 AS mean_rating,
       sum(CASE WHEN rating.rating < 0 THEN 1 ELSE 0 END) AS negative
ORDER BY quarter.start
LIMIT 12
```

Result:

```
window_start         | ratings | mean_rating | negative
---------------------+---------+-------------+---------
2010-08-29T00:00:00Z | 50      | 3.76        | 0       
2010-09-28T00:00:00Z | 114     | 2.97        | 0       
2010-10-28T00:00:00Z | 221     | 2.29        | 0       
2010-11-27T00:00:00Z | 386     | 1.87        | 0       
2010-12-27T00:00:00Z | 532     | 1.65        | 4       
2011-01-26T00:00:00Z | 878     | 1.87        | 5       
2011-02-25T00:00:00Z | 2377    | 1.77        | 33      
2011-03-27T00:00:00Z | 4733    | 1.81        | 89      
2011-04-26T00:00:00Z | 4956    | 1.71        | 121     
2011-05-26T00:00:00Z | 3703    | 1.72        | 100     
2011-06-25T00:00:00Z | 1496    | 1.48        | 51      
2011-07-25T00:00:00Z | 1027    | 1.64        | 21      

12 rows, 481 ms
```

## Where it earns its place

- Per-period rollups over event-bearing relationships.
- Moving averages and smoothed trends, through overlapping hopping windows.
- Bucketing `HISTORY` samples into periods to chart a trajectory.

## Limitations and trade-offs

- `.start` and `.end` are epoch nanoseconds, not datetimes.
- A hopping window places each row in several buckets, so counts across all windows sum to more than the row count. That is correct and routinely misread.
- `EVERY` is only legal with `HOPPING`; a tumbling window with a step is rejected.
- Windows are measured from a fixed grid. Use `ALIGN TO` when the boundaries must match a business calendar rather than the epoch.

## See also

- [`HISTORY`](./history.md) to produce samples to window
- [`CREATE ROLLUP`](../../statements/temporal/create-rollup.md) to precompute the same shape
