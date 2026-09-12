# `max`

> The largest value among the rows that reached this point.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `max(expression)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`max` returns the largest non-null value under the ordering of the value's own type, and null when nothing non-null reached it. On a timestamp it is the most recent event, which makes it the natural way to express recency.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use it for the best, the largest, or the latest. Paired with `min` it gives a range; paired with `avg` it shows how far the extreme sits from typical.

## How it differs from its neighbours

Like `min`, `max` yields a value and not a row. For the row itself, order descending and limit.

## Simple example

The most recent activity of the ten busiest raters.

```cypher
USE trust
MATCH (rater:Account)-[rating:RATED]->()
WITH rater, count(rating) AS ratings, max(rating.at) AS last_seen
RETURN rater.account_id AS account, ratings, last_seen
ORDER BY ratings DESC, account
LIMIT 10
```

Result:

```
account | ratings | last_seen           
--------+---------+---------------------
35      | 763     | 2016-01-04T11:18:57Z
2642    | 406     | 2014-04-24T19:05:00Z
1810    | 404     | 2016-01-24T04:53:07Z
2125    | 397     | 2015-12-22T05:32:03Z
2028    | 293     | 2013-03-29T05:53:59Z
905     | 264     | 2015-08-17T21:08:34Z
4172    | 264     | 2015-05-28T21:27:29Z
7       | 232     | 2014-02-22T05:13:41Z
1       | 215     | 2015-03-24T01:50:08Z
3129    | 212     | 2013-08-23T10:02:40Z

10 rows, 204 ms
```

## Advanced example

How far the best rating an account received sits above its typical one. `max` beside `avg` separates accounts that are consistently well regarded from accounts with one enthusiastic supporter and an otherwise ordinary record.

```cypher
USE trust
MATCH ()-[rating:RATED]->(rated:Account)
WITH rated, count(rating) AS ratings,
     max(rating.rating) AS best,
     avg(rating.rating) AS mean,
     stdev(rating.rating) AS spread
WHERE ratings >= 30 AND spread > 0
RETURN rated.account_id AS account, ratings, best,
       round(mean * 100) / 100.0 AS mean,
       round(spread * 100) / 100.0 AS spread,
       round((best - mean) / spread * 100) / 100.0 AS best_in_deviations
ORDER BY best_in_deviations DESC, account
LIMIT 10
```

Result:

```
account | ratings | best | mean | spread | best_in_deviations
--------+---------+------+------+--------+-------------------
1802    | 62      | 10   | 1.76 | 1.43   | 5.75              
546     | 144     | 10   | 1.38 | 1.55   | 5.55              
2600    | 82      | 6    | 1.4  | 0.83   | 5.54              
1555    | 65      | 8    | 1.54 | 1.19   | 5.44              
3649    | 96      | 10   | 1.74 | 1.54   | 5.35              
4515    | 47      | 10   | 1.74 | 1.55   | 5.32              
1832    | 105     | 9    | 1.86 | 1.35   | 5.3               
1899    | 132     | 10   | 1.8  | 1.55   | 5.3               
2110    | 53      | 10   | 1.7  | 1.58   | 5.27              
3828    | 99      | 10   | 1.93 | 1.55   | 5.22              

10 rows, 223 ms
```

## Where it earns its place

- Last-seen timestamps and recency.
- The best value in a group.
- Measuring how exceptional an extreme is, against `avg` and `stdev`.

## Limitations and trade-offs

- Returns the value, never the row that held it.
- A single extreme row can make a group look unlike itself; check `count` and spread before drawing conclusions.

## See also

- [`min`](./min.md) for the other end of the range
