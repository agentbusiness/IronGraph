# `min`

> The smallest value among the rows that reached this point.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `min(expression)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`min` returns the smallest non-null value, using the ordering of the value's own type: numeric for numbers, chronological for temporal values, lexicographic for strings. Over no non-null values it returns null.

Applied to a timestamp, `min` is the first time something happened — which is how you find an entity's beginning without keeping a separate field for it.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use it for the extreme itself: the worst rating, the earliest event, the cheapest route. Use it on a timestamp whenever you need a first-seen date.

## How it differs from its neighbours

`min` gives the value at the extreme but not the row it came from. When you need the whole row, order and limit instead, or `collect` and index into the result.

## Simple example

The span of the rating network, in both value and time. `min` over a datetime is the first rating ever recorded.

```cypher
USE trust
MATCH ()-[rating:RATED]->()
RETURN min(rating.rating) AS lowest_rating,
       max(rating.rating) AS highest_rating,
       min(rating.at) AS first_rating,
       max(rating.at) AS last_rating
```

Result:

```
lowest_rating | highest_rating | first_rating         | last_rating         
--------------+----------------+----------------------+---------------------
-10           | 10             | 2010-11-08T18:45:11Z | 2016-01-25T01:12:03Z

1 row, 191 ms
```

## Advanced example

How long each account's rating history runs. `min` and `max` over the timestamp give first and last activity, and their difference is a lifetime — computed here for the accounts with the longest histories in the network.

```cypher
USE trust
MATCH (rater:Account)-[rating:RATED]->()
WITH rater, count(rating) AS ratings,
     min(rating.at_epoch) AS first_epoch,
     max(rating.at_epoch) AS last_epoch
WHERE ratings >= 50
RETURN rater.account_id AS account, ratings,
       (last_epoch - first_epoch) / 86400 AS active_days,
       round(1.0 * ratings * 86400 / (last_epoch - first_epoch) * 100) / 100.0
         AS ratings_per_day
ORDER BY active_days DESC, account
LIMIT 10
```

Result:

```
account | ratings | active_days | ratings_per_day
--------+---------+-------------+----------------
13      | 210     | 1903        | 0.11           
41      | 100     | 1864        | 0.05           
35      | 763     | 1861        | 0.41           
104     | 56      | 1743        | 0.03           
215     | 54      | 1715        | 0.03           
361     | 80      | 1703        | 0.05           
546     | 167     | 1679        | 0.1            
1018    | 203     | 1677        | 0.12           
1052    | 54      | 1669        | 0.03           
135     | 92      | 1645        | 0.06           

10 rows, 214 ms
```

## Where it earns its place

- First-seen timestamps without a dedicated field.
- The worst or cheapest value in a group.
- Bounding a range together with `max`.

## Limitations and trade-offs

- Returns the value, never the row that held it.
- Mixing types in one `min` compares across type orderings and is rarely meaningful.

## See also

- [`max`](./max.md) for the other end of the range
