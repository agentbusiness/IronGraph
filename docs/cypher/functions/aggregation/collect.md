# `collect`

> Gathers the values that reached this point into one list.

| | |
| --- | --- |
| Kind | Aggregate function |
| Signature | `collect(expression)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`collect` returns a list of the non-null values in the group, in the order the rows arrived. Over no rows it returns an empty list rather than null — the one aggregate besides `count` that has a meaningful empty result.

It is the aggregate that changes shape rather than reducing to a number: many rows become one row holding many values, which the list functions can then work on.

## How it behaves

An aggregate consumes the rows that reach it and returns one row per distinct combination of the non-aggregated expressions projected beside it. Those expressions are the grouping key: nothing declares it, and adding a column to the projection silently changes the grain. When a projection contains only aggregates, every incoming row collapses into a single result row.

Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that actually carried a value. Over rows that are all null, or over no rows at all, the result is null rather than an error — with the exception of `count`, which counts.

## When to use it

Use it to fold a one-to-many relationship into one row per parent, and to assemble an ordered set of values you want to index into, slice, or compare against another group's.

## How it differs from its neighbours

Every other aggregate throws the individual values away. `collect` keeps them, which is why it is the expensive one: the whole group is held in memory as a list. Reach for it when you need the values, and for `count` or `sum` when you only need a number about them.

## Simple example

The accounts each of the busiest raters rated most negatively, folded into one row each.

```cypher
USE trust
MATCH (rater:Account)-[rating:RATED]->(rated:Account)
WHERE rating.rating <= -8
WITH rater, collect(rated.account_id) AS distrusted
RETURN rater.account_id AS account,
       size(distrusted) AS strongly_distrusts,
       distrusted[0..5] AS first_five
ORDER BY strongly_distrusts DESC, account
LIMIT 10
```

Result:

```
account | strongly_distrusts | first_five                                                                                                                                                                 
--------+--------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1810    | 134                | [{"type":"integer","value":"905"},{"type":"integer","value":"1675"},{"type":"integer","value":"1917"},{"type":"integer","value":"1964"},{"type":"integer","value":"2002"}] 
2125    | 100                | [{"type":"integer","value":"62"},{"type":"integer","value":"2265"},{"type":"integer","value":"4531"},{"type":"integer","value":"4587"},{"type":"integer","value":"4603"}]  
4172    | 64                 | [{"type":"integer","value":"204"},{"type":"integer","value":"347"},{"type":"integer","value":"574"},{"type":"integer","value":"777"},{"type":"integer","value":"905"}]     
2067    | 63                 | [{"type":"integer","value":"25"},{"type":"integer","value":"2017"},{"type":"integer","value":"2096"},{"type":"integer","value":"2260"},{"type":"integer","value":"2367"}]  
2266    | 43                 | [{"type":"integer","value":"642"},{"type":"integer","value":"1543"},{"type":"integer","value":"2017"},{"type":"integer","value":"2090"},{"type":"integer","value":"2343"}] 
2691    | 38                 | [{"type":"integer","value":"2522"},{"type":"integer","value":"2541"},{"type":"integer","value":"2542"},{"type":"integer","value":"2543"},{"type":"integer","value":"2544"}]
905     | 36                 | [{"type":"integer","value":"770"},{"type":"integer","value":"832"},{"type":"integer","value":"1207"},{"type":"integer","value":"1357"},{"type":"integer","value":"1600"}]  
2351    | 35                 | [{"type":"integer","value":"733"},{"type":"integer","value":"962"},{"type":"integer","value":"1043"},{"type":"integer","value":"1377"},{"type":"integer","value":"2574"}]  
2934    | 35                 | [{"type":"integer","value":"1991"},{"type":"integer","value":"2028"},{"type":"integer","value":"2897"},{"type":"integer","value":"3015"},{"type":"integer","value":"3022"}]
2388    | 34                 | [{"type":"integer","value":"3"},{"type":"integer","value":"2017"},{"type":"integer","value":"2498"},{"type":"integer","value":"2690"},{"type":"integer","value":"2898"}]   

10 rows, 155 ms
```

## Advanced example

Pairs of accounts that rated each other. Collecting each account's outbound targets, then testing membership from the other side, expresses reciprocity as a list operation rather than a second traversal.

```cypher
USE trust
MATCH (a:Account)-[out:RATED]->(b:Account)
WHERE out.rating >= 8
WITH a, collect(b.account_id) AS endorsed
WITH a, endorsed WHERE size(endorsed) >= 5
MATCH (b:Account)-[back:RATED]->(a)
WHERE b.account_id IN endorsed
WITH a, endorsed, count(back) AS mutual, avg(back.rating) AS returned
RETURN a.account_id AS account,
       size(endorsed) AS strongly_endorsed,
       mutual AS endorsed_back_by,
       round(1000.0 * mutual / size(endorsed)) / 10.0 AS reciprocity_percent,
       round(returned * 100) / 100.0 AS mean_rating_returned
ORDER BY reciprocity_percent DESC, account
LIMIT 10
```

Result:

```
account | strongly_endorsed | endorsed_back_by | reciprocity_percent | mean_rating_returned
--------+-------------------+------------------+---------------------+---------------------
4       | 5                 | 5                | 100                 | 8.2                 
2642    | 5                 | 5                | 100                 | 8.8                 
2647    | 6                 | 6                | 100                 | 5.83                
3500    | 6                 | 6                | 100                 | 2.5                 
4172    | 5                 | 5                | 100                 | 7.4                 
1       | 11                | 10               | 90.9                | 8.4                 
1217    | 7                 | 6                | 85.7                | 6.5                 
1386    | 7                 | 6                | 85.7                | 8.33                
540     | 6                 | 5                | 83.3                | 4                   
1396    | 6                 | 5                | 83.3                | 6                   

10 rows, 159 ms
```

## Where it earns its place

- Folding a one-to-many relationship into one row per parent.
- Building a list to test membership against with `IN`.
- Assembling an ordered sample you can slice with a range.

## Limitations and trade-offs

- The whole group is held in memory. Collecting over an unbounded pattern is the most reliable way to make a query expensive.
- Order is the order rows arrived, which is not guaranteed to be meaningful unless you ordered them first.
- Nulls are dropped, so the list can be shorter than the group.

## See also

- [`count`](./count.md) when only the size matters
- [`size`](../collection/size.md) to measure the resulting list
