# `CREATE ROLLUP`

> Declares windowed aggregates over a temporal property so they are maintained rather than recomputed.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `CREATE ROLLUP <name> FOR (<var>:<Label>) ON <var>.<property> WINDOW TUMBLING\|HOPPING <width> [EVERY <step>] [ALIGN TO <instant>] [TIME ZONE '<zone>'] AGGREGATE <function>, …` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

A rollup names a windowing and a set of aggregates over one declared temporal property, and asks the database to maintain them. It is a declaration of derived state, in the same family as an index: it changes what work a later query has to do, not what any query returns.

The windowing accepts the same `TUMBLING` and `HOPPING` forms as the `WINDOW` clause, including `ALIGN TO` and `TIME ZONE`, so a rollup can be declared to match exactly the query shape it is meant to serve.

## How it behaves

A rollup is derived state and never graph data. It creates no nodes and no relationships, and nothing in a query result names it. Removing a rollup makes queries slower and never changes an answer.

The aggregates are named as bare identifiers after `AGGREGATE`. Declare the ones the queries actually ask for: each is maintained, so an unused aggregate is pure cost.

Rollups do not appear in `SHOW INDEXES`, which lists index state only.

## When to use it

Declare a rollup when the same windowed aggregate over the same temporal property is asked repeatedly — a dashboard panel, a scheduled report, a threshold check. For a question asked once, the `WINDOW` clause computes the same thing without a standing declaration.

## How it differs from its neighbours

`WINDOW` computes a windowed aggregate for one query, over any rows, whether or not a temporal property is involved. `CREATE ROLLUP` declares one ahead of time over a specific temporal property. The clause is the general tool; the rollup is the standing optimisation for a shape you already know.

## Simple example

A monthly rollup over the reputation series in the reference dataset. It returns no rows: like an index, its effect is on later queries.

```cypher
USE rollup_example
CREATE ROLLUP reputation_monthly FOR (account:Account)
  ON account.reputation
  WINDOW TUMBLING duration('P30D')
  AGGREGATE avg, min, max, count
```

Result:

```
No rows returned. 0 changes committed.
```

## Advanced example

The query a rollup is declared to serve. Its shape mirrors the declaration — the same property, the same window width, aggregates drawn from the declared set — which is what makes the two match up.

```cypher
USE trust
MATCH (account:Account {account_id: 35})
HISTORY account.reputation
  FROM datetime('2011-01-01T00:00:00Z')
  TO datetime('2013-01-01T00:00:00Z') AS sample
WINDOW TUMBLING duration('P30D') ON sample.time AS month
WITH month, sample
RETURN datetime.fromepoch(month.start / 1000000000, 0) AS window_start,
       count(sample) AS samples,
       round(avg(sample.value) * 1000) / 1000.0 AS mean,
       min(sample.value) AS lowest,
       max(sample.value) AS highest
ORDER BY month.start
LIMIT 12
```

Result:

```
window_start         | samples | mean  | lowest | highest
---------------------+---------+-------+--------+--------
2010-12-27T00:00:00Z | 2       | 1.583 | 1.5    | 1.6667 
2011-02-25T00:00:00Z | 1       | 1.6   | 1.6    | 1.6    
2011-03-27T00:00:00Z | 3       | 1.524 | 1.5    | 1.5714 
2011-04-26T00:00:00Z | 8       | 1.505 | 1.4    | 1.5833 
2011-05-26T00:00:00Z | 10      | 1.513 | 1.4783 | 1.5556 
2011-06-25T00:00:00Z | 11      | 1.521 | 1.4815 | 1.5676 
2011-07-25T00:00:00Z | 11      | 1.503 | 1.4583 | 1.5526 
2011-08-24T00:00:00Z | 20      | 1.531 | 1.5    | 1.5862 
2011-09-23T00:00:00Z | 6       | 1.494 | 1.4861 | 1.5072 
2011-10-23T00:00:00Z | 14      | 1.467 | 1.4545 | 1.4805 
2011-11-22T00:00:00Z | 13      | 1.462 | 1.4348 | 1.4747 
2011-12-22T00:00:00Z | 8       | 1.461 | 1.4486 | 1.4815 

12 rows
```

## Where it earns its place

- Dashboard panels that re-ask the same windowed question.
- Scheduled reports over a temporal property.
- Threshold checks that run continuously over a rolling window.

## Limitations and trade-offs

- Derived state only: a rollup changes cost, never answers.
- One temporal property per rollup.
- Declared aggregates are maintained whether or not they are used.
- Not listed by `SHOW INDEXES`.

## See also

- [`WINDOW`](../../clauses/time/window.md) for the per-query form
- [`ALTER … SET TEMPORAL`](./alter-set-temporal.md) for the property it needs
