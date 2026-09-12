# `AT TIME`

> Runs the whole query against the graph as it stood at a past instant.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `AT TIME <datetime>  (before the query body)` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`AT TIME` sits ahead of the query body, beside `USE` and `USE LAYER`, and moves the entire query to a chosen instant. Every declared temporal property read anywhere in that query returns the value in effect then, rather than its current one.

It is a property of the query, not of a clause. There is no way to read two different instants in one statement, which is deliberate: a single query always describes one consistent moment.

## How it behaves

IronGraph separates two clocks. **Event time** is when something happened in the world; it is what a temporal property records and what `AT TIME` and `HISTORY` read. **Write time** is when the database was told, and is the default event time when a write does not say otherwise. Keeping them apart is what lets a correction arrive late without rewriting history, and what lets a backfill land with the timestamps the data actually had.

A declared temporal property has two distinct reads that are easy to confuse.

- Reading it in an ordinary query returns the **canonical** value: whatever an ordinary `SET` last wrote. Writes made with `AT TIME` do not touch it.
- Reading it under `AT TIME`, or through `HISTORY`, returns the **temporal** value: the sample in effect at that instant.

The two can differ, and on the `trust` dataset they do: `reputation` was written entirely through backdated samples, so its canonical value is still the `0.0` set at load while its temporal value follows the real 2010-2016 curve.

Before an entity's first sample, its temporal value is `null` — not its eventual first value, and not an error. A time-travelling query therefore reports genuine absence for entities that did not yet have the property, which is what makes counting them meaningful.

Only declared temporal properties travel. Ordinary properties, labels, relationships and the existence of nodes are read as they are now. `AT TIME` reconstructs the past of declared values, not the past of the whole graph.

## When to use it

Use it to answer "what did we believe then": reproducing a past report, auditing a decision against the information available at the time, or comparing a value now against the same value at a chosen moment.

## How it differs from its neighbours

`AT TIME` gives one instant's value per entity and keeps the ordinary row shape. `HISTORY` gives every sample in a range as its own row, which changes the row grain. Use `AT TIME` for a snapshot and `HISTORY` for a trajectory.

`AT TIME` also appears in a second, unrelated position: attached to a `SET` item it stamps a written sample rather than choosing a read instant. See [`SET … AT TIME`](../../statements/temporal/set-at-time.md).

## Simple example

One account's reputation at four moments in the network's life. Each figure is the value in effect on that date, reconstructed from history.

```cypher
USE trust
AT TIME datetime('2012-06-01T00:00:00Z')
MATCH (account:Account {account_id: 35})
RETURN account.account_id AS account,
       account.reputation AS reputation_mid_2012
```

Result:

```
account | reputation_mid_2012
--------+--------------------
35      | 1.5375             

1 row
```

## Advanced example

How much of the network existed yet. Because a temporal read is `null` before an entity's first sample, counting non-null reputations at an instant counts the accounts that had been rated by then — a measurement that needs no separate created-at field.

```cypher
USE trust
AT TIME datetime('2012-01-01T00:00:00Z')
MATCH (account:Account)
RETURN count(account) AS accounts_in_the_graph,
       count(account.reputation) AS rated_by_2012,
       round(avg(account.reputation) * 1000) / 1000.0 AS mean_reputation,
       min(account.reputation) AS lowest,
       max(account.reputation) AS highest
```

Result:

```
accounts_in_the_graph | rated_by_2012 | mean_reputation | lowest | highest
----------------------+---------------+-----------------+--------+--------
5881                  | 1631          | 1.604           | -10    | 10     

1 row, 8666 ms
```

## Where it earns its place

- Reproducing a report exactly as it read on a past date.
- Auditing a decision against what was known when it was taken.
- Counting when entities entered a dataset, without a created-at field.

## Limitations and trade-offs

- Only declared temporal properties travel in time. Node existence, labels, relationships and ordinary properties are always read as they are now.
- One instant per query. Comparing two moments takes two queries, or a `HISTORY` range.
- A read before an entity's first sample is `null`. Aggregates skip those rows, which is usually right and occasionally surprising.
- The canonical value of the property is a different value and is unaffected.

## See also

- [`HISTORY`](./history.md) for every sample rather than one instant
- [`ALTER … SET TEMPORAL`](../../statements/temporal/alter-set-temporal.md) to declare a property temporal in the first place
