# `HISTORY`

> Expands a temporal property's samples in a time range into one row each.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `HISTORY <variable>.<property> FROM <datetime> TO <datetime> AS <alias>` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`HISTORY` takes a declared temporal property on an already-bound entity and produces one row for every sample recorded in the half-open range `FROM`…`TO`. Each row binds an alias exposing two fields: `.time`, the sample's event time as epoch nanoseconds, and `.value`, what the property was set to.

It multiplies rows. One matched account with 535 samples becomes 535 rows, so `HISTORY` is where a query's grain changes from entities to observations.

## How it behaves

`.time` is an integer count of nanoseconds since the epoch, not a datetime value. Divide by 1,000,000,000 and pass it through `datetime.fromepoch` when a reader needs to see it; keep it as an integer when you are only ordering, differencing or bucketing.

Samples are those written for that entity, in event-time order. An entity with no samples in the range contributes no rows at all, so a `HISTORY` clause can reduce the row count to zero as easily as multiply it.

The property must be declared temporal. Applying `HISTORY` to an ordinary property is rejected rather than returning an empty history, so a missing declaration fails loudly instead of looking like an absence of data.

## When to use it

Use it whenever the question is about a trajectory rather than a state: how a value moved, when it crossed a threshold, how volatile it was, what it did between two dates.

## How it differs from its neighbours

`AT TIME` answers "what was it then" with one value and leaves the row grain alone. `HISTORY` answers "what did it do" and changes the grain to one row per sample. `WINDOW` does not read history at all — it buckets whatever rows it is given, which is often but not necessarily `HISTORY` output.

## Simple example

The first ten reputation samples recorded for one account, with their event times rendered as datetimes.

```cypher
USE trust
MATCH (account:Account {account_id: 35})
HISTORY account.reputation
  FROM datetime('2010-01-01T00:00:00Z')
  TO datetime('2017-01-01T00:00:00Z') AS sample
RETURN datetime.fromepoch(sample.time / 1000000000, 0) AS at,
       sample.value AS reputation
ORDER BY sample.time
LIMIT 10
```

Result:

```
at                   | reputation
---------------------+-----------
2010-12-21T12:52:28Z | 2         
2010-12-27T12:37:43Z | 2         
2011-01-02T19:36:31Z | 1.6667    
2011-01-09T19:12:42Z | 1.5       
2011-03-09T13:57:54Z | 1.6       
2011-04-05T18:23:06Z | 1.5       
2011-04-06T09:38:32Z | 1.5714    
2011-04-12T20:19:54Z | 1.5       
2011-04-26T19:14:03Z | 1.4444    
2011-04-30T11:31:40Z | 1.4       

10 rows
```

## Advanced example

The shape of one account's whole reputation history, and how far it travelled. Because each sample is a row, ordinary aggregates describe the trajectory directly: its range, its spread, and the span of time it covers.

```cypher
USE trust
MATCH (account:Account {account_id: 35})
HISTORY account.reputation
  FROM datetime('2010-01-01T00:00:00Z')
  TO datetime('2017-01-01T00:00:00Z') AS sample
RETURN count(sample) AS samples,
       min(sample.value) AS lowest,
       max(sample.value) AS highest,
       round(avg(sample.value) * 1000) / 1000.0 AS mean,
       round(stdev(sample.value) * 1000) / 1000.0 AS spread,
       (max(sample.time) - min(sample.time)) / 86400000000000
         AS days_covered
```

Result:

```
samples | lowest | highest | mean | spread | days_covered
--------+--------+---------+------+--------+-------------
535     | 1.4    | 2       | 1.65 | 0.132  | 1773        

1 row
```

## Where it earns its place

- Charting how a value moved over a period.
- Finding when a value crossed a threshold.
- Measuring volatility with ordinary aggregates over the samples.

## Limitations and trade-offs

- The property must be declared temporal; an ordinary property is rejected.
- `.time` is epoch nanoseconds, not a datetime. Convert it for display.
- Row grain changes to one row per sample. A broad `MATCH` in front of a long history produces a very large row set.
- Only samples inside `FROM`…`TO` appear. There is no implicit sample carrying the value in effect at the start of the range.

## See also

- [`AT TIME`](./at-time.md) for a single instant
- [`WINDOW`](./window.md) to bucket the samples this produces
