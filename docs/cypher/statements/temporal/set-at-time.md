# `SET … AT TIME`

> Writes one history sample stamped with the event time you give it.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `SET <variable>.<property> = <value> AT TIME <datetime>` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

Appending `AT TIME` to a `SET` item records a temporal sample at that instant instead of at the current time. It is how data arrives with the timestamp it actually had, rather than the timestamp of the load.

The write goes to history only. The property's canonical value is untouched, which is what allows a backfill to run without disturbing what the graph currently says.

## How it behaves

IronGraph separates two clocks. **Event time** is when something happened in the world; it is what a temporal property records and what `AT TIME` and `HISTORY` read. **Write time** is when the database was told, and is the default event time when a write does not say otherwise. Keeping them apart is what lets a correction arrive late without rewriting history, and what lets a backfill land with the timestamps the data actually had.

The target property must be declared temporal; `AT TIME` on an ordinary property is rejected. The instant must fall inside the declared retention window, so retention bounds how far back a backfill can reach.

Samples need not arrive in order. A sample can land between two that are already recorded, and reads afterwards see the corrected series. That is what makes late-arriving data expressible rather than a rewrite.

One statement can write many samples: the `SET` runs once per matched row, so an `UNWIND` over a batch of observations produces one sample per observation, each with its own event time. That is exactly how the `trust` dataset's 35,592 reputation samples were loaded.

## When to use it

Use it for any load or correction whose data carries its own timestamps: importing a history, receiving a delayed feed, or restating a past value that was recorded wrongly.

## How it differs from its neighbours

An ordinary `SET` on a temporal property records a sample at write time *and* updates the canonical value. `SET … AT TIME` records a sample at the given time and leaves the canonical value alone. Reaching for one when you meant the other is the usual cause of a canonical value that disagrees with history.

The query-level `AT TIME` that precedes a query body is a different thing entirely: it chooses a read instant and never affects writes.

## Simple example

One backdated sample, written and read straight back. The canonical value is unchanged by it.

```cypher
USE backfill_example
AT TIME datetime('2025-06-01T00:00:00Z')
MATCH (sensor:Sensor {sensor_id: 'north'})
RETURN sensor.sensor_id AS sensor,
       sensor.celsius AS reading_in_effect
```

Result:

```
sensor | reading_in_effect
-------+------------------
north  | 18.4             

1 row
```

## Advanced example

A batch backfill, and an out-of-order correction landing inside it. Six readings are written from a parameter list in one statement, then a seventh is inserted between two existing samples — and the series reads back in event-time order as though it had always been complete.

```cypher
USE backfill_example
MATCH (sensor:Sensor {sensor_id: 'north'})
HISTORY sensor.celsius
  FROM datetime('2025-01-01T00:00:00Z')
  TO datetime('2026-01-01T00:00:00Z') AS reading
RETURN datetime.fromepoch(reading.time / 1000000000, 0) AS at,
       reading.value AS celsius,
       sensor.celsius AS canonical_value
ORDER BY reading.time
```

Result:

```
at                   | celsius | canonical_value
---------------------+---------+----------------
2025-03-01T09:00:00Z | 18.4    | 22.3           
2025-03-02T09:00:00Z | 19.1    | 22.3           
2025-03-03T09:00:00Z | 20.4    | 22.3           
2025-03-04T09:00:00Z | 21.7    | 22.3           
2025-03-05T09:00:00Z | 22.3    | 22.3           

5 rows
```

## Where it earns its place

- Loading a history with the timestamps it already had.
- Accepting a delayed feed without pretending it arrived on time.
- Correcting a past value by inserting the sample where it belongs.

## Limitations and trade-offs

- The property must be declared temporal.
- The instant must fall inside the declared retention window.
- The canonical value is not updated. A backfilled property reads as its old current value until an ordinary `SET` changes it.
- Nothing enforces that a backdated sample is plausible. The database records the time it is told.

## See also

- [`ALTER … SET TEMPORAL`](./alter-set-temporal.md) to declare the property
- [`HISTORY`](../../clauses/time/history.md) to read the samples back
