# `ALTER … SET TEMPORAL`

> Declares a property temporal, so its values are remembered rather than replaced.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `ALTER NODE\|RELATIONSHIP PROPERTY <label>.<property> SET TEMPORAL <type> RETENTION <duration>` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

This statement declares that a named property on a label or relationship type keeps its history. Once declared, writes can carry an event time, `HISTORY` can read the samples back, and `AT TIME` can reconstruct the value at any past instant.

`RETENTION` bounds how far back history is kept. It is not only a cleanup policy: it is also the window inside which a backdated write is accepted. A sample older than the retention horizon is rejected.

## How it behaves

The label and the property must already exist in the project's schema. A property that has never been written is not yet in the catalogue, so the declaration is rejected — write the property once, then declare it. This is the single most common surprise with this statement.

The declaration is not retrospective. Values written before it are not historical samples. A declared property with no retained samples reads as `null`; history begins with its temporal samples.

The type names the scalar the samples hold and is checked on write, so a declaration is also a type constraint on the temporal series.

A declared temporal property has two distinct reads that are easy to confuse.

- Reading it in an ordinary query returns the **current sample**: the latest retained sample in event-time order.
- Reading it under `AT TIME` returns the sample in effect at the selected instant. `HISTORY` returns samples in its requested range.

Current and historical samples can differ. Backdated samples participate in both reads according to their event times.

## When to use it

Declare a property temporal when its past matters as data rather than as an audit trail: a reputation, a price, a status, a score, a reading. If nothing will ever ask what it used to be, leave it ordinary — history is not free.

## How it differs from its neighbours

A temporal property is not a relationship to a timestamped event node. The event-node modelling keeps every observation as graph data you can traverse and relate; a temporal property keeps a compact series you can read at an instant. This dataset uses both: `RATED` relationships carry the events, and `reputation` carries the derived series.

## Simple example

Declaring a temporal property on a scratch project. The property is written once first, so that it exists in the schema when the declaration runs — without that first write the declaration is rejected.

```cypher
USE temporal_example
ALTER NODE PROPERTY Instrument.price
  SET TEMPORAL FLOAT RETENTION duration('P3650D')
```

Result:

```
No rows returned. 0 changes committed.
```

## Advanced example

Read three backdated samples alongside the current sample. The current price is `96`, the latest value in event-time order.

```cypher
USE temporal_example
MATCH (instrument:Instrument {symbol: 'AAA'})
HISTORY instrument.price
  FROM datetime('2023-01-01T00:00:00Z')
  TO datetime('2026-01-01T00:00:00Z') AS sample
RETURN datetime.fromepoch(sample.time / 1000000000, 0) AS at,
       sample.value AS price,
       instrument.price AS current_value
ORDER BY sample.time
```

Result:

```
at                   | price  | current_value
---------------------+--------+--------------
2024-01-15T00:00:00Z | 101.5  | 96             
2024-06-01T00:00:00Z | 118.25 | 96             
2025-02-01T00:00:00Z | 96     | 96             

3 rows
```

## Where it earns its place

- Prices, scores, reputations and readings whose past is itself data.
- Reproducing a past report without a separate history table.
- Late-arriving corrections that must land at the time they describe.

## Limitations and trade-offs

- The label and property must already exist; declare after the first write.
- Not retrospective. History starts at the declaration.
- A sample older than `RETENTION` is rejected, so the retention window bounds backfill as well as cleanup.
- Documents are never temporal samples: a list or map property cannot be declared temporal.

## See also

- [`SET … AT TIME`](./set-at-time.md) to write a sample with its own event time
- [`AT TIME`](../../clauses/time/at-time.md) to read one back
