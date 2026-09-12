# Temporal schema statements

Three statements that turn an ordinary property into a remembered one. Declaring a property temporal is what gives `AT TIME` and `HISTORY` something to read; `SET … AT TIME` is how a sample gets an event time of its own; a rollup precomputes windowed aggregates over the history that results.

| Page | Summary | Standard |
| --- | --- | --- |
| [`ALTER … SET TEMPORAL`](./alter-set-temporal.md) | Declares a property temporal, so its values are remembered rather than replaced. | extension |
| [`SET … AT TIME`](./set-at-time.md) | Writes one history sample stamped with the event time you give it. | extension |
| [`CREATE ROLLUP`](./create-rollup.md) | Declares windowed aggregates over a temporal property so they are maintained rather than recomputed. | extension |
