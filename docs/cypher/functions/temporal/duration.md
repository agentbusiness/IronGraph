# `duration`

Construct a duration with calendar-month, day, second and fractional-second components. Use explicit temporal values when your query must be reproducible.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `duration(value)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Construct a duration with calendar-month, day, second and fractional-second components.

## How it behaves

Accepts an ISO duration string or numeric component map. Supported map fields are years, months, weeks, days, hours, minutes, seconds, milliseconds, microseconds and nanoseconds. Components may be fractional; normalization preserves calendar months separately from elapsed seconds. Unknown fields, non-finite values and overflow are errors.

A null argument produces null. Calendar operations do not treat a month as a fixed number of days.
Temporal values are ordinary property values; creating one does not declare a temporal property or
make previous graph states readable.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. No imported dataset
or local embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN duration('P1DT2H') AS result
```

Result:

```
result
----------------------------------------------
{"months":0,"days":1,"seconds":7200,"nanos":0}

1 row
```

Expected result: one row with the constructed or computed temporal value.

## Advanced example

```cypher
USE worked_example
RETURN duration({days: 1, hours: 2, minutes: 30}) AS result
```

Result:

```
result
----------------------------------------------
{"months":0,"days":1,"seconds":9000,"nanos":0}

1 row
```

Expected result: one row with the temporal result. Your client may render temporal values as typed
fields rather than formatted strings; use `toString(result)` in a following `WITH` when you need text.

## Limitations and trade-offs

Use a date for calendar days, a local value for an intentionally unzoned wall clock, and a zoned
datetime for an instant that must be compared across timezones. Read errors for incompatible units
rather than assuming an implicit timezone conversion.

## See also

- [Temporal functions](./README.md)
- [Temporal fields and expressions](../../expressions.md)
- [Temporal property declarations](../../statements/temporal/alter-set-temporal.md)

