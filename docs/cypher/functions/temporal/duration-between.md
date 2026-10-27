# `duration.between`

Compute a calendar duration from one temporal value to another. Use explicit temporal values when your query must be reproducible.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `duration.between(from, to)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Compute a calendar duration from one temporal value to another.

## How it behaves

Returns month, day and remaining time components when dates are available. Reversing endpoints changes the sign. Inputs must have compatible temporal components.

A null argument produces null. Calendar operations do not treat a month as a fixed number of days.
Temporal values are ordinary property values; creating one does not declare a temporal property or
make previous graph states readable.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. No imported dataset
or local embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN duration.between(date('2026-01-01'), date('2026-02-03')) AS result
```

Result:

```
result
-------------------------------------------
{"months":1,"days":2,"seconds":0,"nanos":0}

1 row
```

Expected result: one row with the constructed or computed temporal value.

## Advanced example

```cypher
USE worked_example
RETURN duration.between(localtime('10:00:00'), localtime('11:30:00')) AS result
```

Result:

```
result
----------------------------------------------
{"months":0,"days":0,"seconds":5400,"nanos":0}

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

