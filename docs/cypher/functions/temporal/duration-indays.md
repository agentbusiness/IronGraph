# `duration.indays`

Compute a duration expressed in whole calendar days between temporal values. Use explicit temporal values when your query must be reproducible.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `duration.indays(from, to)` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Compute a duration expressed in whole calendar days between temporal values.

## How it behaves

Inputs need compatible date components. The result is a duration, not an integer; read its days field for the count.

A null argument produces null. Calendar operations do not treat a month as a fixed number of days.
Temporal values are ordinary property values; creating one does not declare a temporal property or
make previous graph states readable.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. No imported dataset
or local embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN duration.indays(date('2026-01-01'), date('2026-01-03')) AS result
```

Result:

```
result
-------------------------------------------
{"months":0,"days":2,"seconds":0,"nanos":0}

1 row
```

Expected result: one row with the constructed or computed temporal value.

## Advanced example

```cypher
USE worked_example
RETURN duration.indays(date('2026-01-03'), date('2026-01-01')) AS result
```

Result:

```
result
--------------------------------------------
{"months":0,"days":-2,"seconds":0,"nanos":0}

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

