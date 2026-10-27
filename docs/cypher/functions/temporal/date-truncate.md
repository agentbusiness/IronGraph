# `date.truncate`

Truncate a date value to a supported calendar or clock unit. Use explicit temporal values when your query must be reproducible.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `date.truncate(unit, value [, fields])` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Truncate a date value to a supported calendar or clock unit.

## How it behaves

The first argument is a unit string and the second is a compatible temporal value. An optional third map sets supported fields below the truncation boundary. Date units include millennium, century, decade, year, weekYear, quarter, month, week and day; clock units include hour, minute, second, millisecond and microsecond. Use only units meaningful for the target type. Incompatible units, conflicting fields and invalid timezones report errors.

A null argument produces null. Calendar operations do not treat a month as a fixed number of days.
Temporal values are ordinary property values; creating one does not declare a temporal property or
make previous graph states readable.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. No imported dataset
or local embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN date.truncate('month', date('2026-01-02')) AS result
```

Result:

```
result
----------
2026-01-01

1 row
```

Expected result: one row with the constructed or computed temporal value.

## Advanced example

```cypher
USE worked_example
RETURN date.truncate('year', date('2026-01-02')) AS result
```

Result:

```
result
----------
2026-01-01

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

