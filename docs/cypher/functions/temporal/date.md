# `date`

Construct a calendar date without a timezone or clock. Use explicit temporal values when your query must be reproducible.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `date([value])` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | Self-contained values in an existing project |

## What it does

Construct a calendar date without a timezone or clock.

## How it behaves

Accepts zero or one argument. An explicit ISO string or supported component map gives reproducible values; an omitted value uses the query's resolved current time. Compatible temporal values may also be converted. Missing, conflicting or out-of-range fields report an error.

A null argument produces null. Calendar operations do not treat a month as a fixed number of days.
Temporal values are ordinary property values; creating one does not declare a temporal property or
make previous graph states readable.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. No imported dataset
or local embedding model is required.

## Simple example

```cypher
USE worked_example
RETURN date('2026-01-02') AS result
```

Result:

```
result
----------
2026-01-02

1 row
```

Expected result: one row with the constructed or computed temporal value.

## Advanced example

```cypher
USE worked_example
RETURN date({year: 2026, month: 1, day: 2}) AS result
```

Result:

```
result
----------
2026-01-02

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

