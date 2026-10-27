# `localdatetime.transaction`

Read a localdatetime value from the transaction clock. Use this function when the clock boundary matters to your query.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `localdatetime.transaction([timezone])` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

The value uses the explicit transaction's start time. In autocommit it uses the statement time.

## How it behaves

The optional argument is a timezone string, such as `'UTC'`, `'+01:00'` or a supported named timezone.
The default is UTC. The timezone chooses the calendar or clock fields for that instant; local value
types omit timezone information from the returned value. An explicit null timezone produces null.
More than one argument or a non-string, non-null argument is rejected.

## Prerequisites

Create a project with `CREATE PROJECT IF NOT EXISTS worked_example`. These read-only examples need
no imported data or local embedding model.

## Simple example

```cypher
USE worked_example
RETURN localdatetime.transaction() AS result
```

Expected result: one temporal value from the transaction clock. The actual value depends on execution time.

## Advanced example

```cypher
USE worked_example
RETURN localdatetime.transaction('UTC') AS utc_value, localdatetime.transaction(null) AS missing
```

Expected result: a UTC-based temporal value and null in `missing`.

## Limitations and trade-offs

A clock value does not select a historical graph state. Use [AT TIME](../../clauses/time/at-time.md)
for declared temporal-property reads. Use an explicit constructor value when a reproducible test
must not depend on wall-clock time.

## See also

- [Temporal functions](./README.md)
- [`localdatetime`](./localdatetime.md)

