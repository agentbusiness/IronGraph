# Temporal functions

Construct dates, clocks and durations or compare them inside ordinary queries. Explicit values
make examples reproducible. Temporal values and temporal-property history are different features.

| Function | Signature | Purpose |
| --- | --- | --- |
| [`date.transaction`](./date-transaction.md) | `date.transaction([timezone])` | Read the transaction clock as a date value. |
| [`date.statement`](./date-statement.md) | `date.statement([timezone])` | Read the statement clock as a date value. |
| [`date.realtime`](./date-realtime.md) | `date.realtime([timezone])` | Read the realtime clock as a date value. |
| [`time.transaction`](./time-transaction.md) | `time.transaction([timezone])` | Read the transaction clock as a time value. |
| [`time.statement`](./time-statement.md) | `time.statement([timezone])` | Read the statement clock as a time value. |
| [`time.realtime`](./time-realtime.md) | `time.realtime([timezone])` | Read the realtime clock as a time value. |
| [`localtime.transaction`](./localtime-transaction.md) | `localtime.transaction([timezone])` | Read the transaction clock as a localtime value. |
| [`localtime.statement`](./localtime-statement.md) | `localtime.statement([timezone])` | Read the statement clock as a localtime value. |
| [`localtime.realtime`](./localtime-realtime.md) | `localtime.realtime([timezone])` | Read the realtime clock as a localtime value. |
| [`datetime.transaction`](./datetime-transaction.md) | `datetime.transaction([timezone])` | Read the transaction clock as a datetime value. |
| [`datetime.statement`](./datetime-statement.md) | `datetime.statement([timezone])` | Read the statement clock as a datetime value. |
| [`datetime.realtime`](./datetime-realtime.md) | `datetime.realtime([timezone])` | Read the realtime clock as a datetime value. |
| [`localdatetime.transaction`](./localdatetime-transaction.md) | `localdatetime.transaction([timezone])` | Read the transaction clock as a localdatetime value. |
| [`localdatetime.statement`](./localdatetime-statement.md) | `localdatetime.statement([timezone])` | Read the statement clock as a localdatetime value. |
| [`localdatetime.realtime`](./localdatetime-realtime.md) | `localdatetime.realtime([timezone])` | Read the realtime clock as a localdatetime value. |
| [`date`](./date.md) | `date([value])` | Construct a calendar date without a timezone or clock. |
| [`date.truncate`](./date-truncate.md) | `date.truncate(unit, value [, fields])` | Truncate a date value to a supported calendar or clock unit. |
| [`time`](./time.md) | `time([value])` | Construct a time of day with a utc offset. |
| [`time.truncate`](./time-truncate.md) | `time.truncate(unit, value [, fields])` | Truncate a time value to a supported calendar or clock unit. |
| [`localtime`](./localtime.md) | `localtime([value])` | Construct a time of day without a timezone. |
| [`localtime.truncate`](./localtime-truncate.md) | `localtime.truncate(unit, value [, fields])` | Truncate a localtime value to a supported calendar or clock unit. |
| [`datetime`](./datetime.md) | `datetime([value])` | Construct a date and time with a timezone. |
| [`datetime.truncate`](./datetime-truncate.md) | `datetime.truncate(unit, value [, fields])` | Truncate a datetime value to a supported calendar or clock unit. |
| [`localdatetime`](./localdatetime.md) | `localdatetime([value])` | Construct a date and time without a timezone. |
| [`localdatetime.truncate`](./localdatetime-truncate.md) | `localdatetime.truncate(unit, value [, fields])` | Truncate a localdatetime value to a supported calendar or clock unit. |
| [`datetime.fromepoch`](./datetime-fromepoch.md) | `datetime.fromepoch(seconds, nanoseconds)` | Construct a UTC datetime from Unix epoch seconds and an additional nanosecond fraction. |
| [`datetime.fromepochmillis`](./datetime-fromepochmillis.md) | `datetime.fromepochmillis(milliseconds)` | Construct a UTC datetime from Unix epoch milliseconds. |
| [`duration`](./duration.md) | `duration(value)` | Construct a duration with calendar-month, day, second and fractional-second components. |
| [`duration.between`](./duration-between.md) | `duration.between(from, to)` | Compute a calendar duration from one temporal value to another. |
| [`duration.indays`](./duration-indays.md) | `duration.indays(from, to)` | Compute a duration expressed in whole calendar days between temporal values. |
| [`duration.inmonths`](./duration-inmonths.md) | `duration.inmonths(from, to)` | Compute a duration expressed in whole calendar months between temporal values. |
| [`duration.inseconds`](./duration-inseconds.md) | `duration.inseconds(from, to)` | Compute an elapsed duration in seconds and nanoseconds between temporal values. |

See [temporal fields](../../expressions.md) for date, clock, timezone, epoch and duration accessors.

