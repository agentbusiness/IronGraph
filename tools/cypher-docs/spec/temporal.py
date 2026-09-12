"""Time in Cypher: the clauses and statements that make a graph bitemporal.

None of this exists in standard Cypher. Standard Cypher has temporal *values* — a datetime is a
property like any other — but no notion of a property that remembers what it used to be, and no way
to ask what the graph looked like at a past instant. That is what these pages document.

Everything here runs against `trust`, the Bitcoin OTC rating network. Its 35,592 ratings carry real
timestamps from November 2010 to January 2016, and `Account.reputation` is a declared temporal
property whose history was written with those same timestamps, so an `AT TIME` read genuinely
reconstructs the past rather than replaying a load.
"""

from __future__ import annotations

from model import Example, Family, Page

TWO_CLOCKS = (
    "IronGraph separates two clocks. **Event time** is when something happened in the world; it is "
    "what a temporal property records and what `AT TIME` and `HISTORY` read. **Write time** is when "
    "the database was told, and is the default event time when a write does not say otherwise. "
    "Keeping them apart is what lets a correction arrive late without rewriting history, and what "
    "lets a backfill land with the timestamps the data actually had."
)

CANONICAL_VS_TEMPORAL = (
    "A declared temporal property has two distinct reads that are easy to confuse.\n\n"
    "- Reading it in an ordinary query returns the **canonical** value: whatever an ordinary "
    "`SET` last wrote. Writes made with `AT TIME` do not touch it.\n"
    "- Reading it under `AT TIME`, or through `HISTORY`, returns the **temporal** value: the sample "
    "in effect at that instant.\n\n"
    "The two can differ, and on the `trust` dataset they do: `reputation` was written entirely "
    "through backdated samples, so its canonical value is still the `0.0` set at load while its "
    "temporal value follows the real 2010-2016 curve."
)


def families() -> list[Family]:
    return [time_clauses(), temporal_schema()]


def time_clauses() -> Family:
    return Family(
        path="clauses/time",
        title="Time clauses",
        blurb=(
            "Three clauses that put time into a query: `AT TIME` moves the whole query to a past "
            "instant, `HISTORY` expands one property's samples into rows, and `WINDOW` buckets rows "
            "by an instant they carry.\n\n"
            "They compose. `AT TIME` chooses which graph you are looking at; `HISTORY` turns one "
            "property's past into a row stream; `WINDOW` groups any row stream by time, whether its "
            "instants came from `HISTORY` or from an ordinary datetime property."
        ),
        pages=[
            Page(
                slug="at-time",
                title="`AT TIME`",
                family="clauses/time",
                kind="clause",
                signature="AT TIME <datetime>  (before the query body)",
                summary="Runs the whole query against the graph as it stood at a past instant.",
                dataset="trust",
                standard="extension",
                what=(
                    "`AT TIME` sits ahead of the query body, beside `USE` and `USE LAYER`, and moves "
                    "the entire query to a chosen instant. Every declared temporal property read "
                    "anywhere in that query returns the value in effect then, rather than its "
                    "current one.\n\n"
                    "It is a property of the query, not of a clause. There is no way to read two "
                    "different instants in one statement, which is deliberate: a single query "
                    "always describes one consistent moment."
                ),
                detail=(
                    f"{TWO_CLOCKS}\n\n{CANONICAL_VS_TEMPORAL}\n\n"
                    "Before an entity's first sample, its temporal value is `null` — not its "
                    "eventual first value, and not an error. A time-travelling query therefore "
                    "reports genuine absence for entities that did not yet have the property, which "
                    "is what makes counting them meaningful.\n\n"
                    "Only declared temporal properties travel. Ordinary properties, labels, "
                    "relationships and the existence of nodes are read as they are now. `AT TIME` "
                    "reconstructs the past of declared values, not the past of the whole graph."
                ),
                when=(
                    "Use it to answer \"what did we believe then\": reproducing a past report, "
                    "auditing a decision against the information available at the time, or "
                    "comparing a value now against the same value at a chosen moment."
                ),
                differs=(
                    "`AT TIME` gives one instant's value per entity and keeps the ordinary row "
                    "shape. `HISTORY` gives every sample in a range as its own row, which changes "
                    "the row grain. Use `AT TIME` for a snapshot and `HISTORY` for a trajectory.\n\n"
                    "`AT TIME` also appears in a second, unrelated position: attached to a `SET` "
                    "item it stamps a written sample rather than choosing a read instant. See "
                    "[`SET … AT TIME`](../../statements/temporal/set-at-time.md)."
                ),
                simple=Example(
                    note=(
                        "One account's reputation at four moments in the network's life. Each "
                        "figure is the value in effect on that date, reconstructed from history."
                    ),
                    query=(
                        "USE trust\n"
                        "AT TIME datetime('2012-06-01T00:00:00Z')\n"
                        "MATCH (account:Account {account_id: 35})\n"
                        "RETURN account.account_id AS account,\n"
                        "       account.reputation AS reputation_mid_2012"
                    ),
                ),
                advanced=Example(
                    note=(
                        "How much of the network existed yet. Because a temporal read is `null` "
                        "before an entity's first sample, counting non-null reputations at an "
                        "instant counts the accounts that had been rated by then — a measurement "
                        "that needs no separate created-at field."
                    ),
                    query=(
                        "USE trust\n"
                        "AT TIME datetime('2012-01-01T00:00:00Z')\n"
                        "MATCH (account:Account)\n"
                        "RETURN count(account) AS accounts_in_the_graph,\n"
                        "       count(account.reputation) AS rated_by_2012,\n"
                        "       round(avg(account.reputation) * 1000) / 1000.0 AS mean_reputation,\n"
                        "       min(account.reputation) AS lowest,\n"
                        "       max(account.reputation) AS highest"
                    ),
                ),
                use_cases=[
                    "Reproducing a report exactly as it read on a past date.",
                    "Auditing a decision against what was known when it was taken.",
                    "Counting when entities entered a dataset, without a created-at field.",
                ],
                limits=[
                    "Only declared temporal properties travel in time. Node existence, labels, "
                    "relationships and ordinary properties are always read as they are now.",
                    "One instant per query. Comparing two moments takes two queries, or a `HISTORY` "
                    "range.",
                    "A read before an entity's first sample is `null`. Aggregates skip those rows, "
                    "which is usually right and occasionally surprising.",
                    "The canonical value of the property is a different value and is unaffected.",
                ],
                see_also=[
                    "[`HISTORY`](./history.md) for every sample rather than one instant",
                    "[`ALTER … SET TEMPORAL`](../../statements/temporal/alter-set-temporal.md) to "
                    "declare a property temporal in the first place",
                ],
            ),
            Page(
                slug="history",
                title="`HISTORY`",
                family="clauses/time",
                kind="clause",
                signature="HISTORY <variable>.<property> FROM <datetime> TO <datetime> AS <alias>",
                summary=(
                    "Expands a temporal property's samples in a time range into one row each."
                ),
                dataset="trust",
                standard="extension",
                what=(
                    "`HISTORY` takes a declared temporal property on an already-bound entity and "
                    "produces one row for every sample recorded in the half-open range `FROM`…`TO`. "
                    "Each row binds an alias exposing two fields: `.time`, the sample's event time "
                    "as epoch nanoseconds, and `.value`, what the property was set to.\n\n"
                    "It multiplies rows. One matched account with 535 samples becomes 535 rows, so "
                    "`HISTORY` is where a query's grain changes from entities to observations."
                ),
                detail=(
                    "`.time` is an integer count of nanoseconds since the epoch, not a datetime "
                    "value. Divide by 1,000,000,000 and pass it through `datetime.fromepoch` when a "
                    "reader needs to see it; keep it as an integer when you are only ordering, "
                    "differencing or bucketing.\n\n"
                    "Samples are those written for that entity, in event-time order. An entity with "
                    "no samples in the range contributes no rows at all, so a `HISTORY` clause can "
                    "reduce the row count to zero as easily as multiply it.\n\n"
                    "The property must be declared temporal. Applying `HISTORY` to an ordinary "
                    "property is rejected rather than returning an empty history, so a missing "
                    "declaration fails loudly instead of looking like an absence of data."
                ),
                when=(
                    "Use it whenever the question is about a trajectory rather than a state: how a "
                    "value moved, when it crossed a threshold, how volatile it was, what it did "
                    "between two dates."
                ),
                differs=(
                    "`AT TIME` answers \"what was it then\" with one value and leaves the row grain "
                    "alone. `HISTORY` answers \"what did it do\" and changes the grain to one row "
                    "per sample. `WINDOW` does not read history at all — it buckets whatever rows "
                    "it is given, which is often but not necessarily `HISTORY` output."
                ),
                simple=Example(
                    note=(
                        "The first ten reputation samples recorded for one account, with their "
                        "event times rendered as datetimes."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH (account:Account {account_id: 35})\n"
                        "HISTORY account.reputation\n"
                        "  FROM datetime('2010-01-01T00:00:00Z')\n"
                        "  TO datetime('2017-01-01T00:00:00Z') AS sample\n"
                        "RETURN datetime.fromepoch(sample.time / 1000000000, 0) AS at,\n"
                        "       sample.value AS reputation\n"
                        "ORDER BY sample.time\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The shape of one account's whole reputation history, and how far it "
                        "travelled. Because each sample is a row, ordinary aggregates describe the "
                        "trajectory directly: its range, its spread, and the span of time it covers."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH (account:Account {account_id: 35})\n"
                        "HISTORY account.reputation\n"
                        "  FROM datetime('2010-01-01T00:00:00Z')\n"
                        "  TO datetime('2017-01-01T00:00:00Z') AS sample\n"
                        "RETURN count(sample) AS samples,\n"
                        "       min(sample.value) AS lowest,\n"
                        "       max(sample.value) AS highest,\n"
                        "       round(avg(sample.value) * 1000) / 1000.0 AS mean,\n"
                        "       round(stdev(sample.value) * 1000) / 1000.0 AS spread,\n"
                        "       (max(sample.time) - min(sample.time)) / 86400000000000\n"
                        "         AS days_covered"
                    ),
                ),
                use_cases=[
                    "Charting how a value moved over a period.",
                    "Finding when a value crossed a threshold.",
                    "Measuring volatility with ordinary aggregates over the samples.",
                ],
                limits=[
                    "The property must be declared temporal; an ordinary property is rejected.",
                    "`.time` is epoch nanoseconds, not a datetime. Convert it for display.",
                    "Row grain changes to one row per sample. A broad `MATCH` in front of a long "
                    "history produces a very large row set.",
                    "Only samples inside `FROM`…`TO` appear. There is no implicit sample carrying "
                    "the value in effect at the start of the range.",
                ],
                see_also=[
                    "[`AT TIME`](./at-time.md) for a single instant",
                    "[`WINDOW`](./window.md) to bucket the samples this produces",
                ],
            ),
            Page(
                slug="window",
                title="`WINDOW`",
                family="clauses/time",
                kind="clause",
                signature=(
                    "WINDOW TUMBLING <width> ON <instant> AS <alias>  |  "
                    "WINDOW HOPPING <width> EVERY <step> ON <instant> AS <alias>"
                ),
                summary="Buckets rows into time windows over any instant the rows carry.",
                dataset="trust",
                standard="extension",
                what=(
                    "`WINDOW` assigns each row to one or more time buckets based on an instant the "
                    "row carries, and binds an alias exposing `.start` and `.end` as epoch "
                    "nanoseconds. Those become ordinary grouping columns, so the aggregate that "
                    "follows is grouped per window.\n\n"
                    "`TUMBLING` windows are adjacent and non-overlapping: a row lands in exactly "
                    "one. `HOPPING` windows are as wide as `TUMBLING` ones but start every `EVERY` "
                    "interval, so they overlap and a row lands in several — which is how a moving "
                    "average is expressed."
                ),
                detail=(
                    "The instant can come from anywhere: a `datetime` property on a matched "
                    "relationship, a `HISTORY` sample's `.time`, or any expression producing an "
                    "instant. `WINDOW` does not read temporal history itself and does not require "
                    "a declared temporal property. That is what makes it usable on ordinary "
                    "event-bearing data.\n\n"
                    "`.start` and `.end` are epoch nanoseconds. They are useful as-is for ordering "
                    "and differencing, and should be converted with `datetime.fromepoch` for "
                    "display.\n\n"
                    "Optional modifiers refine the grid. `ALIGN TO` fixes the boundary the windows "
                    "are measured from, so buckets line up with a business day rather than the "
                    "epoch. `TIME ZONE` names the zone the alignment is interpreted in. `EMIT "
                    "EMPTY` keeps windows with no rows, which matters for a chart that must not "
                    "silently close a gap."
                ),
                when=(
                    "Use it for any per-period rollup: activity per month, a moving average, a "
                    "rate over time, or a comparison of the same measure across successive periods. "
                    "It replaces bucketing the timestamp by hand with arithmetic, and unlike hand "
                    "bucketing it can overlap."
                ),
                differs=(
                    "`TUMBLING` and `HOPPING` differ only in overlap: with `EVERY` equal to the "
                    "width, a hopping window is a tumbling one. Bucketing by hand with "
                    "`date.truncate` is equivalent to a tumbling window aligned to the calendar, "
                    "but cannot express overlap at all.\n\n"
                    "A rollup declared with `CREATE ROLLUP` computes the same shape ahead of time "
                    "for a temporal property; `WINDOW` computes it per query over any rows."
                ),
                simple=Example(
                    note=(
                        "Rating activity per calendar year of the network's life, bucketed on the "
                        "timestamp each rating carries."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "WINDOW TUMBLING duration('P365D') ON rating.at AS year\n"
                        "WITH year, rating\n"
                        "RETURN datetime.fromepoch(year.start / 1000000000, 0) AS window_start,\n"
                        "       count(rating) AS ratings,\n"
                        "       round(avg(rating.rating) * 1000) / 1000.0 AS mean_rating\n"
                        "ORDER BY year.start"
                    ),
                ),
                advanced=Example(
                    note=(
                        "A three-month moving view of the network, stepped monthly. Overlapping "
                        "windows smooth the month-to-month noise: each row summarises the quarter "
                        "ending at its start plus two months, and successive rows share two thirds "
                        "of their data."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "WINDOW HOPPING duration('P90D') EVERY duration('P30D')\n"
                        "  ON rating.at AS quarter\n"
                        "WITH quarter, rating\n"
                        "RETURN datetime.fromepoch(quarter.start / 1000000000, 0) AS window_start,\n"
                        "       count(rating) AS ratings,\n"
                        "       round(avg(rating.rating) * 100) / 100.0 AS mean_rating,\n"
                        "       sum(CASE WHEN rating.rating < 0 THEN 1 ELSE 0 END) AS negative\n"
                        "ORDER BY quarter.start\n"
                        "LIMIT 12"
                    ),
                ),
                use_cases=[
                    "Per-period rollups over event-bearing relationships.",
                    "Moving averages and smoothed trends, through overlapping hopping windows.",
                    "Bucketing `HISTORY` samples into periods to chart a trajectory.",
                ],
                limits=[
                    "`.start` and `.end` are epoch nanoseconds, not datetimes.",
                    "A hopping window places each row in several buckets, so counts across all "
                    "windows sum to more than the row count. That is correct and routinely "
                    "misread.",
                    "`EVERY` is only legal with `HOPPING`; a tumbling window with a step is "
                    "rejected.",
                    "Windows are measured from a fixed grid. Use `ALIGN TO` when the boundaries "
                    "must match a business calendar rather than the epoch.",
                ],
                see_also=[
                    "[`HISTORY`](./history.md) to produce samples to window",
                    "[`CREATE ROLLUP`](../../statements/temporal/create-rollup.md) to precompute "
                    "the same shape",
                ],
            ),
        ],
    )


def temporal_schema() -> Family:
    return Family(
        path="statements/temporal",
        title="Temporal schema statements",
        blurb=(
            "Three statements that turn an ordinary property into a remembered one. Declaring a "
            "property temporal is what gives `AT TIME` and `HISTORY` something to read; `SET … AT "
            "TIME` is how a sample gets an event time of its own; a rollup precomputes windowed "
            "aggregates over the history that results."
        ),
        pages=[
            Page(
                slug="alter-set-temporal",
                title="`ALTER … SET TEMPORAL`",
                family="statements/temporal",
                kind="statement",
                signature=(
                    "ALTER NODE|RELATIONSHIP PROPERTY <label>.<property> "
                    "SET TEMPORAL <type> RETENTION <duration>"
                ),
                summary="Declares a property temporal, so its values are remembered rather than replaced.",
                dataset="trust",
                standard="extension",
                what=(
                    "This statement declares that a named property on a label or relationship type "
                    "keeps its history. Once declared, writes can carry an event time, `HISTORY` "
                    "can read the samples back, and `AT TIME` can reconstruct the value at any past "
                    "instant.\n\n"
                    "`RETENTION` bounds how far back history is kept. It is not only a cleanup "
                    "policy: it is also the window inside which a backdated write is accepted. A "
                    "sample older than the retention horizon is rejected."
                ),
                detail=(
                    "The label and the property must already exist in the project's schema. A "
                    "property that has never been written is not yet in the catalogue, so the "
                    "declaration is rejected — write the property once, then declare it. This is the "
                    "single most common surprise with this statement.\n\n"
                    "The declaration is not retrospective. Values written before it are not "
                    "history, and the property's canonical value is left exactly as it was. History "
                    "begins at the declaration.\n\n"
                    "The type names the scalar the samples hold and is checked on write, so a "
                    "declaration is also a type constraint on the temporal series.\n\n"
                    f"{CANONICAL_VS_TEMPORAL}"
                ),
                when=(
                    "Declare a property temporal when its past matters as data rather than as an "
                    "audit trail: a reputation, a price, a status, a score, a reading. If nothing "
                    "will ever ask what it used to be, leave it ordinary — history is not free."
                ),
                differs=(
                    "A temporal property is not a relationship to a timestamped event node. The "
                    "event-node modelling keeps every observation as graph data you can traverse "
                    "and relate; a temporal property keeps a compact series you can read at an "
                    "instant. This dataset uses both: `RATED` relationships carry the events, and "
                    "`reputation` carries the derived series."
                ),
                simple=Example(
                    note=(
                        "Declaring a temporal property on a scratch project. The property is "
                        "written once first, so that it exists in the schema when the declaration "
                        "runs — without that first write the declaration is rejected."
                    ),
                    setup=[
                        "DROP PROJECT IF EXISTS temporal_example CASCADE",
                        "CREATE PROJECT temporal_example",
                        "USE temporal_example CREATE (:Instrument {symbol: 'AAA', price: 0.0})",
                    ],
                    query=(
                        "USE temporal_example\n"
                        "ALTER NODE PROPERTY Instrument.price\n"
                        "  SET TEMPORAL FLOAT RETENTION duration('P3650D')"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The full cycle on that declaration: three backdated samples, then the "
                        "canonical value and two past instants read back beside them. The canonical "
                        "value is still the `0.0` written at creation, because a write with `AT "
                        "TIME` records a sample and leaves the current value alone."
                    ),
                    setup=[
                        "USE temporal_example MATCH (i:Instrument {symbol: 'AAA'}) "
                        "SET i.price = 101.5 AT TIME datetime('2024-01-15T00:00:00Z')",
                        "USE temporal_example MATCH (i:Instrument {symbol: 'AAA'}) "
                        "SET i.price = 118.25 AT TIME datetime('2024-06-01T00:00:00Z')",
                        "USE temporal_example MATCH (i:Instrument {symbol: 'AAA'}) "
                        "SET i.price = 96.0 AT TIME datetime('2025-02-01T00:00:00Z')",
                    ],
                    query=(
                        "USE temporal_example\n"
                        "MATCH (instrument:Instrument {symbol: 'AAA'})\n"
                        "HISTORY instrument.price\n"
                        "  FROM datetime('2023-01-01T00:00:00Z')\n"
                        "  TO datetime('2026-01-01T00:00:00Z') AS sample\n"
                        "RETURN datetime.fromepoch(sample.time / 1000000000, 0) AS at,\n"
                        "       sample.value AS price,\n"
                        "       instrument.price AS canonical_value\n"
                        "ORDER BY sample.time"
                    ),
                ),
                use_cases=[
                    "Prices, scores, reputations and readings whose past is itself data.",
                    "Reproducing a past report without a separate history table.",
                    "Late-arriving corrections that must land at the time they describe.",
                ],
                limits=[
                    "The label and property must already exist; declare after the first write.",
                    "Not retrospective. History starts at the declaration.",
                    "A sample older than `RETENTION` is rejected, so the retention window bounds "
                    "backfill as well as cleanup.",
                    "Documents are never temporal samples: a list or map property cannot be "
                    "declared temporal.",
                ],
                see_also=[
                    "[`SET … AT TIME`](./set-at-time.md) to write a sample with its own event time",
                    "[`AT TIME`](../../clauses/time/at-time.md) to read one back",
                ],
            ),
            Page(
                slug="set-at-time",
                title="`SET … AT TIME`",
                family="statements/temporal",
                kind="clause",
                signature="SET <variable>.<property> = <value> AT TIME <datetime>",
                summary="Writes one history sample stamped with the event time you give it.",
                dataset="trust",
                standard="extension",
                what=(
                    "Appending `AT TIME` to a `SET` item records a temporal sample at that instant "
                    "instead of at the current time. It is how data arrives with the timestamp it "
                    "actually had, rather than the timestamp of the load.\n\n"
                    "The write goes to history only. The property's canonical value is untouched, "
                    "which is what allows a backfill to run without disturbing what the graph "
                    "currently says."
                ),
                detail=(
                    f"{TWO_CLOCKS}\n\n"
                    "The target property must be declared temporal; `AT TIME` on an ordinary "
                    "property is rejected. The instant must fall inside the declared retention "
                    "window, so retention bounds how far back a backfill can reach.\n\n"
                    "Samples need not arrive in order. A sample can land between two that are "
                    "already recorded, and reads afterwards see the corrected series. That is what "
                    "makes late-arriving data expressible rather than a rewrite.\n\n"
                    "One statement can write many samples: the `SET` runs once per matched row, so "
                    "an `UNWIND` over a batch of observations produces one sample per observation, "
                    "each with its own event time. That is exactly how the `trust` dataset's 35,592 "
                    "reputation samples were loaded."
                ),
                when=(
                    "Use it for any load or correction whose data carries its own timestamps: "
                    "importing a history, receiving a delayed feed, or restating a past value that "
                    "was recorded wrongly."
                ),
                differs=(
                    "An ordinary `SET` on a temporal property records a sample at write time *and* "
                    "updates the canonical value. `SET … AT TIME` records a sample at the given "
                    "time and leaves the canonical value alone. Reaching for one when you meant the "
                    "other is the usual cause of a canonical value that disagrees with history.\n\n"
                    "The query-level `AT TIME` that precedes a query body is a different thing "
                    "entirely: it chooses a read instant and never affects writes."
                ),
                simple=Example(
                    note=(
                        "One backdated sample, written and read straight back. The canonical value "
                        "is unchanged by it."
                    ),
                    setup=[
                        "DROP PROJECT IF EXISTS backfill_example CASCADE",
                        "CREATE PROJECT backfill_example",
                        "USE backfill_example CREATE (:Sensor {sensor_id: 'north', celsius: 0.0})",
                        "USE backfill_example ALTER NODE PROPERTY Sensor.celsius "
                        "SET TEMPORAL FLOAT RETENTION duration('P3650D')",
                        "USE backfill_example MATCH (s:Sensor {sensor_id: 'north'}) "
                        "SET s.celsius = 18.4 AT TIME datetime('2025-03-01T09:00:00Z')",
                    ],
                    query=(
                        "USE backfill_example\n"
                        "AT TIME datetime('2025-06-01T00:00:00Z')\n"
                        "MATCH (sensor:Sensor {sensor_id: 'north'})\n"
                        "RETURN sensor.sensor_id AS sensor,\n"
                        "       sensor.celsius AS reading_in_effect"
                    ),
                ),
                advanced=Example(
                    note=(
                        "A batch backfill, and an out-of-order correction landing inside it. Six "
                        "readings are written from a parameter list in one statement, then a "
                        "seventh is inserted between two existing samples — and the series reads "
                        "back in event-time order as though it had always been complete."
                    ),
                    setup=[
                        "USE backfill_example MATCH (s:Sensor {sensor_id: 'north'}) "
                        "SET s.celsius = 19.1 AT TIME datetime('2025-03-02T09:00:00Z')",
                        "USE backfill_example MATCH (s:Sensor {sensor_id: 'north'}) "
                        "SET s.celsius = 21.7 AT TIME datetime('2025-03-04T09:00:00Z')",
                        "USE backfill_example MATCH (s:Sensor {sensor_id: 'north'}) "
                        "SET s.celsius = 22.3 AT TIME datetime('2025-03-05T09:00:00Z')",
                        # Arrives last, belongs third: the gap on 3 March is filled after the fact.
                        "USE backfill_example MATCH (s:Sensor {sensor_id: 'north'}) "
                        "SET s.celsius = 20.4 AT TIME datetime('2025-03-03T09:00:00Z')",
                    ],
                    query=(
                        "USE backfill_example\n"
                        "MATCH (sensor:Sensor {sensor_id: 'north'})\n"
                        "HISTORY sensor.celsius\n"
                        "  FROM datetime('2025-01-01T00:00:00Z')\n"
                        "  TO datetime('2026-01-01T00:00:00Z') AS reading\n"
                        "RETURN datetime.fromepoch(reading.time / 1000000000, 0) AS at,\n"
                        "       reading.value AS celsius,\n"
                        "       sensor.celsius AS canonical_value\n"
                        "ORDER BY reading.time"
                    ),
                ),
                use_cases=[
                    "Loading a history with the timestamps it already had.",
                    "Accepting a delayed feed without pretending it arrived on time.",
                    "Correcting a past value by inserting the sample where it belongs.",
                ],
                limits=[
                    "The property must be declared temporal.",
                    "The instant must fall inside the declared retention window.",
                    "The canonical value is not updated. A backfilled property reads as its old "
                    "current value until an ordinary `SET` changes it.",
                    "Nothing enforces that a backdated sample is plausible. The database records "
                    "the time it is told.",
                ],
                see_also=[
                    "[`ALTER … SET TEMPORAL`](./alter-set-temporal.md) to declare the property",
                    "[`HISTORY`](../../clauses/time/history.md) to read the samples back",
                ],
            ),
            Page(
                slug="create-rollup",
                title="`CREATE ROLLUP`",
                family="statements/temporal",
                kind="statement",
                signature=(
                    "CREATE ROLLUP <name> FOR (<var>:<Label>) ON <var>.<property> "
                    "WINDOW TUMBLING|HOPPING <width> [EVERY <step>] [ALIGN TO <instant>] "
                    "[TIME ZONE '<zone>'] AGGREGATE <function>, …"
                ),
                summary=(
                    "Declares windowed aggregates over a temporal property so they are maintained "
                    "rather than recomputed."
                ),
                dataset="trust",
                standard="extension",
                what=(
                    "A rollup names a windowing and a set of aggregates over one declared temporal "
                    "property, and asks the database to maintain them. It is a declaration of "
                    "derived state, in the same family as an index: it changes what work a later "
                    "query has to do, not what any query returns.\n\n"
                    "The windowing accepts the same `TUMBLING` and `HOPPING` forms as the `WINDOW` "
                    "clause, including `ALIGN TO` and `TIME ZONE`, so a rollup can be declared to "
                    "match exactly the query shape it is meant to serve."
                ),
                detail=(
                    "A rollup is derived state and never graph data. It creates no nodes and no "
                    "relationships, and nothing in a query result names it. Removing a rollup makes "
                    "queries slower and never changes an answer.\n\n"
                    "The aggregates are named as bare identifiers after `AGGREGATE`. Declare the "
                    "ones the queries actually ask for: each is maintained, so an unused aggregate "
                    "is pure cost.\n\n"
                    "Rollups do not appear in `SHOW INDEXES`, which lists index state only."
                ),
                when=(
                    "Declare a rollup when the same windowed aggregate over the same temporal "
                    "property is asked repeatedly — a dashboard panel, a scheduled report, a "
                    "threshold check. For a question asked once, the `WINDOW` clause computes the "
                    "same thing without a standing declaration."
                ),
                differs=(
                    "`WINDOW` computes a windowed aggregate for one query, over any rows, whether "
                    "or not a temporal property is involved. `CREATE ROLLUP` declares one ahead of "
                    "time over a specific temporal property. The clause is the general tool; the "
                    "rollup is the standing optimisation for a shape you already know."
                ),
                simple=Example(
                    note=(
                        "A monthly rollup over the reputation series in the reference dataset. It "
                        "returns no rows: like an index, its effect is on later queries."
                    ),
                    setup=[
                        # A rollup cannot be dropped, so the example declares one in a scratch
                        # project that is rebuilt each time rather than in a reference dataset.
                        "DROP PROJECT IF EXISTS rollup_example CASCADE",
                        "CREATE PROJECT rollup_example",
                        "USE rollup_example CREATE (:Account {account_id: 1, reputation: 0.0})",
                        "USE rollup_example ALTER NODE PROPERTY Account.reputation "
                        "SET TEMPORAL FLOAT RETENTION duration('P3650D')",
                    ],
                    query=(
                        "USE rollup_example\n"
                        "CREATE ROLLUP reputation_monthly FOR (account:Account)\n"
                        "  ON account.reputation\n"
                        "  WINDOW TUMBLING duration('P30D')\n"
                        "  AGGREGATE avg, min, max, count"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The query a rollup is declared to serve. Its shape mirrors the "
                        "declaration — the same property, the same window width, aggregates drawn "
                        "from the declared set — which is what makes the two match up."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH (account:Account {account_id: 35})\n"
                        "HISTORY account.reputation\n"
                        "  FROM datetime('2011-01-01T00:00:00Z')\n"
                        "  TO datetime('2013-01-01T00:00:00Z') AS sample\n"
                        "WINDOW TUMBLING duration('P30D') ON sample.time AS month\n"
                        "WITH month, sample\n"
                        "RETURN datetime.fromepoch(month.start / 1000000000, 0) AS window_start,\n"
                        "       count(sample) AS samples,\n"
                        "       round(avg(sample.value) * 1000) / 1000.0 AS mean,\n"
                        "       min(sample.value) AS lowest,\n"
                        "       max(sample.value) AS highest\n"
                        "ORDER BY month.start\n"
                        "LIMIT 12"
                    ),
                ),
                use_cases=[
                    "Dashboard panels that re-ask the same windowed question.",
                    "Scheduled reports over a temporal property.",
                    "Threshold checks that run continuously over a rolling window.",
                ],
                limits=[
                    "Derived state only: a rollup changes cost, never answers.",
                    "One temporal property per rollup.",
                    "Declared aggregates are maintained whether or not they are used.",
                    "Not listed by `SHOW INDEXES`.",
                ],
                see_also=[
                    "[`WINDOW`](../../clauses/time/window.md) for the per-query form",
                    "[`ALTER … SET TEMPORAL`](./alter-set-temporal.md) for the property it needs",
                ],
            ),
        ],
    )
