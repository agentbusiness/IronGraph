"""Aggregate functions.

Six of the twelve are standard Cypher. The dispersion and percentile aggregates are the additions,
and they are what turn a graph query into a measurement without a second system in the loop.

Every example runs against `trust`, the Bitcoin OTC rating network: 5,881 accounts and 35,592
ratings, each an integer from -10 to +10 carrying a real timestamp between 2010 and 2016. It is
chosen because it has a genuine numeric measure to aggregate and a real timeline to group by.
"""

from __future__ import annotations

from model import Example, Family, Page

GRAIN = (
    "An aggregate consumes the rows that reach it and returns one row per distinct combination of "
    "the non-aggregated expressions projected beside it. Those expressions are the grouping key: "
    "nothing declares it, and adding a column to the projection silently changes the grain. When a "
    "projection contains only aggregates, every incoming row collapses into a single result row."
)

NULLS = (
    "Null inputs are skipped rather than treated as zero, so an aggregate reports on the rows that "
    "actually carried a value. Over rows that are all null, or over no rows at all, the result is "
    "null rather than an error — with the exception of `count`, which counts."
)


def families() -> list[Family]:
    return [basic(), dispersion(), percentiles()]


def _page(
    slug: str,
    title: str,
    signature: str,
    summary: str,
    what: str,
    when: str,
    differs: str,
    simple: Example,
    advanced: Example,
    use_cases: list[str],
    limits: list[str],
    see_also: list[str],
    standard: str = "standard",
    detail: str = "",
    family: str = "functions/aggregation",
) -> Page:
    return Page(
        slug=slug,
        title=title,
        family=family,
        kind="aggregate",
        signature=signature,
        summary=summary,
        dataset="trust",
        standard=standard,
        what=what,
        detail=detail or f"{GRAIN}\n\n{NULLS}",
        when=when,
        differs=differs,
        simple=simple,
        advanced=advanced,
        use_cases=use_cases,
        limits=limits,
        see_also=see_also,
    )


# --------------------------------------------------------------------------------------------


def basic() -> Family:
    return Family(
        path="functions/aggregation",
        title="Aggregate functions",
        blurb=(
            "Twelve aggregates. `count`, `sum`, `avg`, `min`, `max` and `collect` are standard "
            "Cypher; the variance, standard deviation and percentile aggregates go beyond it. All "
            "twelve group the same way and skip nulls the same way, so the choice between them is "
            "purely about what you want measured.\n\n"
            "Every example on these pages runs against the `trust` dataset."
        ),
        pages=[
            _page(
                slug="count",
                title="`count`",
                signature="count(expression) | count(*)",
                summary="How many rows reached this point, or how many carried a value.",
                what=(
                    "`count(*)` counts rows. `count(expression)` counts the rows where the "
                    "expression is not null. The two differ exactly by the number of nulls, and "
                    "that difference is often the measurement you actually want.\n\n"
                    "`count` is the only aggregate that returns a number rather than null when it "
                    "sees nothing: over an empty input it returns `0`."
                ),
                when=(
                    "Use `count(*)` for volume and `count(expression)` for coverage. Use both "
                    "together when you need to know how complete a property is across the rows a "
                    "pattern produced."
                ),
                differs=(
                    "`count` is the only aggregate that returns `0` rather than null on empty "
                    "input, which makes it the safe one to divide *by* only after checking it is "
                    "not zero. `collect` keeps the values instead of counting them."
                ),
                simple=Example(
                    note="Volume and coverage of the rating network in one row.",
                    query=(
                        "USE trust\n"
                        "MATCH (rater:Account)-[rating:RATED]->(rated:Account)\n"
                        "RETURN count(*) AS ratings,\n"
                        "       count(rating.rating) AS with_a_score,\n"
                        "       count(rating.at) AS with_a_timestamp,\n"
                        "       count(rated.reputation) AS rated_has_reputation"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The long tail of participation. Grouping accounts by how many ratings they "
                        "gave, then counting the groups, turns a per-account count into the shape "
                        "of the whole population — the two levels of counting that most "
                        "distribution questions need."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH (rater:Account)-[rating:RATED]->()\n"
                        "WITH rater, count(rating) AS given\n"
                        "WITH CASE\n"
                        "       WHEN given = 1 THEN '1'\n"
                        "       WHEN given <= 5 THEN '2-5'\n"
                        "       WHEN given <= 20 THEN '6-20'\n"
                        "       WHEN given <= 100 THEN '21-100'\n"
                        "       ELSE 'over 100'\n"
                        "     END AS band, given\n"
                        "RETURN band, count(*) AS accounts, sum(given) AS ratings_given\n"
                        "ORDER BY ratings_given DESC"
                    ),
                ),
                use_cases=[
                    "Measuring how many rows a pattern actually produced.",
                    "Measuring property coverage as the gap between `count(*)` and `count(x)`.",
                    "Counting distinct participants with `count(DISTINCT ...)`.",
                ],
                limits=[
                    "`count(*)` counts pattern matches, not distinct entities. A node matched by "
                    "several paths is counted several times.",
                    "`count(DISTINCT ...)` must retain the distinct values it has seen, so it costs "
                    "more than a plain count on a high-cardinality expression.",
                ],
                see_also=[
                    "[`collect`](./collect.md) to keep the values rather than count them",
                    "[`sum`](./sum.md) to total them",
                ],
            ),
            _page(
                slug="sum",
                title="`sum`",
                signature="sum(expression)",
                summary="The total of the numeric values in the rows that reached this point.",
                what=(
                    "`sum` adds up numeric values, skipping nulls. Over an input with no numeric "
                    "values it returns null.\n\n"
                    "On signed data — a rating from -10 to +10, a balance, a delta — a sum is a net "
                    "position, and a net position near zero can mean either no activity or a great "
                    "deal of activity in both directions. Pair it with `count` whenever the sign "
                    "varies."
                ),
                when=(
                    "Use it for quantities that genuinely add: totals, net positions, accumulated "
                    "weight along a path. Do not use it for rates or ratios, which do not."
                ),
                differs=(
                    "`avg` divides the same total by the count and so hides volume; `sum` keeps "
                    "volume and hides typicality. Reporting both costs nothing and prevents the "
                    "most common misreading of either."
                ),
                simple=Example(
                    note="Net trust in the network, alongside the volume that produced it.",
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "RETURN sum(rating.rating) AS net_trust,\n"
                        "       count(rating) AS ratings,\n"
                        "       round(avg(rating.rating) * 10000) / 10000.0 AS mean_rating"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Accounts whose net trust is near zero for opposite reasons. Splitting the "
                        "sum into its positive and negative halves separates an account nobody has "
                        "an opinion about from one the network actively disagrees over — a "
                        "distinction the net figure alone destroys."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->(rated:Account)\n"
                        "WITH rated,\n"
                        "     sum(rating.rating) AS net,\n"
                        "     sum(CASE WHEN rating.rating > 0 THEN rating.rating ELSE 0 END) AS positive,\n"
                        "     sum(CASE WHEN rating.rating < 0 THEN -rating.rating ELSE 0 END) AS negative,\n"
                        "     count(rating) AS ratings\n"
                        "WHERE ratings >= 20 AND net > -5 AND net < 5\n"
                        "RETURN rated.account_id AS account, ratings, net, positive, negative,\n"
                        "       round(100.0 * negative / (positive + negative) * 10) / 10.0 AS percent_negative\n"
                        "ORDER BY negative DESC, account\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Totals and net positions over signed measures.",
                    "Accumulating a weight along matched relationships.",
                    "Splitting a total into signed halves with `CASE` to expose disagreement.",
                ],
                limits=[
                    "A sum over signed values conceals volume. Report `count` beside it.",
                    "Summing rates, ratios or percentages produces a number with no meaning.",
                    "Floating-point sums depend on row order for their last digits; round before "
                    "comparing for equality.",
                ],
                see_also=[
                    "[`avg`](./avg.md) for the same total per row",
                    "[`count`](./count.md) for the volume behind it",
                ],
            ),
            _page(
                slug="avg",
                title="`avg`",
                signature="avg(expression)",
                summary="The arithmetic mean of the numeric values that reached this point.",
                what=(
                    "`avg` returns the sum of the non-null numeric values divided by how many there "
                    "were, always as a float. Over no numeric values it returns null rather than "
                    "zero, which keeps an empty group distinguishable from a group averaging zero."
                ),
                when=(
                    "Use it when you want a typical value and the data is roughly symmetric. On "
                    "skewed data — which most graph measurements are — the mean sits away from "
                    "anything typical, and `median` describes the population better."
                ),
                differs=(
                    "`median`, spelled `percentilecont(x, 0.5)`, is unmoved by extremes; `avg` is "
                    "dragged by every one of them. Where the two disagree, the data is skewed, and "
                    "the size of the disagreement is a useful measurement in itself."
                ),
                simple=Example(
                    note="Mean and median rating, side by side.",
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "RETURN count(rating) AS ratings,\n"
                        "       round(avg(rating.rating) * 10000) / 10000.0 AS mean,\n"
                        "       percentilecont(rating.rating, 0.5) AS median,\n"
                        "       min(rating.rating) AS lowest,\n"
                        "       max(rating.rating) AS highest"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Mean against median per year. The gap between them measures the skew of "
                        "each year's ratings: where the mean sits well below the median, a minority "
                        "of strongly negative ratings is pulling the average away from the typical "
                        "experience."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "WINDOW TUMBLING duration('P365D') ON rating.at AS year\n"
                        "WITH year, rating\n"
                        "RETURN year.start AS window_start,\n"
                        "       count(rating) AS ratings,\n"
                        "       round(avg(rating.rating) * 1000) / 1000.0 AS mean,\n"
                        "       percentilecont(rating.rating, 0.5) AS median,\n"
                        "       round(avg(rating.rating) - percentilecont(rating.rating, 0.5) * 1000) / 1000.0 AS skew\n"
                        "ORDER BY window_start"
                    ),
                ),
                use_cases=[
                    "A typical value over roughly symmetric data.",
                    "Comparing groups on a common scale.",
                    "Detecting skew by differencing against the median.",
                ],
                limits=[
                    "Extremes move the mean without limit. One outlier in a small group dominates "
                    "it.",
                    "A mean over a handful of rows is not a measurement. Report `count` beside it.",
                    "Averaging values that are themselves averages weights the groups wrongly.",
                ],
                see_also=[
                    "[`percentilecont`](../aggregation/percentilecont.md) for the median",
                    "[`stdev`](./stdev.md) for how spread the values are",
                ],
            ),
            _page(
                slug="min",
                title="`min`",
                signature="min(expression)",
                summary="The smallest value among the rows that reached this point.",
                what=(
                    "`min` returns the smallest non-null value, using the ordering of the value's "
                    "own type: numeric for numbers, chronological for temporal values, "
                    "lexicographic for strings. Over no non-null values it returns null.\n\n"
                    "Applied to a timestamp, `min` is the first time something happened — which is "
                    "how you find an entity's beginning without keeping a separate field for it."
                ),
                when=(
                    "Use it for the extreme itself: the worst rating, the earliest event, the "
                    "cheapest route. Use it on a timestamp whenever you need a first-seen date."
                ),
                differs=(
                    "`min` gives the value at the extreme but not the row it came from. When you "
                    "need the whole row, order and limit instead, or `collect` and index into the "
                    "result."
                ),
                simple=Example(
                    note=(
                        "The span of the rating network, in both value and time. `min` over a "
                        "datetime is the first rating ever recorded."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "RETURN min(rating.rating) AS lowest_rating,\n"
                        "       max(rating.rating) AS highest_rating,\n"
                        "       min(rating.at) AS first_rating,\n"
                        "       max(rating.at) AS last_rating"
                    ),
                ),
                advanced=Example(
                    note=(
                        "How long each account's rating history runs. `min` and `max` over the "
                        "timestamp give first and last activity, and their difference is a lifetime "
                        "— computed here for the accounts with the longest histories in the network."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH (rater:Account)-[rating:RATED]->()\n"
                        "WITH rater, count(rating) AS ratings,\n"
                        "     min(rating.at_epoch) AS first_epoch,\n"
                        "     max(rating.at_epoch) AS last_epoch\n"
                        "WHERE ratings >= 50\n"
                        "RETURN rater.account_id AS account, ratings,\n"
                        "       (last_epoch - first_epoch) / 86400 AS active_days,\n"
                        "       round(1.0 * ratings * 86400 / (last_epoch - first_epoch) * 100) / 100.0\n"
                        "         AS ratings_per_day\n"
                        "ORDER BY active_days DESC, account\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "First-seen timestamps without a dedicated field.",
                    "The worst or cheapest value in a group.",
                    "Bounding a range together with `max`.",
                ],
                limits=[
                    "Returns the value, never the row that held it.",
                    "Mixing types in one `min` compares across type orderings and is rarely "
                    "meaningful.",
                ],
                see_also=["[`max`](./max.md) for the other end of the range"],
            ),
            _page(
                slug="max",
                title="`max`",
                signature="max(expression)",
                summary="The largest value among the rows that reached this point.",
                what=(
                    "`max` returns the largest non-null value under the ordering of the value's own "
                    "type, and null when nothing non-null reached it. On a timestamp it is the most "
                    "recent event, which makes it the natural way to express recency."
                ),
                when=(
                    "Use it for the best, the largest, or the latest. Paired with `min` it gives a "
                    "range; paired with `avg` it shows how far the extreme sits from typical."
                ),
                differs=(
                    "Like `min`, `max` yields a value and not a row. For the row itself, order "
                    "descending and limit."
                ),
                simple=Example(
                    note="The most recent activity of the ten busiest raters.",
                    query=(
                        "USE trust\n"
                        "MATCH (rater:Account)-[rating:RATED]->()\n"
                        "WITH rater, count(rating) AS ratings, max(rating.at) AS last_seen\n"
                        "RETURN rater.account_id AS account, ratings, last_seen\n"
                        "ORDER BY ratings DESC, account\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "How far the best rating an account received sits above its typical one. "
                        "`max` beside `avg` separates accounts that are consistently well regarded "
                        "from accounts with one enthusiastic supporter and an otherwise ordinary "
                        "record."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->(rated:Account)\n"
                        "WITH rated, count(rating) AS ratings,\n"
                        "     max(rating.rating) AS best,\n"
                        "     avg(rating.rating) AS mean,\n"
                        "     stdev(rating.rating) AS spread\n"
                        "WHERE ratings >= 30 AND spread > 0\n"
                        "RETURN rated.account_id AS account, ratings, best,\n"
                        "       round(mean * 100) / 100.0 AS mean,\n"
                        "       round(spread * 100) / 100.0 AS spread,\n"
                        "       round((best - mean) / spread * 100) / 100.0 AS best_in_deviations\n"
                        "ORDER BY best_in_deviations DESC, account\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Last-seen timestamps and recency.",
                    "The best value in a group.",
                    "Measuring how exceptional an extreme is, against `avg` and `stdev`.",
                ],
                limits=[
                    "Returns the value, never the row that held it.",
                    "A single extreme row can make a group look unlike itself; check `count` and "
                    "spread before drawing conclusions.",
                ],
                see_also=["[`min`](./min.md) for the other end of the range"],
            ),
            _page(
                slug="collect",
                title="`collect`",
                signature="collect(expression)",
                summary="Gathers the values that reached this point into one list.",
                what=(
                    "`collect` returns a list of the non-null values in the group, in the order the "
                    "rows arrived. Over no rows it returns an empty list rather than null — the one "
                    "aggregate besides `count` that has a meaningful empty result.\n\n"
                    "It is the aggregate that changes shape rather than reducing to a number: many "
                    "rows become one row holding many values, which the list functions can then "
                    "work on."
                ),
                when=(
                    "Use it to fold a one-to-many relationship into one row per parent, and to "
                    "assemble an ordered set of values you want to index into, slice, or compare "
                    "against another group's."
                ),
                differs=(
                    "Every other aggregate throws the individual values away. `collect` keeps them, "
                    "which is why it is the expensive one: the whole group is held in memory as a "
                    "list. Reach for it when you need the values, and for `count` or `sum` when you "
                    "only need a number about them."
                ),
                simple=Example(
                    note=(
                        "The accounts each of the busiest raters rated most negatively, folded into "
                        "one row each."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH (rater:Account)-[rating:RATED]->(rated:Account)\n"
                        "WHERE rating.rating <= -8\n"
                        "WITH rater, collect(rated.account_id) AS distrusted\n"
                        "RETURN rater.account_id AS account,\n"
                        "       size(distrusted) AS strongly_distrusts,\n"
                        "       distrusted[0..5] AS first_five\n"
                        "ORDER BY strongly_distrusts DESC, account\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Pairs of accounts that rated each other. Collecting each account's outbound "
                        "targets, then testing membership from the other side, expresses "
                        "reciprocity as a list operation rather than a second traversal."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH (a:Account)-[out:RATED]->(b:Account)\n"
                        "WHERE out.rating >= 8\n"
                        "WITH a, collect(b.account_id) AS endorsed\n"
                        "WITH a, endorsed WHERE size(endorsed) >= 5\n"
                        "MATCH (b:Account)-[back:RATED]->(a)\n"
                        "WHERE b.account_id IN endorsed\n"
                        "WITH a, endorsed, count(back) AS mutual, avg(back.rating) AS returned\n"
                        "RETURN a.account_id AS account,\n"
                        "       size(endorsed) AS strongly_endorsed,\n"
                        "       mutual AS endorsed_back_by,\n"
                        "       round(1000.0 * mutual / size(endorsed)) / 10.0 AS reciprocity_percent,\n"
                        "       round(returned * 100) / 100.0 AS mean_rating_returned\n"
                        "ORDER BY reciprocity_percent DESC, account\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Folding a one-to-many relationship into one row per parent.",
                    "Building a list to test membership against with `IN`.",
                    "Assembling an ordered sample you can slice with a range.",
                ],
                limits=[
                    "The whole group is held in memory. Collecting over an unbounded pattern is the "
                    "most reliable way to make a query expensive.",
                    "Order is the order rows arrived, which is not guaranteed to be meaningful "
                    "unless you ordered them first.",
                    "Nulls are dropped, so the list can be shorter than the group.",
                ],
                see_also=[
                    "[`count`](./count.md) when only the size matters",
                    "[`size`](../collection/size.md) to measure the resulting list",
                ],
            ),
        ],
    )


def dispersion() -> Family:
    return Family(
        path="functions/aggregation/dispersion",
        title="Dispersion aggregates",
        blurb=(
            "Four aggregates that measure spread rather than position: how far the values in a "
            "group sit from their own mean. `stdev` and `stdevp` are widely implemented; "
            "`variance` and `variancep` are IronGraph additions that expose the squared form "
            "directly, so a query that needs to combine or weight dispersions does not have to "
            "square a standard deviation back up."
        ),
        pages=[
            _page(
                slug="stdev",
                title="`stdev`",
                family="functions/aggregation/dispersion",
                standard="extended",
                signature="stdev(expression)",
                summary="Sample standard deviation: typical distance from the mean, in the original units.",
                what=(
                    "`stdev` returns the sample standard deviation of the non-null numeric values "
                    "in the group — the square root of the sample variance, divided by one less "
                    "than the count. It is expressed in the same units as the input, which is what "
                    "makes it directly comparable to the mean.\n\n"
                    "Use the sample form when the rows are a sample of some larger process, which "
                    "is the usual case for observational data."
                ),
                when=(
                    "Use it whenever you report a mean over data that might not be tightly "
                    "clustered. A mean without a spread is an assertion that the group is "
                    "homogeneous, and `stdev` is the cheapest way to check that assertion."
                ),
                differs=(
                    "`stdevp` divides by the count rather than the count less one, and is correct "
                    "when the rows are the entire population rather than a sample of it. The two "
                    "differ noticeably on small groups and negligibly on large ones. `variance` is "
                    "the same quantity before the square root, in squared units."
                ),
                simple=Example(
                    note=(
                        "Spread of ratings overall. A standard deviation larger than the mean says "
                        "the ratings are not clustered around it at all."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "RETURN count(rating) AS ratings,\n"
                        "       round(avg(rating.rating) * 1000) / 1000.0 AS mean,\n"
                        "       round(stdev(rating.rating) * 1000) / 1000.0 AS sample_stdev,\n"
                        "       round(stdevp(rating.rating) * 1000) / 1000.0 AS population_stdev"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The accounts the network cannot agree about. A high mean with a low spread "
                        "is a solid reputation; the same mean with a high spread is a contested "
                        "one. Dividing the spread by the mean gives a coefficient of variation that "
                        "makes accounts with different reputations comparable."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->(rated:Account)\n"
                        "WITH rated, count(rating) AS ratings,\n"
                        "     avg(rating.rating) AS mean,\n"
                        "     stdev(rating.rating) AS spread\n"
                        "WHERE ratings >= 25 AND mean > 0.5\n"
                        "RETURN rated.account_id AS account, ratings,\n"
                        "       round(mean * 100) / 100.0 AS mean_rating,\n"
                        "       round(spread * 100) / 100.0 AS spread,\n"
                        "       round(spread / mean * 100) / 100.0 AS coefficient_of_variation\n"
                        "ORDER BY coefficient_of_variation DESC, account\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Qualifying a mean with how much the values actually vary.",
                    "Finding contested entities: same average, much wider spread.",
                    "Expressing a value as a number of deviations from its group's mean.",
                ],
                limits=[
                    "Undefined for a single row; a group of one has no sample spread.",
                    "Assumes the values are a sample. Use `stdevp` for a complete population.",
                    "Like the mean, it is pulled by extremes. On heavily skewed data an "
                    "interquartile range built from `percentilecont` describes spread better.",
                ],
                see_also=[
                    "[`stdevp`](./stdevp.md) for the population form",
                    "[`variance`](./variance.md) for the squared form",
                ],
            ),
            _page(
                slug="stdevp",
                title="`stdevp`",
                family="functions/aggregation/dispersion",
                standard="extended",
                signature="stdevp(expression)",
                summary="Population standard deviation: spread when the rows are everything, not a sample.",
                what=(
                    "`stdevp` returns the population standard deviation: the square root of the "
                    "mean squared deviation, dividing by the count rather than the count less one. "
                    "Unlike `stdev` it is defined for a single row, where it is `0`."
                ),
                when=(
                    "Use it when the rows in the group are the complete population you are "
                    "describing — every rating an account ever received, every relationship in the "
                    "graph — rather than a sample drawn from something larger."
                ),
                differs=(
                    "The only difference from `stdev` is the divisor. `stdevp` is always the "
                    "smaller of the two, and the gap matters only on small groups. Choosing between "
                    "them is a statement about what the rows represent, not about the arithmetic."
                ),
                simple=Example(
                    note=(
                        "Where the sample and population forms diverge. The gap is large on small "
                        "groups and vanishes on large ones."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->(rated:Account)\n"
                        "WITH rated, count(rating) AS ratings,\n"
                        "     stdev(rating.rating) AS sample,\n"
                        "     stdevp(rating.rating) AS population\n"
                        "WHERE ratings >= 2\n"
                        "WITH CASE\n"
                        "       WHEN ratings <= 3 THEN '2-3'\n"
                        "       WHEN ratings <= 10 THEN '4-10'\n"
                        "       WHEN ratings <= 50 THEN '11-50'\n"
                        "       ELSE 'over 50'\n"
                        "     END AS band, sample, population\n"
                        "RETURN band, count(*) AS accounts,\n"
                        "       round(avg(sample) * 10000) / 10000.0 AS mean_sample_stdev,\n"
                        "       round(avg(population) * 10000) / 10000.0 AS mean_population_stdev,\n"
                        "       round(avg(sample) - avg(population) * 10000) / 10000.0 AS gap\n"
                        "ORDER BY gap DESC"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Standardising a rating against the population it belongs to. Each "
                        "account's complete set of received ratings is a population, so `stdevp` is "
                        "the correct divisor, and the resulting z-score says how unusual the "
                        "harshest rating each account received was relative to its own record."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->(rated:Account)\n"
                        "WITH rated, count(rating) AS ratings,\n"
                        "     avg(rating.rating) AS mean,\n"
                        "     stdevp(rating.rating) AS spread,\n"
                        "     min(rating.rating) AS harshest\n"
                        "WHERE ratings >= 25 AND spread > 0\n"
                        "RETURN rated.account_id AS account, ratings,\n"
                        "       round(mean * 100) / 100.0 AS mean_rating,\n"
                        "       harshest,\n"
                        "       round((harshest - mean) / spread * 100) / 100.0 AS harshest_z_score\n"
                        "ORDER BY harshest_z_score, account\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Describing a complete population rather than a sample of one.",
                    "Standardising values into z-scores within their own group.",
                    "Groups small enough that the sample correction would distort the result.",
                ],
                limits=[
                    "Returns `0` for a single row, which is arithmetically right and easy to "
                    "misread as \"no variation observed\".",
                    "Understates spread if the rows really are a sample.",
                ],
                see_also=["[`stdev`](./stdev.md) for the sample form"],
            ),
            _page(
                slug="variance",
                title="`variance`",
                family="functions/aggregation/dispersion",
                standard="extension",
                signature="variance(expression)",
                summary="Sample variance: the squared spread, before the square root.",
                what=(
                    "`variance` returns the sample variance of the non-null numeric values — the "
                    "mean squared deviation from the mean, divided by one less than the count. It "
                    "is `stdev` squared, in squared units.\n\n"
                    "Standard Cypher offers only the standard deviation. Exposing the variance "
                    "directly matters because variances combine and standard deviations do not: "
                    "adding two variances is meaningful, adding two standard deviations is not."
                ),
                when=(
                    "Use it when the value feeds further arithmetic — pooling dispersions across "
                    "groups, weighting them, or decomposing total variation into within-group and "
                    "between-group parts. Report `stdev` when a person is going to read the number."
                ),
                differs=(
                    "`stdev` is the square root of this and is in the original units, which makes "
                    "it the one to display. `variancep` uses the population divisor. The choice "
                    "between variance and standard deviation is about what happens next to the "
                    "number, not about what it measures."
                ),
                simple=Example(
                    note="Variance and standard deviation of the same values, related by a square root.",
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "RETURN count(rating) AS ratings,\n"
                        "       round(variance(rating.rating) * 10000) / 10000.0 AS sample_variance,\n"
                        "       round(variancep(rating.rating) * 10000) / 10000.0 AS population_variance,\n"
                        "       round(stdev(rating.rating) * 10000) / 10000.0 AS sample_stdev,\n"
                        "       round(sqrt(variance(rating.rating)) * 10000) / 10000.0 AS sqrt_of_variance"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Decomposing total variation into the part explained by which account is "
                        "being rated and the part that remains inside each account's own ratings. "
                        "This is the calculation variances exist for: weighting each group's "
                        "variance by its size and pooling them is only valid in squared units."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->(rated:Account)\n"
                        "WITH rated, count(rating) AS ratings,\n"
                        "     avg(rating.rating) AS group_mean,\n"
                        "     variancep(rating.rating) AS group_variance\n"
                        "WHERE ratings >= 10\n"
                        "WITH sum(ratings) AS total_ratings,\n"
                        "     count(*) AS accounts,\n"
                        "     sum(ratings * group_variance) AS weighted_within,\n"
                        "     variancep(group_mean) AS between_group_variance,\n"
                        "     avg(group_mean) AS grand_mean\n"
                        "RETURN accounts, total_ratings,\n"
                        "       round(grand_mean * 1000) / 1000.0 AS grand_mean,\n"
                        "       round(weighted_within / total_ratings * 10000) / 10000.0 AS within_group_variance,\n"
                        "       round(between_group_variance * 10000) / 10000.0 AS between_group_variance,\n"
                        "       round(100.0 * between_group_variance /\n"
                        "             (between_group_variance + weighted_within / total_ratings) * 10) / 10.0\n"
                        "         AS percent_explained_by_account"
                    ),
                ),
                use_cases=[
                    "Pooling dispersion across groups, which requires squared units.",
                    "Decomposing total variation into within-group and between-group parts.",
                    "Feeding a dispersion into further arithmetic without a round trip through a "
                    "square root.",
                ],
                limits=[
                    "Squared units. A variance of rating points is in rating points squared and "
                    "should not be shown to a reader as though it were a rating.",
                    "Undefined for a single row.",
                    "Squaring amplifies outliers even more than the standard deviation does.",
                ],
                see_also=[
                    "[`variancep`](./variancep.md) for the population form",
                    "[`stdev`](./stdev.md) for the readable form",
                ],
            ),
            _page(
                slug="variancep",
                title="`variancep`",
                family="functions/aggregation/dispersion",
                standard="extension",
                signature="variancep(expression)",
                summary="Population variance: squared spread when the rows are the whole population.",
                what=(
                    "`variancep` returns the population variance — the mean squared deviation from "
                    "the mean, dividing by the count. It is `stdevp` squared and, like `stdevp`, is "
                    "defined for a single row, where it is `0`."
                ),
                when=(
                    "Use it when the group is complete and the result feeds further arithmetic. It "
                    "is the correct form inside a variance decomposition, where each group's "
                    "variance describes that group entirely rather than sampling it."
                ),
                differs=(
                    "`variance` applies the sample correction. `stdevp` is the square root of this "
                    "and is what to display. As with the standard deviations, the choice is a claim "
                    "about what the rows represent."
                ),
                simple=Example(
                    note="Population variance per rating year, with the count that produced it.",
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "WINDOW TUMBLING duration('P365D') ON rating.at AS year\n"
                        "WITH year, rating\n"
                        "RETURN year.start AS window_start,\n"
                        "       count(rating) AS ratings,\n"
                        "       round(avg(rating.rating) * 1000) / 1000.0 AS mean,\n"
                        "       round(variancep(rating.rating) * 1000) / 1000.0 AS population_variance,\n"
                        "       round(stdevp(rating.rating) * 1000) / 1000.0 AS population_stdev\n"
                        "ORDER BY window_start"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Which raters are consistent and which are erratic. Each rater's complete "
                        "output is a population, so `variancep` describes it exactly; comparing "
                        "each rater's variance against the network's overall variance says whether "
                        "they discriminate more or less than the network does as a whole."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[all_ratings:RATED]->()\n"
                        "WITH variancep(all_ratings.rating) AS network_variance\n"
                        "MATCH (rater:Account)-[rating:RATED]->()\n"
                        "WITH network_variance, rater, count(rating) AS given,\n"
                        "     avg(rating.rating) AS mean,\n"
                        "     variancep(rating.rating) AS rater_variance\n"
                        "WHERE given >= 40\n"
                        "RETURN rater.account_id AS account, given AS ratings_given,\n"
                        "       round(mean * 100) / 100.0 AS mean_given,\n"
                        "       round(rater_variance * 1000) / 1000.0 AS rater_variance,\n"
                        "       round(network_variance * 1000) / 1000.0 AS network_variance,\n"
                        "       round(rater_variance / network_variance * 1000) / 1000.0 AS variance_ratio\n"
                        "ORDER BY variance_ratio, account\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Group variances inside a decomposition.",
                    "Comparing one group's dispersion against the whole graph's.",
                    "Complete populations where the sample correction would be wrong.",
                ],
                limits=[
                    "Squared units, and not a number to display directly.",
                    "Returns `0` for a single row.",
                ],
                see_also=["[`variance`](./variance.md) for the sample form"],
            ),
        ],
    )


def percentiles() -> Family:
    return Family(
        path="functions/aggregation/percentiles",
        title="Percentile aggregates",
        blurb=(
            "Position within a distribution rather than its centre. Both take a percentile between "
            "`0` and `1`; they differ in whether they are allowed to invent a value that is not in "
            "the data."
        ),
        pages=[
            _page(
                slug="percentilecont",
                title="`percentilecont`",
                family="functions/aggregation/percentiles",
                standard="extended",
                signature="percentilecont(expression, percentile)",
                summary=(
                    "The value at a percentile, interpolating between the two rows that surround it."
                ),
                what=(
                    "`percentilecont` orders the non-null numeric values and returns the value at "
                    "the requested position, interpolating linearly when the position falls between "
                    "two rows. `percentilecont(x, 0.5)` is the median; `0` and `1` give the minimum "
                    "and maximum.\n\n"
                    "Because it interpolates, the result need not be a value that appears in the "
                    "data — which is correct for a continuous measure and wrong for a discrete one."
                ),
                when=(
                    "Use it on continuous quantities: durations, distances, prices, scores. It is "
                    "the right tool for describing skewed data, where the mean is dragged away from "
                    "anything typical and a set of percentiles describes the shape honestly."
                ),
                differs=(
                    "`percentiledisc` returns an actual value from the data instead of "
                    "interpolating, which is what you want when the value is a category or a count. "
                    "`avg` gives the centre of mass; a percentile gives a position, and on skewed "
                    "data the two say very different things."
                ),
                simple=Example(
                    note=(
                        "The distribution of ratings as five positions. Reading them together shows "
                        "a shape the mean alone conceals."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "RETURN count(rating) AS ratings,\n"
                        "       percentilecont(rating.rating, 0.05) AS p05,\n"
                        "       percentilecont(rating.rating, 0.25) AS p25,\n"
                        "       percentilecont(rating.rating, 0.5) AS median,\n"
                        "       percentilecont(rating.rating, 0.75) AS p75,\n"
                        "       percentilecont(rating.rating, 0.95) AS p95,\n"
                        "       round(avg(rating.rating) * 1000) / 1000.0 AS mean"
                    ),
                ),
                advanced=Example(
                    note=(
                        "How the distribution moved over the life of the network. Each year gets "
                        "its own quartiles and interquartile range, which shows whether the "
                        "network's ratings became more generous, more polarised, or simply more "
                        "numerous."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "WINDOW TUMBLING duration('P365D') ON rating.at AS year\n"
                        "WITH year, rating\n"
                        "RETURN year.start AS window_start,\n"
                        "       count(rating) AS ratings,\n"
                        "       percentilecont(rating.rating, 0.25) AS q1,\n"
                        "       percentilecont(rating.rating, 0.5) AS median,\n"
                        "       percentilecont(rating.rating, 0.75) AS q3,\n"
                        "       percentilecont(rating.rating, 0.75)\n"
                        "         - percentilecont(rating.rating, 0.25) AS interquartile_range\n"
                        "ORDER BY window_start"
                    ),
                ),
                use_cases=[
                    "Describing skewed distributions honestly with a set of positions.",
                    "Interquartile range as a spread measure that ignores extremes.",
                    "Service-level style thresholds on continuous measures.",
                ],
                limits=[
                    "The percentile argument is a fraction between `0` and `1`, not a number out of "
                    "one hundred.",
                    "The interpolated result may not exist in the data. On a discrete measure use "
                    "`percentiledisc`.",
                    "Computing a percentile requires ordering the group, so it costs more than "
                    "`avg` over the same rows.",
                ],
                see_also=[
                    "[`percentiledisc`](./percentiledisc.md) for a value drawn from the data",
                    "[`avg`](../avg.md) for the centre of mass",
                ],
            ),
            _page(
                slug="percentiledisc",
                title="`percentiledisc`",
                family="functions/aggregation/percentiles",
                standard="extended",
                signature="percentiledisc(expression, percentile)",
                summary="The value at a percentile, always one that actually occurs in the data.",
                what=(
                    "`percentiledisc` orders the non-null numeric values and returns the first one "
                    "at or past the requested position. It never interpolates, so the result is "
                    "always a value present in the group."
                ),
                when=(
                    "Use it when a value between two observations would be meaningless: counts, "
                    "ordinal ratings, category codes, anything where 'two and a half' is not a "
                    "thing that can exist. On this dataset ratings are whole numbers from -10 to "
                    "+10, so the discrete form is the honest one."
                ),
                differs=(
                    "`percentilecont` interpolates and so can return a value that never occurred. "
                    "On a large group of continuous values the two agree closely; on a small group "
                    "of discrete values they differ visibly, and the discrete one is right."
                ),
                simple=Example(
                    note=(
                        "The same percentiles under both definitions. Where they disagree, "
                        "`percentilecont` has invented a rating nobody gave."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->()\n"
                        "RETURN percentiledisc(rating.rating, 0.25) AS q1_discrete,\n"
                        "       percentilecont(rating.rating, 0.25) AS q1_continuous,\n"
                        "       percentiledisc(rating.rating, 0.5) AS median_discrete,\n"
                        "       percentilecont(rating.rating, 0.5) AS median_continuous,\n"
                        "       percentiledisc(rating.rating, 0.9) AS p90_discrete,\n"
                        "       percentilecont(rating.rating, 0.9) AS p90_continuous"
                    ),
                ),
                advanced=Example(
                    note=(
                        "A per-account rating profile expressed only in ratings that were actually "
                        "given. Because every reported figure occurs in the data, the row can be "
                        "read as a description of real behaviour rather than of a fitted "
                        "distribution."
                    ),
                    query=(
                        "USE trust\n"
                        "MATCH ()-[rating:RATED]->(rated:Account)\n"
                        "WITH rated, count(rating) AS ratings,\n"
                        "     percentiledisc(rating.rating, 0.1) AS p10,\n"
                        "     percentiledisc(rating.rating, 0.5) AS median,\n"
                        "     percentiledisc(rating.rating, 0.9) AS p90\n"
                        "WHERE ratings >= 40\n"
                        "RETURN rated.account_id AS account, ratings, p10, median, p90,\n"
                        "       p90 - p10 AS span\n"
                        "ORDER BY span DESC, account\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Percentiles over counts, ordinal scales and category codes.",
                    "Reports where every figure must be a value that genuinely occurred.",
                    "Small groups, where interpolation would invent precision.",
                ],
                limits=[
                    "The percentile argument is a fraction between `0` and `1`.",
                    "On a small group the result jumps between observed values as the percentile "
                    "moves; it does not vary smoothly.",
                    "Requires ordering the group, like `percentilecont`.",
                ],
                see_also=["[`percentilecont`](./percentilecont.md) for the interpolating form"],
            ),
        ],
    )
