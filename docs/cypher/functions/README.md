# Functions

Aggregates and scalar functions. The aggregates are documented first because they are where a graph query becomes a measurement, and because the dispersion and percentile aggregates go beyond what standard Cypher offers.

## [Aggregate functions](./aggregation/README.md)

Twelve aggregates. `count`, `sum`, `avg`, `min`, `max` and `collect` are standard Cypher; the variance, standard deviation and percentile aggregates go beyond it. All twelve group the same way and skip nulls the same way, so the choice between them is purely about what you want measured.

Every example on these pages runs against the `trust` dataset.

| Page | Summary |
| --- | --- |
| [`count`](./aggregation/count.md) | How many rows reached this point, or how many carried a value. |
| [`sum`](./aggregation/sum.md) | The total of the numeric values in the rows that reached this point. |
| [`avg`](./aggregation/avg.md) | The arithmetic mean of the numeric values that reached this point. |
| [`min`](./aggregation/min.md) | The smallest value among the rows that reached this point. |
| [`max`](./aggregation/max.md) | The largest value among the rows that reached this point. |
| [`collect`](./aggregation/collect.md) | Gathers the values that reached this point into one list. |

## [Dispersion aggregates](./aggregation/dispersion/README.md)

Four aggregates that measure spread rather than position: how far the values in a group sit from their own mean. `stdev` and `stdevp` are widely implemented; `variance` and `variancep` are IronGraph additions that expose the squared form directly, so a query that needs to combine or weight dispersions does not have to square a standard deviation back up.

| Page | Summary |
| --- | --- |
| [`stdev`](./aggregation/dispersion/stdev.md) | Sample standard deviation: typical distance from the mean, in the original units. |
| [`stdevp`](./aggregation/dispersion/stdevp.md) | Population standard deviation: spread when the rows are everything, not a sample. |
| [`variance`](./aggregation/dispersion/variance.md) | Sample variance: the squared spread, before the square root. |
| [`variancep`](./aggregation/dispersion/variancep.md) | Population variance: squared spread when the rows are the whole population. |

## [Percentile aggregates](./aggregation/percentiles/README.md)

Position within a distribution rather than its centre. Both take a percentile between `0` and `1`; they differ in whether they are allowed to invent a value that is not in the data.

| Page | Summary |
| --- | --- |
| [`percentilecont`](./aggregation/percentiles/percentilecont.md) | The value at a percentile, interpolating between the two rows that surround it. |
| [`percentiledisc`](./aggregation/percentiles/percentiledisc.md) | The value at a percentile, always one that actually occurs in the data. |

## [Vector functions](./vector/README.md)

Four functions over list-valued numbers. They compute similarity and distance directly in a query, without an index — useful for comparing a few vectors, and the wrong tool for searching a corpus.

| Page | Summary |
| --- | --- |
| [`vector.cosine`](./vector/vector-cosine.md) | Cosine similarity: how aligned two vectors are, ignoring their lengths. |
| [`vector.dot`](./vector/vector-dot.md) | Dot product: alignment scaled by both vectors' magnitudes. |
| [`vector.distance`](./vector/vector-distance.md) | Euclidean distance: how far apart two vectors are in space. |
| [`vector.normalize`](./vector/vector-normalize.md) | Scales a vector to length one, keeping its direction. |
