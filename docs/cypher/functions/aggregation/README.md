# Aggregate functions

Twelve aggregates. `count`, `sum`, `avg`, `min`, `max` and `collect` are standard Cypher; the variance, standard deviation and percentile aggregates go beyond it. All twelve group the same way and skip nulls the same way, so the choice between them is purely about what you want measured.

Every example on these pages runs against the `trust` dataset.

| Page | Summary | Standard |
| --- | --- | --- |
| [`count`](./count.md) | How many rows reached this point, or how many carried a value. | standard |
| [`sum`](./sum.md) | The total of the numeric values in the rows that reached this point. | standard |
| [`avg`](./avg.md) | The arithmetic mean of the numeric values that reached this point. | standard |
| [`min`](./min.md) | The smallest value among the rows that reached this point. | standard |
| [`max`](./max.md) | The largest value among the rows that reached this point. | standard |
| [`collect`](./collect.md) | Gathers the values that reached this point into one list. | standard |
