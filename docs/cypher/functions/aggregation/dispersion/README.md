# Dispersion aggregates

Four aggregates that measure spread rather than position: how far the values in a group sit from their own mean. `stdev` and `stdevp` are widely implemented; `variance` and `variancep` are IronGraph additions that expose the squared form directly, so a query that needs to combine or weight dispersions does not have to square a standard deviation back up.

| Page | Summary | Standard |
| --- | --- | --- |
| [`stdev`](./stdev.md) | Sample standard deviation: typical distance from the mean, in the original units. | extended |
| [`stdevp`](./stdevp.md) | Population standard deviation: spread when the rows are everything, not a sample. | extended |
| [`variance`](./variance.md) | Sample variance: the squared spread, before the square root. | extension |
| [`variancep`](./variancep.md) | Population variance: squared spread when the rows are the whole population. | extension |
