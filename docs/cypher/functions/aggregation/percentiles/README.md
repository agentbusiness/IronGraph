# Percentile aggregates

Position within a distribution rather than its centre. Both take a percentile between `0` and `1`; they differ in whether they are allowed to invent a value that is not in the data.

| Page | Summary | Standard |
| --- | --- | --- |
| [`percentilecont`](./percentilecont.md) | The value at a percentile, interpolating between the two rows that surround it. | extended |
| [`percentiledisc`](./percentiledisc.md) | The value at a percentile, always one that actually occurs in the data. | extended |
