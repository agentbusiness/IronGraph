# Layer clauses

Every project is partitioned into three layers, and every query chooses which it reads and which it writes. The choice is made once, ahead of the query body, and applies to the whole statement.

The three layers carry fixed meanings.

- `OBSERVED` — facts captured from source activity. What happened.
- `KNOWLEDGE` — curated understanding. What has been concluded.
- `WORKSPACE` — provisional or application working state. What is being tried.

`OBSERVED` and `KNOWLEDGE` together form the default read view. `WORKSPACE` is never in it unless a query asks, which is what keeps scratch data out of results that did not request it.

| Page | Summary | Standard |
| --- | --- | --- |
| [`USE LAYER`](./use-layer.md) | Chooses which layers the query reads. | extension |
| [`WRITE LAYER`](./write-layer.md) | Chooses which layer the query's writes land in. | extension |
