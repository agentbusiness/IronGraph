# Index statements

Four index kinds, one statement shape. An equality index answers exact lookups, a range index answers ordered comparisons, a text index answers word matching, and a vector index answers similarity. Declaring the right one is what turns a scan into a lookup; declaring the wrong one costs maintenance and buys nothing.

| Page | Summary | Standard |
| --- | --- | --- |
| [`CREATE INDEX`](./create-index.md) | Declares an access path over a label's property or properties. | extension |
| [`SHOW INDEXES`](./show-indexes.md) | Lists a project's indexes with their kind, state and diagnostic. | extension |
| [`REBUILD INDEX`](./rebuild-index.md) | Rebuilds an index from the current graph. | extension |
| [`DROP INDEX`](./drop-index.md) | Removes an index declaration and everything it maintained. | extension |
