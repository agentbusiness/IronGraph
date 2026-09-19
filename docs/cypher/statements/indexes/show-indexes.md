# `SHOW INDEXES`

Inspect the indexes in an existing project and read their availability diagnostics. Use this
statement when confirming index creation or troubleshooting a query that cannot use an index.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `SHOW INDEXES` |
| Relationship to standard Cypher | IronGraph extension |

## Inspect a project

```cypher
USE knowledge
SHOW INDEXES
```

The result has one row per index with these columns:

| Column | Meaning |
| --- | --- |
| `name` | Index name used by queries and administration |
| `kind` | Index category, such as `VECTOR`, `TEXT`, or `EQUALITY` |
| `state` | `ONLINE` when usable, or another state requiring preparation or repair |
| `diagnostic` | Failure details when available; otherwise `null` |

When local embedding is enabled, the project automatically maintains `semantic_nodes` and
`semantic_relationships`. The combined `graph_semantic` search uses those indexes; it is a search
name rather than a third stored index. You may also see indexes explicitly declared by your
application.

## Investigate an unavailable index

Read the diagnostic and address the reported cause, such as unavailable embedding support or device
capacity. An index marked `FAILED` is unavailable until its cause is resolved.

Index availability does not certify the relevance of every result. Test representative queries
against your own data and inspect the returned entities and scores.

## Scope

This statement lists the selected project's indexes. It cannot be composed with `YIELD`, `WITH`, or
`WHERE`; inspect or filter its returned rows in your client. Temporal rollups do not appear here.

Next, run [`SEARCH`](../../clauses/search/search.md) to retrieve ranked entities, or
[`CREATE INDEX`](./create-index.md) to declare an application index.
