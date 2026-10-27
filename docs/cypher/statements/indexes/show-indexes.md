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
| `scope` | On `semantic_relationships`, `relationships: all`, `relationships: none`, or `relationships: all except <types>`; otherwise `null` |
| `reconciliation` | On `semantic_relationships`, `pending` while scope cleanup or catch-up is unfinished, otherwise `complete`; other rows contain `null` |

When local embedding is enabled, the project automatically maintains `semantic_nodes` and
`semantic_relationships`. The combined `graph_semantic` search uses those indexes; it is a search
name rather than a third stored index. You may also see indexes explicitly declared by your
application.

Use [`ALTER AUTOMATIC SEMANTIC`](./alter-automatic-semantic.md) to choose relationship participation.
`NONE` intentionally leaves an empty `semantic_relationships` index defined. Narrowing excludes
relationships from search when the command completes; reconciliation removes their existing derived
vectors asynchronously. Widening schedules missing vectors. A `pending` value can remain while the
encoder is unavailable; it is not a percentage or an estimate of remaining time.
`complete` describes the scope reconciliation pass, not the backlog of unrelated embedding work
from normal graph writes.

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
