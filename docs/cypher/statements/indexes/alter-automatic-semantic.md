# `ALTER AUTOMATIC SEMANTIC`

Choose which relationship types participate in a project's automatic semantic search. Use this
statement when structural relationships should remain in the graph without generating embeddings.
Node search and ordinary graph queries continue to work.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `ALTER AUTOMATIC SEMANTIC SET RELATIONSHIPS ALL\|NONE\|ALL EXCEPT <type>[, <type> ...]` |
| Relationship to standard Cypher | IronGraph extension |

## Prerequisites

Select an existing project with `USE`. The setting belongs to that project in both embedded and
standalone deployments. You can configure an empty project before its first write or before the
local encoder is enabled. Text search requires the local encoder to be enabled and ready.

## Simple example

Keep automatic node embeddings and omit all automatic relationship embeddings:

```cypher
USE code
ALTER AUTOMATIC SEMANTIC SET RELATIONSHIPS NONE
```

Expected result: the setting changes without returning graph rows. New searches through
`semantic_relationships` return no relationships. `graph_semantic` searches nodes only.
The `semantic_relationships` index remains defined.

## Advanced example

Exclude structural relationship types while retaining semantic search for other relationships:

```cypher
USE code
ALTER AUTOMATIC SEMANTIC SET RELATIONSHIPS ALL EXCEPT CALLS, CONTAINS, DEFINES, IMPORTS
```

Expected result: those four types are excluded from automatic semantic search. Other types remain
eligible. Names are case-sensitive and may refer to types not yet present. Use ordinary Cypher
backtick quoting for a name containing spaces or punctuation, for example `` `GENERATED FROM` ``.

Restore the default scope:

```cypher
USE code
ALTER AUTOMATIC SEMANTIC SET RELATIONSHIPS ALL
```

Expected result: all relationship types become eligible and missing embeddings are scheduled.
Search can return newly included relationships as their embeddings become available.

## Behavior

`ALL` is the default. `NONE` excludes every relationship. `ALL EXCEPT` replaces the exclusion set;
it does not append to a previous set. Repeating the same setting leaves it unchanged. There is no
`ONLY` form.

When narrowing completes, subsequent searches exclude disallowed relationships immediately.
Removal of existing derived vectors proceeds asynchronously. Widening schedules asynchronous
catch-up. Scope and recoverable work survive close and reopen through normal durability; use your
supported flush operation when you need an explicit durability boundary.

Excluded relationships retain their properties and remain usable in `MATCH`, traversals and graph
algorithms. Their creation and updates do not generate automatic embeddings. Endpoint changes do
not schedule excluded relationships for semantic refresh. The scope does not exclude node labels,
disable the local encoder, or change field-specific node embedding indexes.

## Storage and operating limits

Removing automatic vectors releases their payloads for reclamation. It does not promise an immediate
decrease in process physical footprint or database-file size. The shared text-embedding cache is
reusable across projects and is not purged by this command.

Inspect [`SHOW INDEXES`](./show-indexes.md) for the configured relationship scope and catch-up
status. An intentionally empty relationship index is different from an unfinished build.
When the encoder is disabled, the stored scope still applies and reconciliation may remain pending
until the encoder is enabled. Completion of a scope reconciliation pass does not certify that
unrelated embedding work from later graph writes has drained.

Next, run [`SEARCH`](../../clauses/search/search.md) against `semantic_nodes`,
`semantic_relationships`, or `graph_semantic` according to the entities your application needs.
