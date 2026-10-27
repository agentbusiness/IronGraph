# `ALTER PROJECT`

Rename a project while preserving its stable identity and graph. Use this statement when changing
the display name that queries select with `USE`.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `ALTER PROJECT <name> RENAME TO <new-name>` |
| Relationship to standard Cypher | IronGraph extension |

## What it does

The project identity, graph records, indexes, embedding scope and broker resources stay with the
project. Only its display name changes. The new name must not already belong to another project.
Update application queries that select the old display name.

## Simple example

Prerequisite: create a disposable project with `CREATE PROJECT rename_example`.

```cypher
ALTER PROJECT rename_example RENAME TO renamed_example
```

Expected result: no graph rows. `SHOW PROJECTS` lists `renamed_example` with the original identity.

## Advanced example

```cypher
ALTER PROJECT renamed_example RENAME TO `Renamed example`
```

Expected result: the display name contains a space. Use backticks when selecting it with `USE`.

## Limitations and trade-offs

Rename is a catalogue operation, not a cross-project copy. Queries that already hold a supported
stable project identity need not rediscover it from the new name.

Next, run [`SHOW PROJECTS`](./show-projects.md) to inspect the final name and identity.
