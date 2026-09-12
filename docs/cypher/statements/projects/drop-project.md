# `DROP PROJECT`

> Removes a project and, with `CASCADE`, everything inside it.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `DROP PROJECT [IF EXISTS] <name> [CASCADE]` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`DROP PROJECT` removes a project. `CASCADE` removes its contents with it — nodes, relationships, indexes, constraints, temporal declarations and rollups. `IF EXISTS` makes the statement idempotent.

This is the most destructive statement in the language. There is no undo and no recycle bin.

## How it behaves

Without `CASCADE` a project that still holds data is not dropped, which is the guard against removing a populated graph by mistake. With `CASCADE` the guard is gone by request.

The bundled dataset importer never runs this statement implicitly. It checks the project catalog and skips an existing dataset project; replacing sample data requires an explicit destructive choice.

## When to use it

Use it to reclaim a tenant, retire an environment, or rebuild a dataset from scratch. In anything scripted, pair `IF EXISTS` with `CASCADE` so a re-run behaves the same as a first run.

## How it differs from its neighbours

`DELETE` removes matched graph entities and leaves the project, its schema and its indexes standing. `DROP PROJECT` removes the container.

## Simple example

Dropping a scratch project idempotently. Nothing is returned; the effect is the removal.

```cypher
DROP PROJECT IF EXISTS worked_example_other CASCADE
```

Result:

```
No rows returned. 1 change committed.
```

## Advanced example

The explicit rebuild pattern for sample data. Dropping and recreating before importing makes the result depend on the bundled data alone, not on whatever the project happened to contain before.

```cypher
USE worked_example
MATCH (row:Row)
RETURN count(row) AS rows_after_rebuild, sum(row.value) AS total
```

Result:

```
rows_after_rebuild | total
-------------------+------
2                  | 3    

1 row
```

## Where it earns its place

- Reclaiming a tenant or environment.
- Making a data load deterministic by rebuilding rather than appending.
- Removing an experiment completely.

## Limitations and trade-offs

- Irreversible. There is no undo.
- Without `CASCADE`, a project holding data is not dropped.
- Dropping a project invalidates anything holding its identity.

## See also

- [`CREATE PROJECT`](./create-project.md)
