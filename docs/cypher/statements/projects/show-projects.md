# `SHOW PROJECTS`

> Lists every project with its stable identity and display name.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `SHOW PROJECTS` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`SHOW PROJECTS` returns one row per project: `project_id`, the stable identity the database uses, and
`display_name`, the name `USE` matches. It runs without selecting a project because it describes the
project catalogue. Project creation and renaming also name their targets directly.

## How it behaves

The identity survives a rename and the display name does not, so anything that needs to refer to a project across time should hold the identity.

The statement stands alone. It cannot be followed by `YIELD`, `WITH` or `WHERE`, and cannot be combined with another statement in the same execution, so filtering and ordering the listing happens in the client.

## When to use it

Use it to discover what exists on a node, to confirm a create or a rename landed, and to resolve a display name to an identity.

## How it differs from its neighbours

`SHOW INDEXES` and `SHOW CONSTRAINTS` describe one project's schema and require a `USE`. `SHOW PROJECTS` describes the node.

## Simple example

Every project on this node, in name order.

```cypher
SHOW PROJECTS
```

Expected result: one row per current project, with `project_id` and `display_name`. Names and identities depend on your database.

## Advanced example

The identity is stable and the display name is not. This listing is taken after a rename: the project appears under its new name, carrying the identity it was created with.

```cypher
SHOW PROJECTS
```

Expected result: one row per current project, with `project_id` and `display_name`. Names and identities depend on your database.

## Where it earns its place

- Discovering what a node holds.
- Confirming a create or rename.
- Resolving a display name to a stable identity.

## Limitations and trade-offs

- Names and identities only; it reports nothing about size or contents.
- Every project is listed, with no filtering by permission at this layer.

## See also

- [`CREATE PROJECT`](./create-project.md)
