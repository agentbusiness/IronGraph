# `CREATE PROJECT`

> Creates a named, isolated graph.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `CREATE PROJECT [IF NOT EXISTS] <name>` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`CREATE PROJECT` creates an empty graph under a name. The project is the isolation boundary: labels, relationship types, properties, indexes, constraints, temporal declarations and layers all belong to one project and are invisible from any other.

`IF NOT EXISTS` makes the statement idempotent, which is what a start-up path or a migration wants.

## How it behaves

A project has a stable identity and a display name. `SHOW PROJECTS` returns both; the identity is what the database uses and the display name is what `USE` matches. Renaming changes the display name and leaves the identity alone, so a rename does not orphan anything.

Creating a project is a schema operation, not a data one. It commits through the same durability path as a write and is visible to the next statement.

## When to use it

Create a project per graph that should not see another: one per tenant, per environment, per dataset. Reference datasets in this documentation are one project each, which is why an example that says `USE trust` cannot accidentally read `epinions`.

## How it differs from its neighbours

It is not a label or a namespace inside one graph. Two projects share no storage, no schema and no index, and no single query can read across them — which is stronger isolation than a label prefix and cheaper than a separate process.

## Simple example

Creating a project idempotently, the way a start-up path should.

```cypher
CREATE PROJECT IF NOT EXISTS worked_example
```

Result:

```
No rows returned. 1 change committed.
```

## Advanced example

Isolation demonstrated rather than asserted. The same label and the same property name exist in two projects with different contents, and a query in one sees only its own.

```cypher
USE worked_example
MATCH (node:Shared)
RETURN count(node) AS visible_here,
       collect(node.origin) AS origins
```

Result:

```
visible_here | origins                                    
-------------+--------------------------------------------
1            | [{"type":"string","value":"first project"}]

1 row
```

## Where it earns its place

- One graph per tenant, environment or dataset.
- Keeping an experiment from touching production data.
- Making the graph a query runs against explicit in the statement itself.

## Limitations and trade-offs

- No query reads across projects. Combining two means reading both and joining in the client.
- There is no implicit default project; a graph query without `USE` has nothing to run against.
- Every project admitted to an accelerator holds its own resident copy, so project count is a capacity decision.

## See also

- [`SHOW PROJECTS`](./show-projects.md)
- [`DROP PROJECT`](./drop-project.md)
