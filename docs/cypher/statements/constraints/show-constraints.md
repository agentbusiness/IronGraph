# `SHOW CONSTRAINTS`

Inspect uniqueness requirements in one project. Use this statement to confirm a constraint or
understand a duplicate-value write error.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `SHOW CONSTRAINTS` |
| Relationship to standard Cypher | IronGraph extension |

## What it does

Returns one row per uniqueness constraint, with `name`, `label` and `property`. Each requirement
applies to nodes with that label in the selected project. It is distinct from an ordinary index,
which improves access without necessarily rejecting duplicates.

## Simple example

Prerequisite: an existing project named `worked_example`.

```cypher
USE worked_example
SHOW CONSTRAINTS
```

Expected result: the project's current constraints, or no rows when none are declared.

## Advanced example

After creating a constraint with the [creation statement](./create-constraint.md), inspect the same
project again:

```cypher
USE worked_example
SHOW CONSTRAINTS
```

Expected result: a row naming each declared constraint and its label and property.

## Scope and limitations

This is a standalone statement. It does not accept YIELD, WHERE or WITH; filter its returned rows in
your client. It does not list temporal declarations or replace index diagnostics.

Next, use [`SHOW INDEXES`](../indexes/show-indexes.md) to inspect access paths separately.
