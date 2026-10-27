# `USE`

Select the project that a query reads or changes. Use this clause at the start of graph queries;
IronGraph has no implicit default project.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `USE <project>` |
| Relationship to standard Cypher | IronGraph extension |

## What it does

A project is an isolated graph with its own labels, relationship types, properties and indexes.
`USE` selects one existing project by name. Use backticks to quote a name containing spaces or
punctuation. A query cannot join data across projects.

## Simple example

Prerequisite: create `worked_example` with `CREATE PROJECT IF NOT EXISTS worked_example`.

```cypher
USE worked_example
MATCH (node)
RETURN count(node) AS nodes
```

Expected result: one row with the number of nodes in that project.

## Advanced example

```cypher
USE worked_example
SHOW INDEXES
```

Expected result: the selected project's index status rows. Administrative statements such as
`SHOW PROJECTS` and `CREATE PROJECT` operate on the project catalogue rather than an implicit graph.

## Scope and limitations

Query API and SDK callers may supply a project identity through their supported request options.
Choose a single project consistently; a mismatched identity and `USE` name is an error. Layer
selection is separate: use [`USE LAYER`](../layers/use-layer.md) for graph layers and
[`WRITE LAYER`](../layers/write-layer.md) for write placement.

Next, use [`MATCH`](./match.md) to read the selected graph.
