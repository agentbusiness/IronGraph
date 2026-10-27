# `IMPORT DATASET`

Import a bundled example dataset into its named project. Use this statement to explore IronGraph
without preparing your own import files.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `IMPORT DATASET <name>` |
| Relationship to standard Cypher | IronGraph extension |

## What it does

Each bundled dataset has a fixed project name. The import creates that project and its ordinary
graph records through the normal write path. If the project already exists, the import is skipped;
it does not append duplicate examples or replace user data.

## Simple example

Prerequisite: a running database distribution containing its bundled example data.

```cypher
IMPORT DATASET fraud
```

Expected result: the compact fraud example is available in project `fraud`, or an existing `fraud`
project is preserved without another import.

## Advanced example

```cypher
USE fraud
MATCH (account:Account)-[transfer:TRANSFERRED_TO]->(destination:Account)
RETURN account.id, destination.id, transfer.amount
ORDER BY transfer.amount DESC
LIMIT 5
```

Expected result: up to five transfer relationships with their account identifiers and amounts.
An existing user-created project with the same name may contain different data.

## Available datasets and limitations

The [dataset reference](../datasets.md) lists `fraud`, `flights`, `trust`, `epinions`, `dblp`,
`citations`, `library`, `social`, `email` and `overflow`, including their model and attribution.
Large imports consume time and memory according to their graph and asynchronous embedding work.
Unknown names are rejected. This statement does not import an arbitrary URL, CSV file or directory.

Next, open Training or query the imported project with `USE <name>`.
