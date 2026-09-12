# `USE LAYER`

> Chooses which layers the query reads.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `USE LAYER <layer> [, <layer> …]  (before the query body)` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`USE LAYER` names the layers a query may read. Without it the query reads the default view, `OBSERVED` and `KNOWLEDGE` together. Naming layers replaces that view rather than adding to it, so `USE LAYER OBSERVED` reads observed facts and nothing else.

The three layers carry fixed meanings.

- `OBSERVED` — facts captured from source activity. What happened.
- `KNOWLEDGE` — curated understanding. What has been concluded.
- `WORKSPACE` — provisional or application working state. What is being tried.

`OBSERVED` and `KNOWLEDGE` together form the default read view. `WORKSPACE` is never in it unless a query asks, which is what keeps scratch data out of results that did not request it.

## How it behaves

The choice is total. A node in a layer the query did not select does not exist for that query: it cannot be matched, counted, traversed through, or reached by an algorithm. That is what makes a layer a scope rather than a filter — there is no way for unselected data to leak into a result.

Because the graph procedures read the project graph under the query's selected layers, `USE LAYER` is also how an algorithm is scoped.

## When to use it

Name layers when the distinction matters: reading only what was observed before a conclusion was drawn, keeping curated data out of a raw count, or including workspace data that the default view deliberately excludes.

## How it differs from its neighbours

`USE LAYER` chooses what is visible; `WRITE LAYER` chooses where new data lands. The two are linked: a write layer must be among the layers selected for reading, so a query that writes to the workspace has to select it.

## Simple example

The same count under three views. Every account in this dataset was written to `OBSERVED`, so the knowledge-only view is empty and the default view matches the observed one.

```cypher
USE trust
USE LAYER OBSERVED
MATCH (account:Account)
RETURN count(account) AS observed_accounts
```

Result:

```
observed_accounts
-----------------
5881             

1 row
```

## Advanced example

Workspace data is invisible to the default view. A draft is written to `WORKSPACE`, counted there, and then counted again without naming the layer — where it does not appear at all.

```cypher
USE trust
USE LAYER WORKSPACE
MATCH (draft:Draft)
RETURN count(draft) AS drafts_in_workspace,
       collect(draft.note) AS notes
```

Result:

```
drafts_in_workspace | notes                                        
--------------------+----------------------------------------------
1                   | [{"type":"string","value":"candidate merge"}]

1 row, 54 ms
```

## Where it earns its place

- Separating what was observed from what was concluded.
- Keeping provisional work out of ordinary results by default.
- Scoping a graph algorithm to one layer of the graph.

## Limitations and trade-offs

- Naming layers replaces the default view rather than extending it.
- Unselected data is invisible, not filtered: it cannot be traversed through either.
- One layer selection per query.

## See also

- [`WRITE LAYER`](./write-layer.md)
