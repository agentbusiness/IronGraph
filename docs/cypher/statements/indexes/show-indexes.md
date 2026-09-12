# `SHOW INDEXES`

> Lists a project's indexes with their kind, state and diagnostic.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `SHOW INDEXES` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`citations`](../../datasets.md#citations) — arXiv hep-th citation network with abstracts |

## What it does

`SHOW INDEXES` returns one row per index in the current project: `name`, `kind`, `state`, and a `diagnostic` that is null unless something went wrong.

`state` is the column that matters. `ONLINE` means the index is being used. `FAILED` means it exists but is not usable, and the diagnostic says why.

## How it behaves

A failed index is not a silent degradation. A query that would have used it is rejected rather than answered more slowly from a broken structure — so `SHOW INDEXES` is the first thing to check when a retrieval query starts failing rather than starting to crawl.

The most common failure is a vector index that missed its recall floor: the diagnostic reports the measured figure against the required one.

Rollups are derived state but are not indexes, and do not appear here.

## When to use it

Use it to confirm a declaration landed, to check an index is usable before depending on it, and to read the diagnostic when one is not.

## How it differs from its neighbours

`SHOW CONSTRAINTS` lists schema authority rather than access paths; the two listings do not overlap even though a constraint is backed by an index.

## Simple example

The indexes on the citation dataset: one lookup path and two text indexes.

```cypher
USE citations
SHOW INDEXES
```

Result:

```
name                | kind     | state  | diagnostic
--------------------+----------+--------+-----------
paper_abstract_text | TEXT     | ONLINE | null      
paper_by_id         | EQUALITY | ONLINE | null      
paper_title_text    | TEXT     | ONLINE | null      

3 rows
```

## Advanced example

A failed index and its diagnostic, produced rather than described. Sixteen short texts on deliberately unrelated subjects are embedded — twice the corpus size at which the approximation still agrees with exact search — so the index is built, misses the 90% recall floor, and reports `FAILED` with the agreement it measured. The equality index declared beside it is unaffected.

```cypher
USE failed_index_example
SHOW INDEXES
```

Result:

```
name          | kind     | state  | diagnostic                                                                  
--------------+----------+--------+-----------------------------------------------------------------------------
note_by_id    | EQUALITY | ONLINE | null                                                                        
note_semantic | VECTOR   | FAILED | IndexUnavailable: IVF-PQ recall 8000bp is below the 9000bp publication floor

2 rows
```

## Where it earns its place

- Confirming a declaration took effect.
- Diagnosing a retrieval query that fails rather than slows.
- Auditing what a project maintains.

## Limitations and trade-offs

- Current project only.
- Cannot be composed with `YIELD`, `WITH` or `WHERE`; filter in the client.
- Rollups are not listed.

## See also

- [`CREATE INDEX`](./create-index.md)
