# `CREATE EMBEDDING INDEX`

> Encodes a text property into vectors with the local model and keeps them searchable.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `CREATE EMBEDDING INDEX <name> FOR (<var>:<Label>) FROM <var>.<source> INTO <var>.<target> USING MODEL default SIMILARITY COSINE\|DOT\|EUCLIDEAN` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`library`](../../datasets.md#library) — Eight arXiv papers with embedded abstracts |

## What it does

This statement declares that a text property on a label should be encoded into vectors by the database's own embedding model, and that those vectors should be maintained and searchable. `FROM` names the text, `INTO` names the vector property, and `SIMILARITY` fixes how distance is measured.

It is what makes `SEARCH … FOR TEXT` possible: a query supplies a phrase, the database encodes it with the same model, and the comparison is meaningful because both sides came from one encoder.

## How it behaves

Both properties must already exist in the project's schema when the statement runs. The source is the text you already have; the target has to be brought into existence first, which in practice means writing a placeholder vector to every row. The bundled `library` dataset uses `embedding: [0.0]`.

The vectors live in the index, not in the target property. That placeholder keeps whatever value it was given — reading it back shows the placeholder, not a 768-element vector — so the target property names the index's slot rather than storing its contents.

`MODEL` accepts only `default`: the model is the one verified artifact the database loads at start-up, and the similarity must match the profile that artifact was activated with.

The index validates itself before publishing. It runs a sample of queries through both the approximate index and exact search, and refuses to come online if agreement falls below 90%. On the real abstracts in this dataset that threshold is met at eight rows and not above it — measured agreement was 80% at 16 rows, 82% at 1,000 and 86% at 5,000. A vector index that misses the floor reports `FAILED` in `SHOW INDEXES` with the measured figure, and searching against it is rejected rather than silently answered from a worse index. There is no automatic fall back to exact search.

## When to use it

Declare one when retrieval should follow meaning rather than wording — finding the paper about horizon thermodynamics when the query says nothing about horizons. Where the words themselves are the query, a text index is the right tool and is far cheaper.

## How it differs from its neighbours

A `TEXT` index matches the words that are present. An embedding index matches what the text is about, and will rank a document that shares no vocabulary with the query above one that shares several words. A plain `VECTOR` index searches vectors you supply and computed yourself; an embedding index computes them for you and keeps them current.

## Simple example

The index in the reference dataset, as the bundled importer declares it. The statement returns no rows; `SHOW INDEXES` is how you see the result.

```cypher
USE library
SHOW INDEXES
```

Result:

```
name                | kind     | state  | diagnostic
--------------------+----------+--------+-----------
abstract_semantic   | VECTOR   | ONLINE | null      
library_paper_by_id | EQUALITY | ONLINE | null      
library_title_text  | TEXT     | ONLINE | null      

3 rows
```

## Advanced example

The whole cycle on a fresh project: text written, a placeholder vector written so the target property exists, the index declared, and the index state read back. The corpus is four documents, comfortably inside the recall floor.

```cypher
USE embedding_example
MATCH (note:Note)
SEARCH note IN (EMBEDDING INDEX note_semantic
                FOR TEXT 'storing facts that change over time' LIMIT 4)
  SCORE AS score
RETURN note.title AS title,
       round(score * 10000) / 10000.0 AS score,
       size(note.embedding) AS stored_property_size
ORDER BY score DESC
```

Result:

```
title              | score  | stored_property_size
-------------------+--------+---------------------
Bitemporal records | 0.2607 | 1                   
Bitemporal records | 0.2607 | 1                   
Bitemporal records | 0.2607 | 1                   
Bitemporal records | 0.2607 | 1                   
Graph storage      | 0.1982 | 1                   
Graph storage      | 0.1982 | 1                   
Graph storage      | 0.1982 | 1                   
Graph storage      | 0.1982 | 1                   
Similarity search  | 0.0577 | 1                   
Similarity search  | 0.0577 | 1                   
Similarity search  | 0.0577 | 1                   
Similarity search  | 0.0577 | 1                   
... 4 more rows

16 rows
```

## Where it earns its place

- Retrieval by meaning rather than by shared vocabulary.
- Ranking documents against a phrase a user typed.
- Combining a semantic ranking with ordinary graph filters in one query.

## Limitations and trade-offs

- Both the source and target properties must exist before the statement runs.
- The target property stores its placeholder, not the vectors; the index holds those.
- Only `MODEL default` is accepted, and the similarity must match the active profile.
- The index refuses to publish below 90% measured recall, which on real embeddings bounds the practical corpus size severely.

## See also

- [`SEARCH`](./search.md) to query the index
- [`vector.cosine`](../../functions/vector/vector-cosine.md) for the arithmetic it is built on
