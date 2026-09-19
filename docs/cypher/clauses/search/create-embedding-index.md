# `CREATE EMBEDDING INDEX`

Create a field-specific semantic index when you want to search one text property on one node label.
Applications using automatic graph-wide semantic search do not need this statement.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `CREATE EMBEDDING INDEX <name> FOR (<var>:<Label>) FROM <var>.<source> INTO <var>.<target> USING MODEL default SIMILARITY COSINE\|DOT\|EUCLIDEAN` |
| Relationship to standard Cypher | IronGraph extension |

## Prerequisites

Select an existing project with `USE` and enable the local embedding model. The node label and source
text property must exist. `INTO` names the index's vector field; IronGraph declares that field when
needed. You do not need to write placeholder vectors to graph nodes.

## Create and search a field index

Run these statements in order in an existing project named `knowledge`:

```cypher
USE knowledge
CREATE (:Document {
  title: 'Device handbook',
  body: 'The handbook explains how to configure the local execution device.'
})
```

```cypher
USE knowledge
CREATE EMBEDDING INDEX document_body_semantic
FOR (document:Document)
FROM document.body INTO document.embedding
USING MODEL default SIMILARITY COSINE
```

```cypher
USE knowledge
MATCH (document:Document)
SEARCH document IN (EMBEDDING INDEX document_body_semantic
                    FOR TEXT 'configure the local device' LIMIT 10)
  SCORE AS score
RETURN document, score
ORDER BY score DESC
```

Expected result: matching `Document` nodes with numeric scores, ordered from highest score to lowest.
The exact scores depend on your text and active model. Inspect `SHOW INDEXES` to confirm that
`document_body_semantic` is `ONLINE`.

## Behavior

IronGraph embeds existing source text when the index is created and maintains the vectors as text
changes or nodes are deleted. The complete text stays on its original node. Generated vectors are
derived index data; reading `document.embedding` does not return the generated vector.

`MODEL` accepts `default`, the verified local model loaded by the database. The similarity must match
the active embedding profile. GPU-backed instances perform vector search on their selected device;
CPU instances use the CPU backend.

Model availability, valid input, and device capacity remain operating requirements. Failed index
operations report a diagnostic through `SHOW INDEXES`.

## Choose automatic or field-specific search

Use `graph_semantic` to search meaningful content across all nodes and relationships without a
declaration. Use a field-specific index when a particular property, such as `Document.body`, should
determine the ranking. A plain vector index is for vectors you supply yourself. A text index searches
the words present in the source.

Next, use [`SEARCH`](./search.md) to retrieve ranked entities or combine semantic retrieval with
ordinary graph filters.
