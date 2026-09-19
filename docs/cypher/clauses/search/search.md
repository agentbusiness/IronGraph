# `SEARCH`

Retrieve nodes and relationships by meaning, or apply semantic retrieval to an existing graph query.
This page is for applications that need ranked graph results from text or vectors.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `SEARCH <variable> IN (EMBEDDING INDEX <name> FOR TEXT\|VECTOR <input> LIMIT <n>) SCORE AS <alias>` |
| Relationship to standard Cypher | IronGraph extension |

## Search the whole project

Prerequisites: an existing project with graph data and the local embedding model enabled. Automatic
semantic indexes are prepared on the first project operation and include existing data. Subsequent
writes maintain them. You do not need to declare an index or write vector properties.

```cypher
USE knowledge
SEARCH entity IN (EMBEDDING INDEX graph_semantic
                  FOR TEXT 'who is responsible for the launch plan' LIMIT 10)
  SCORE AS score
RETURN entity, score
```

Expected result: up to ten matching nodes and relationships, ranked by descending similarity score.
The result preserves the entity kind, so a relationship remains a relationship. Ties have a stable
order for the same graph state. An empty project returns no matches.

Use `semantic_nodes` or `semantic_relationships` instead of `graph_semantic` to search one entity
kind. Query layer selection still applies; the default includes `OBSERVED` and `KNOWLEDGE`.

## What automatic embedding includes

Nodes contribute labels and meaningful properties: names, titles, subjects, descriptions, document
bodies, email content, table values, and task or calendar details. String lists and nested table or
map content contribute their meaningful values. Domain dates, times, durations, numbers, and booleans
retain their field names so the value has context.

Relationships contribute their type, meaningful properties, and the names or titles of their
endpoints. A renamed endpoint updates the relationship's searchable description. A deleted entity
is removed from semantic search. Documents, people, emails, tables, plans, and tasks are ordinary
graph records, so their content follows the same rules.

Automatic field selection excludes recognized operational metadata, identifiers, source URLs,
creation and synchronization timestamps, credentials, and existing vectors. It does not infer the
meaning of every application-specific field name. Keep credentials outside graph properties. Use
descriptive content field names; use a declared field index when you need explicit field selection.

Source text remains complete on its owner. Relationships use bounded identifying excerpts of their
endpoints rather than duplicating entire document bodies.

## Search a declared field index

Use a preceding `MATCH` with a field index created by
[`CREATE EMBEDDING INDEX`](./create-embedding-index.md):

```cypher
USE knowledge
MATCH (document:Document)
SEARCH document IN (EMBEDDING INDEX document_body_semantic
                    FOR TEXT 'configure the local device' LIMIT 10)
  SCORE AS score
RETURN document.title AS title, score
ORDER BY score DESC
```

This form filters already-bound rows against the selected index's candidates. You can combine it
with graph patterns and ordinary predicates. The inner `LIMIT` bounds the candidates; later filters
can leave fewer rows, while later traversals can produce multiple rows per match. Add a final query
`LIMIT` when you also need to bound expanded results.

## Scores and execution

`FOR TEXT` embeds the query using the same model as the stored text. `FOR VECTOR` accepts a supplied
vector of the required dimensions. Larger scores rank first; the interpretation depends on the
index's similarity. A score is not a confidence percentage or a guarantee of relevance.

Vector search runs on the selected execution device. GPU-backed instances keep admitted indexes
resident on that device. CPU is an explicit execution backend. Device capacity and model availability
remain requirements.

Next, run [`SHOW INDEXES`](../../statements/indexes/show-indexes.md) to inspect availability if a search
reports an index or model error.
