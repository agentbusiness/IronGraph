# `SEARCH`

Retrieve nodes and relationships by meaning, or apply semantic retrieval to an existing graph query.
This page is for applications that need ranked graph results from text or vectors.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `SEARCH <variable> IN (EMBEDDING INDEX <name> FOR TEXT\|VECTOR\|HYBRID TEXT <input> [USING <method>] LIMIT <n>) SCORE AS <alias> [EVIDENCE AS <path>]` |
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

### Choose a retrieval method

You can select a method explicitly. Existing queries without `USING` retain their normal index
selection. Use `EXACT` when you need full-dimensional scoring of every eligible owner:

```cypher
USE knowledge
SEARCH entity IN (EMBEDDING INDEX graph_semantic
                  FOR TEXT 'configure the local device' USING EXACT LIMIT 10)
  SCORE AS score
RETURN entity, score
```

| Method | Behavior |
| --- | --- |
| `EXACT` | Score every eligible owner using complete vectors. |
| `PREFIX` | Select candidates using a trained shorter prefix, then rank them using complete vectors. Requires text input and a compatible local encoder. |
| `QUANTIZED` | Select candidates using compressed binary comparisons, then rank them using complete vectors. |
| `ADAPTIVE` | Choose precise scoring for small eligible sets and compressed candidate selection for larger sets. |

Methods that select candidates can miss matches compared with `EXACT`. Compare relevance and
latency on your own data before choosing one. Complete final scores do not make an approximate
candidate search exact. `PREFIX` uses the existing embedding; it does not run a second model or
store a second full vector. These methods execute on the CPU.

### Combine terminology and meaning

`FOR HYBRID TEXT` combines lexical and semantic rankings. It is useful for queries containing exact
names, codes, or phrases alongside a natural-language description:

```cypher
USE knowledge
SEARCH entity IN (EMBEDDING INDEX graph_semantic
                  FOR HYBRID TEXT 'device error E104' USING EXACT LIMIT 10)
  SCORE AS score
RETURN entity, score
```

Hybrid scores describe rank fusion and are not directly comparable with ordinary vector scores.
The lexical branch reads meaningful owner content and can cost more than semantic search alone.
Filtering and graph layer visibility still apply. A declared text-field embedding index uses only
that selected field for lexical matching. A supplied-vector index has no declared text source and
does not support `FOR HYBRID TEXT`.

### Return relationship evidence

Search relationships directly and request their existing graph path:

```cypher
USE knowledge
SEARCH relationship IN (EMBEDDING INDEX semantic_relationships
                        FOR HYBRID TEXT 'responsible for the delayed shipment'
                        USING EXACT LIMIT 10)
  SCORE AS score EVIDENCE AS evidence
RETURN relationship, evidence, score
```

For a relationship, `evidence` contains its source node, relationship, and target node. For a node
result, it contains that node as a zero-length path. This exposes graph context; it does not prove
that the relationship is true or generate an explanation. Continue with ordinary Cypher to explore
the supporting records.

Long text remains complete on its owner. Retrieval can match a relevant part of a long value and
return the owning record without creating passage nodes.

### Interpret scores

`FOR TEXT` embeds the query using the same model as the stored text. `FOR VECTOR` accepts a supplied
vector of the required dimensions. Larger scores rank first; the interpretation depends on the
index's similarity. A score is not a confidence percentage or a guarantee of relevance.

Vector search runs on the CPU. Local text inference uses an independently selected device and
runs asynchronously. Host capacity and model availability
remain requirements.

Next, run [`SHOW INDEXES`](../../statements/indexes/show-indexes.md) to inspect availability if a search
reports an index or model error.
