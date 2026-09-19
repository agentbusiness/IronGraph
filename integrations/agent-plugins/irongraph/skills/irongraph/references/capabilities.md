# Capability map

## Semantic recall

`irongraph_search` needs `project` and `query`. It defaults to `graph_semantic`, which ranks meaningful content from nodes and relationships together, including documents, emails, people, tables, calendar entries, and tasks. Use `semantic_nodes` or `semantic_relationships` to restrict the entity kind. For a declared text-field index, provide both `index` and its node `label`. Keep `include_connections` enabled unless the task needs isolated matches: separate bounded context results explain connections without changing the search ranking.

## Exact graph work

`irongraph_run_cypher` is the universal path for reads and writes. Use it for property predicates, multi-hop patterns, counts, grouping, ordering, graph algorithms, time-aware queries, project/layer inspection, and administration. Do not divide work into artificial read and write modes.

## Documents

Documents are ordinary graph nodes, normally `(:Document {title, body, ...})`. `irongraph_save_document` provides a safe, focused upsert for source text. Link documents to the entities, decisions, events, and evidence they support by running Cypher after saving them.

## Schema and language help

`irongraph_get_schema` returns the live database shape. `irongraph_search_cypher_docs` searches the bundled IronGraph language reference and returns resource URIs. Read the best resource before composing unfamiliar syntax.
