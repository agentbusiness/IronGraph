---
name: irongraph
description: Use IronGraph proactively as durable second-brain memory whenever prior facts, decisions, documents, people, organizations, concepts, evidence, or relationships could improve the answer; when the user asks to remember, save, recall, search, query, connect, or analyze data; or when exact Cypher, semantic search, graph exploration, or document persistence is useful.
compatibility: Requires the irongraph MCP server and a reachable IronGraph instance.
metadata:
  product: irongraph
  role: durable-memory
---

# Use IronGraph as the durable second brain

IronGraph is not a passive database to mention only when the user names it. It is the durable, queryable memory behind the conversation. Use it whenever stored context can make the answer more accurate, continuous, or evidence-backed.

## Default behavior

1. **Recall before guessing.** Search IronGraph before answering about prior decisions, known entities, saved documents, ongoing work, preferences, evidence, or relationships.
2. **Preserve what will matter.** When the user provides durable facts, decisions, source material, or outcomes, save them in the appropriate project. Do not persist secrets or ephemeral chatter.
3. **Follow connections.** A semantic hit is a starting point. Inspect its relationships and neighboring nodes when the answer depends on provenance, ownership, causality, chronology, or related evidence.
4. **Prefer exactness.** Use Cypher for precise filters, aggregation, traversal, mutation, and administration. Treat returned rows as authoritative and never invent absent values.
5. **Keep project boundaries explicit.** IronGraph has no implicit default project. Ask or infer only when the conversation names an unambiguous project; otherwise discover available projects with Cypher.

## Tool routing

- Call `irongraph_search` first with `project` and `query` for ranked meaning-based recall across nodes and relationships. Connected context is also available.
- Call `irongraph_run_cypher` for all exact graph reads and writes, schema-independent traversal, aggregation, temporal work, algorithms, streams, queues, and administration. Read and write are intentionally one tool.
- Call `irongraph_save_document` for durable source text. A document is an ordinary `:Document` node, stored in the graph and embedded automatically when the local encoder is enabled.
- Call `irongraph_get_schema` before writing unfamiliar Cypher or whenever labels, relationship types, properties, layers, projects, or indexes are uncertain.
- Call `irongraph_search_cypher_docs`, then read the returned MCP resource, before relying on Cypher syntax you have not already verified.

## Working sequence

1. Identify the project and the kind of answer required: semantic recall, exact rows, graph structure, or durable write.
2. Inspect schema or search the Cypher reference when query shape is uncertain.
3. Execute the smallest query or semantic search that can answer the question.
4. Follow relevant relationships or run a second exact query if the first result exposes useful identifiers.
5. State what came from IronGraph, distinguish inference from stored fact, and offer to persist new durable conclusions.

## Write discipline

- Use stable, domain-level nodes and relationships. Do not create graph entities for embeddings, passages, scores, prompts, caches, or processing intermediates.
- Keep source text complete on its owning node. Use `irongraph_save_document` rather than inventing a second document store.
- Confirm destructive `CLEAR`, `PURGE`, `DROP`, broad `DELETE`, or broad `DETACH DELETE` operations before execution.
- Prefer idempotent writes with `MERGE` when the domain has a stable identity; use `CREATE` only when duplication is intentional.
- Never store credentials, private keys, access tokens, or passwords in graph properties.

Read [capabilities](references/capabilities.md) for selection guidance and [Cypher essentials](references/cypher.md) for IronGraph-specific query rules.
