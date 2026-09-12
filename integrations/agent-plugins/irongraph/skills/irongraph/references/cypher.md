# IronGraph Cypher essentials

- Every operation targets an explicit project. There is no implicit default project.
- `OBSERVED`, `KNOWLEDGE`, and `WORKSPACE` are first-class graph layers; preserve their meaning when reading or writing layered data.
- Use parameters for user-supplied values. Parameters are data, not identifiers or syntax.
- Use `MATCH`/`OPTIONAL MATCH` for patterns, `WHERE` for predicates, `WITH` for pipeline boundaries, and `RETURN` for result shape.
- Use `MERGE` only with a stable identity pattern. Apply mutable properties separately so changing data does not create duplicates.
- Vector search operates through declared graph indexes over node properties. The matched owner is a canonical node; embeddings and spans remain derived state.
- Use `SHOW` statements for projects, indexes, topics, queues, exchanges, and consumer lag rather than relying on hidden administration APIs.

When exact grammar matters, search the bundled reference with `irongraph_search_cypher_docs` and read the returned resource. The reference shipped by the running MCP server is authoritative for that IronGraph version.
