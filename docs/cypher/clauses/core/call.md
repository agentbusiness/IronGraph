# `CALL / YIELD`

Run a procedure and bind its declared output columns. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `CALL <procedure>(<arguments>) [YIELD <field> [AS <name>] [, ...]]` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Run a procedure and bind its declared output columns.

## How it behaves

A procedure runs once per incoming row. Arguments and YIELD fields must match its signature.
Built-in inputs use explicit parentheses and arguments; a standalone zero-input call may omit them.
For execution-scoped custom procedures registered by an embedding host, a standalone call without
parentheses obtains its declared inputs from same-named query parameters. Such procedures are
available only in the execution where the host registers them.

A standalone call with omitted YIELD exposes its declared output fields. In a query pipeline, name
the fields needed by later clauses with YIELD and optional AS aliases. YIELD * is permitted only for
a standalone call. Unknown procedures, wrong argument counts and unknown output fields are errors.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
CALL graph.degree() YIELD node, degree RETURN degree ORDER BY degree DESC LIMIT 3
```

Expected result: the query returns the projected rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
CREATE (a:ReferenceBfs)-[:REFERENCE_BFS]->(b:ReferenceBfs) WITH a CALL graph.bfs(a) YIELD node, distance RETURN distance ORDER BY distance
```

Expected result: the projected rows reflect the clause behavior described above.

## Limitations and trade-offs

Every graph query selects a project explicitly. Reads remain structurally valid during concurrent
writes, but do not promise snapshot isolation or atomic visibility across several changed records.
Use explicit ordering when result order matters.

## See also

- [Core clauses](./README.md)
- [Expressions and nulls](../../expressions.md)

