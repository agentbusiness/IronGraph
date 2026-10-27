# `MERGE`

Match a pattern or create it when it is absent. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `MERGE <pattern> [ON CREATE SET ...] [ON MATCH SET ...]` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Match a pattern or create it when it is absent.

## How it behaves

MERGE matches the whole pattern. Use a uniqueness constraint when the domain requires one node per property value. ON CREATE and ON MATCH apply their SET items to the selected branch.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
MERGE (n:ReferenceMerged {name: 'Ada'}) RETURN n.name AS name
```

Expected result: the query returns the projected rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
MERGE (n:ReferenceMerged {name: 'Ada'}) ON CREATE SET n.visits = 1 ON MATCH SET n.visits = coalesce(n.visits, 0) + 1 RETURN n.name, n.visits
```

Expected result: the projected rows reflect the clause behavior described above.

## Limitations and trade-offs

Every graph query selects a project explicitly. Reads remain structurally valid during concurrent
writes, but do not promise snapshot isolation or atomic visibility across several changed records.
Use explicit ordering when result order matters.

## See also

- [Core clauses](./README.md)
- [Expressions and nulls](../../expressions.md)

