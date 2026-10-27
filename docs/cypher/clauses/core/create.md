# `CREATE`

Create nodes and directed relationships. This page is for applications composing query pipelines.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `CREATE <pattern> [, <pattern> ...]` |
| Relationship to standard Cypher | Standard Cypher |
| Reference dataset | An existing example project |

## What it does

Create nodes and directed relationships.

## How it behaves

Use a bound variable to reuse an existing entity. Write patterns require directed relationships. Source property content stays on its owning entity; generated vectors are derived data.

## Prerequisites

Create an example project with `CREATE PROJECT IF NOT EXISTS worked_example`. The examples use that
project. Examples containing CREATE, MERGE, SET, REMOVE or DELETE change its graph; use a disposable
project for practice. No local embedding model is required.

## Simple example

```cypher
USE worked_example
CREATE (n:ReferenceCreated {name: 'Ada'}) RETURN n.name AS name
```

Result:

```
name
----
Ada

1 row
```

Expected result: the query returns the projected rows. Graph-dependent values depend on the project's current contents.

## Advanced example

```cypher
USE worked_example
CREATE (a:ReferenceCreated {name: 'A'})-[r:REFERENCE_CREATED {weight: 2}]->(b:ReferenceCreated {name: 'B'}) RETURN a.name, type(r), b.name
```

Result:

```
a.name | type(r)           | b.name
-------+-------------------+-------
A      | REFERENCE_CREATED | B

1 row
```

Expected result: the projected rows reflect the clause behavior described above.

## Limitations and trade-offs

Every graph query selects a project explicitly. Reads remain structurally valid during concurrent
writes, but do not promise snapshot isolation or atomic visibility across several changed records.
Use explicit ordering when result order matters.

## See also

- [Core clauses](./README.md)
- [Expressions and nulls](../../expressions.md)

