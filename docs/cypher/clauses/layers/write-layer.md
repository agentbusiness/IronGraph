# `WRITE LAYER`

> Chooses which layer the query's writes land in.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `WRITE LAYER <layer>  (before the query body)` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`WRITE LAYER` names the layer new nodes and relationships are created in. Without it, writes go to `OBSERVED`.

It is not independent of `USE LAYER`: a query may only write to a layer it also reads. Writing to `WORKSPACE` therefore means selecting it, either alone or alongside the layers the query reads from — `USE LAYER OBSERVED, WORKSPACE WRITE LAYER WORKSPACE` is the shape for deriving provisional data from authoritative data without mixing the two. A write layer outside the read set is rejected with `write layer is not visible`.

## How it behaves

The layer is a property of where an entity lives, fixed when it is created. Writing to `WORKSPACE` then reading without naming that layer will not find what was just written — not a fault, but the most common surprise. Read back with `USE LAYER WORKSPACE`.

A query that only reads may still name a write layer; it simply has no effect, and the visibility rule is not enforced against it.

## When to use it

Name a write layer whenever new data should not join the authoritative view: a candidate merge, a suggested link, an application's own working state. It is what lets a pipeline stage its output where a later stage can find it and an ordinary reader cannot.

## How it differs from its neighbours

`USE LAYER` controls visibility and `WRITE LAYER` controls placement, but the second is constrained by the first. A derivation that reads authoritative data and writes provisional data names both: the layers it reads from, and the workspace among them.

## Simple example

Writing to the workspace while reading the default view. The statement returns no rows; its effect is the write.

```cypher
USE trust
USE LAYER OBSERVED, WORKSPACE
WRITE LAYER WORKSPACE
MATCH (account:Account)
WHERE account.account_id = 35
CREATE (:Candidate {account_id: account.account_id, reason: 'high volume'})
```

Result:

```
No rows returned. 1 change committed.
```

## Advanced example

A derivation staged in the workspace: the busiest raters, read from the authoritative view and written where they will not disturb it. The result reads them back from the layer they landed in.

```cypher
USE trust
USE LAYER WORKSPACE
MATCH (candidate:Candidate)
RETURN candidate.account_id AS account,
       candidate.ratings_given AS ratings_given
ORDER BY ratings_given DESC, account
```

Result:

```
account | ratings_given
--------+--------------
35      | 763          
2642    | 406          
1810    | 404          
2125    | 397          
2028    | 293          

5 rows
```

## Where it earns its place

- Staging a derivation where ordinary readers will not see it.
- Recording curated conclusions in `KNOWLEDGE` beside the observations they came from.
- Giving an application its own working state inside the same graph.

## Limitations and trade-offs

- Data written to a layer is invisible until a query selects that layer.
- An entity's layer is fixed at creation.
- One write layer per query.

## See also

- [`USE LAYER`](./use-layer.md)
