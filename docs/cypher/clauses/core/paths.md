# Path selectors and modes

Choose which paths a graph pattern returns and which repetitions it allows. Use a finite hop bound
when expanding through a graph that may contain cycles.

| | |
| --- | --- |
| Kind | Pattern syntax |
| Signature | `MATCH [p =] [<selector>\|<mode>] (a)-[:TYPE*<min>..<max>]->(b)` |
| Relationship to standard Cypher | Standard Cypher, extended by IronGraph |

## What it does

Relationship patterns may be outgoing `-->`, incoming `<--`, or undirected `--`. A relationship
variable binds a single relationship; a variable-length relationship variable binds the traversed
relationship list. Assign `p =` to bind the whole path. A path preserves its node and relationship
order and supports `nodes(p)`, `relationships(p)` and `length(p)`.

## Select paths

| Prefix | Result |
| --- | --- |
| No prefix | All matching paths with distinct relationships within each path |
| `ANY` | A matching path per endpoint pair |
| `ANY SHORTEST` | One minimum-hop path per endpoint pair |
| `ALL SHORTEST` | All minimum-hop paths per endpoint pair |
| `SHORTEST <n>` | Up to the first `n` shortest paths per endpoint pair |
| `SHORTEST <n> GROUPS` | Paths in the first `n` distinct path-length groups |

`SHORTEST` defaults to a count of one. Counts must be positive integers. Prefixes occur after
an optional `p =`, before the first node pattern. These selectors count hops; for weighted costs use
[`graph.dijkstra`](../../procedures/shortest-path/graph-dijkstra.md).

## Control repetition

| Mode | Meaning |
| --- | --- |
| `DIFFERENT RELATIONSHIPS` | Default: a relationship cannot occur twice in one path |
| `REPEATABLE ELEMENTS` | Permit repeated relationships and nodes |
| `ACYCLIC` | A node cannot occur twice in one path |

A pattern takes one selector or mode prefix. Do not combine prefixes into an unsupported spelling.
Inside brackets, relationship quantifiers include `*` for one or more hops, `*2` for exactly two,
`*1..3` for one through three, and `*..3` for one through three. The postfix form follows the arrow:
`-[:TYPE]->+` means one or more, `-[:TYPE]->*` means zero or more, and `-[:TYPE]->{1,3}` supplies
explicit lower and upper bounds. Postfix forms also accept `{2}`, `{,3}` and `{2,}`. A maximum below
the minimum is an error. Explicit finite bounds make cycle behavior clear.

## Simple example

Prerequisite: an existing `worked_example` project. This query creates its own graph records.

```cypher
USE worked_example
CREATE (a:ReferenceRoute {name: 'A'})-[:REFERENCE_ROUTE]->(b:ReferenceRoute {name: 'B'})
WITH a, b
MATCH p = ANY SHORTEST (a)-[:REFERENCE_ROUTE*1..3]->(b)
RETURN length(p) AS hops
```

Result:

```
hops
----
1

1 row
```

Expected result: one route with one hop.

## Advanced example

```cypher
USE worked_example
CREATE (a:ReferenceWalk)-[:REFERENCE_WALK]->(b:ReferenceWalk), (b)-[:REFERENCE_WALK]->(a)
WITH a
MATCH p = REPEATABLE ELEMENTS (a)-[:REFERENCE_WALK*2..2]->(a)
RETURN length(p) AS hops
```

Result:

```
hops
----
2

1 row
```

Expected result: a two-hop return walk. With `ACYCLIC`, revisiting the starting node would be excluded.

## Limitations and trade-offs

The number of paths can grow much faster than the number of nodes. Narrow endpoint patterns,
relationship types and hop bounds before projecting path values. Selecting shortest paths does not
change graph data or create a stored routing index.

Next, compare [traversal procedures](../../procedures/traversal/README.md) when you need reachable
nodes rather than individual paths.
