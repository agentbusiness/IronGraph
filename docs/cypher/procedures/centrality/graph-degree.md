# `graph.degree`

> Relationship counts per node, split by direction.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.degree() YIELD node, outDegree, inDegree, degree` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`epinions`](../../datasets.md#epinions) — Epinions directed trust network |

## What it does

`graph.degree` yields one row for every node in the graph with three counts: `outDegree`, the relationships leaving it; `inDegree`, the relationships arriving; and `degree`, their sum. Isolated nodes are yielded too, with zeros, so the row count is the node count.

## How it behaves

The split matters more than the total in a directed graph. In a trust network `outDegree` is how many people this account trusts and `inDegree` is how many trust it — two quite different things that the combined `degree` averages away. Reach for the total only when direction genuinely carries no meaning.

Every relationship is counted, including parallel relationships between the same pair and self-relationships. Degree is a count of relationships, not of distinct neighbours.

## When to use it

Use it as the first measurement on any unfamiliar graph. Degree is cheap, it needs no parameters, and its distribution tells you immediately whether the graph is broadly even or dominated by a few hubs — which decides whether the more expensive algorithms will tell you anything.

## How it differs from its neighbours

Degree is local: it sees only a node's own relationships. `graph.pagerank` is recursive and asks who those relationships come *from*. A node can have high degree and low PageRank when its many endorsements come from nowhere in particular, and the reverse when a handful come from the centre of the graph.

## Simple example

The ten most-trusted accounts in the Epinions network, by inbound trust.

```cypher
USE epinions
CALL graph.degree() YIELD node, inDegree, outDegree
RETURN node.user_id AS user, inDegree AS trusted_by, outDegree AS trusts
ORDER BY inDegree DESC, user
LIMIT 10
```

Result:

```
user | trusted_by | trusts
-----+------------+-------
0    | 636        | 139   
1    | 802        | 320   
2    | 237        | 54    
3    | 40         | 41    
4    | 125        | 76    
5    | 176        | 101   
6    | 232        | 27    
7    | 30         | 15    
8    | 104        | 29    
9    | 15         | 16    

10 rows, 95 ms
```

## Advanced example

The shape of the whole degree distribution, bucketed by order of magnitude. This is the measurement worth taking before any other algorithm: it shows how heavily the graph is concentrated in a few nodes.

```cypher
USE epinions
CALL graph.degree() YIELD node, degree
WITH CASE
       WHEN degree = 0 THEN 0
       ELSE toInteger(floor(log10(toFloat(degree))))
     END AS magnitude, degree
RETURN magnitude,
       count(*) AS accounts,
       min(degree) AS lowest,
       max(degree) AS highest,
       round(avg(degree) * 10) / 10.0 AS mean
ORDER BY magnitude
```

Result:

```
magnitude | accounts | lowest | highest | mean  
----------+----------+--------+---------+-------
0         | 62252    | 1      | 9       | 2.4   
1         | 11413    | 10     | 99      | 29.9  
2         | 2191     | 100    | 968     | 225.8 
3         | 23       | 1019   | 3079    | 1424.1

4 rows, 165 ms
```

## Where it earns its place

- The first look at an unfamiliar graph.
- Separating who acts from who is acted upon in a directed graph.
- Providing a baseline that a recursive score has to beat to be worth running.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- One row per node in the graph, including isolated ones. On a large graph, aggregate or filter inside the query.
- Parallel and self relationships are counted individually; degree is not a count of distinct neighbours.

## See also

- [`graph.pagerank`](./graph-pagerank.md) for recursive importance
