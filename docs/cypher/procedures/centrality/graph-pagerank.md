# `graph.pagerank`

> Recursive importance: a score that weighs who points at you, not how many.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.pagerank([damping, tolerance, maxIterations]) YIELD node, score` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`epinions`](../../datasets.md#epinions) — Epinions directed trust network |

## What it does

`graph.pagerank` yields one row per node with a score expressing how much of the graph's attention settles on it. Importance is recursive: a relationship from a node that is itself important contributes more than one from a node that is not. Scores across the graph sum to one, so a score is a share of the whole rather than an absolute quantity.

Called with no arguments the algorithm uses its default damping, tolerance and iteration ceiling. All three may be supplied together: damping and tolerance as numbers, the iteration ceiling as an integer. Supplying some but not all is rejected.

## How it behaves

Damping is the probability that the walk follows a relationship rather than restarting somewhere at random. Lower damping concentrates score near well-connected regions and converges faster; higher damping lets influence travel further from its source. Tolerance and the iteration ceiling bound the work: iteration stops when scores stop moving by more than the tolerance, or when the ceiling is reached, whichever comes first.

Because scores are a share of one, they shrink as the graph grows. Never compare a raw score between two graphs of different sizes, and never read a score as a probability of anything in the domain. Ranks compare; scores do not.

## When to use it

Use PageRank when you want influence rather than volume, and when the relationships in your graph genuinely mean endorsement — a citation, a trust declaration, a link, a recommendation. On a graph whose relationships mean "happened near" or "belongs to", the recursion has no meaning to propagate.

## How it differs from its neighbours

`graph.degree` counts relationships and stops there. PageRank asks where they came from, which is why the two rankings differ and why running both is more informative than running either. For grouping rather than ranking, the community procedures answer a different question entirely.

## Simple example

The ten most influential accounts in the Epinions trust network.

```cypher
USE epinions
CALL graph.pagerank() YIELD node, score
RETURN node.user_id AS user, round(score * 1000000) / 1000000.0 AS score
ORDER BY score DESC, user
LIMIT 10
```

Result:

```
user | score   
-----+---------
18   | 0.004535
737  | 0.00315 
118  | 0.002122
1719 | 0.002078
136  | 0.001987
790  | 0.001969
143  | 0.001957
40   | 0.001825
1619 | 0.001536
725  | 0.001496

10 rows, 223 ms
```

## Advanced example

Where influence and volume disagree. This ranks accounts by PageRank and by inbound degree in the same query and reports the accounts whose influence is least explained by how many people trust them — endorsement arriving from the centre of the network rather than in bulk.

```cypher
USE epinions
CALL graph.pagerank() YIELD node, score
WITH node, score ORDER BY score DESC LIMIT 100
MATCH (node)<-[trust:TRUSTS]-()
WITH node, score, count(trust) AS trusted_by
RETURN node.user_id AS user,
       round(score * 1000000) / 1000000.0 AS score,
       trusted_by,
       round(score * 100000000 / trusted_by) / 100.0 AS score_per_endorsement
ORDER BY score_per_endorsement DESC, user
LIMIT 10
```

Result:

```
user | score    | trusted_by | score_per_endorsement
-----+----------+------------+----------------------
1918 | 0.001074 | 328        | 3.27                 
1815 | 0.000819 | 270        | 3.03                 
843  | 0.001101 | 365        | 3.02                 
1471 | 0.000791 | 269        | 2.94                 
2227 | 0.001083 | 369        | 2.94                 
381  | 0.000854 | 299        | 2.86                 
301  | 0.001212 | 439        | 2.76                 
1935 | 0.000764 | 280        | 2.73                 
3685 | 0.000926 | 339        | 2.73                 
918  | 0.001333 | 497        | 2.68                 

10 rows, 23864 ms
```

## Where it earns its place

- Ranking influence in citation, trust, link and recommendation graphs.
- Prioritising review, moderation or crawling effort.
- Finding nodes whose standing is not explained by their raw connection count.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- Scores are a share of one and shrink with graph size. Compare ranks between graphs, never raw scores.
- A high score is structural importance, not business value. The graph knows nothing about which nodes matter to you.
- Damping, tolerance and the iteration ceiling are supplied together or not at all; a partial argument list is rejected.
- Direction is meaning here. Reversing the relationships reverses what the score says.

## See also

- [`graph.degree`](./graph-degree.md) for the local comparison
- [`graph.louvain`](../community/graph-louvain.md) for grouping rather than ranking
