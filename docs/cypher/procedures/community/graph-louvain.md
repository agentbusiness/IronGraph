# `graph.louvain`

> Communities found by modularity optimisation: groups that are denser inside than the graph is on average.

| | |
| --- | --- |
| Kind | Procedure |
| Signature | `graph.louvain() YIELD node, community` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`email`](../../datasets.md#email) — European research institution email network |

## What it does

`graph.louvain` partitions the graph into communities by repeatedly moving nodes between groups to increase modularity — the degree to which relationships fall inside groups rather than between them. It yields one row per node with an integer community identifier.

Unlike the component procedures, this is an optimisation and not a fact. The partition it returns is a good one, not the only good one.

## How it behaves

Every node receives a community, including nodes that belong nowhere in particular. The algorithm does not report confidence, so a community assignment carries no claim that the node really belongs there. Judge the partition as a whole, by whether communities line up with something you already know about the data, rather than trusting any individual assignment.

Community identifiers have no meaning beyond grouping, and modularity optimisation has a known resolution limit: below a certain size, genuinely distinct groups get merged because splitting them does not improve the global score. Small communities in the output are less trustworthy than large ones.

## When to use it

Use it to find structure you have no labels for — the natural groupings in a network nobody has categorised. Where labels already exist, the interesting use is comparison: agreement confirms the labels describe real structure, and disagreement points at where they do not.

## How it differs from its neighbours

`graph.wcc` and `graph.scc` answer a connectivity question with one correct answer. Louvain answers a density question with a good answer. Two nodes in different Louvain communities are usually still connected; two nodes in different weakly connected components never are.

## Simple example

The communities Louvain finds in the email network of a European research institution, by size.

```cypher
USE email
CALL graph.louvain() YIELD node, community
WITH community, count(node) AS members
RETURN community, members
ORDER BY members DESC, community
LIMIT 10
```

Result:

```
community | members
----------+--------
3         | 262    
4         | 240    
0         | 183    
1         | 150    
2         | 95     
5         | 56     
6         | 1      
7         | 1      
8         | 1      
9         | 1      

10 rows
```

## Advanced example

The same communities checked against ground truth. Every member of this network has a known department, which the algorithm never sees. For each community this reports its dominant department and what share of the community that department accounts for — a direct measure of whether the structure found matches the structure that exists.

```cypher
USE email
CALL graph.louvain() YIELD node, community
WITH community, node.department AS department, count(*) AS members
ORDER BY community, members DESC
WITH community, collect(department) AS departments,
     collect(members) AS counts, sum(members) AS size
WHERE size >= 20
RETURN community, size,
       head(departments) AS dominant_department,
       head(counts) AS from_that_department,
       round(1000.0 * head(counts) / size) / 10.0 AS purity_percent
ORDER BY size DESC, community
```

Result:

```
community | size | dominant_department | from_that_department | purity_percent
----------+------+---------------------+----------------------+---------------
3         | 262  | 15                  | 48                   | 18.3          
4         | 240  | 4                   | 97                   | 40.4          
0         | 183  | 1                   | 57                   | 31.1          
1         | 150  | 21                  | 53                   | 35.3          
2         | 95   | 14                  | 88                   | 92.6          
5         | 56   | 17                  | 32                   | 57.1          

6 rows
```

## Where it earns its place

- Finding structure in a network that has never been categorised.
- Testing whether existing labels describe real structural groups.
- Reducing a large graph to a manageable number of groups before analysis.

## Limitations and trade-offs

- An algorithm reads the whole project graph under the query's selected layers. There is no separate projection step and no way to restrict an algorithm to a label or relationship type; filter the rows it yields, or keep the data you want analysed in its own project.
- Scores and components are query results. They are not written back to the graph, so nothing derived becomes a canonical node or relationship unless you write it yourself.
- A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows you want it driven by, or the algorithm runs again for each one.
- The partition is an optimisation result, not a fact about the data. A different run over changed data can reorganise groups substantially.
- Every node gets a community, including nodes that belong to none. There is no confidence output.
- Modularity has a resolution limit: small genuine communities are merged into larger ones. Treat small communities with suspicion.
- Community identifiers are grouping labels only and are not stable identity.

## See also

- [`graph.wcc`](./graph-wcc.md) for connectivity as a fact
- [`graph.kcore`](../structure/graph-kcore.md) for cohesion by depth rather than grouping
