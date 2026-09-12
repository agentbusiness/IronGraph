# Procedures

Twelve built-in graph algorithms. Every one is an IronGraph extension: standard Cypher has no procedure catalogue, and these run inside an ordinary Cypher pipeline rather than over a separately projected graph.

They divide by the question they answer. Traversal asks what is reachable. Routing asks how to get there and what it costs. Centrality asks which nodes matter. Community asks how the graph divides. Structure asks how tightly it is knit.

## [Traversal procedures](./traversal/README.md)

Breadth-first and depth-first expansion from one source node. Both follow outgoing relationships only, and both report reachability rather than a route: use the routing procedures when you need the path itself.

| Page | Summary |
| --- | --- |
| [`graph.bfs`](./traversal/graph-bfs.md) | Every node reachable from one source, with its hop distance. |
| [`graph.dfs`](./traversal/graph-dfs.md) | Every node reachable from one source, in depth-first visit order. |

## [Shortest-path procedures](./shortest-path/README.md)

One route between two nodes, or the cheapest cost to everywhere from one node. The difference between them is what "shortest" is measured in: relationships, or the value of a relationship property.

| Page | Summary |
| --- | --- |
| [`graph.shortestpath`](./shortest-path/graph-shortestpath.md) | One path with the fewest relationships between two nodes, and its length. |
| [`graph.dijkstra`](./shortest-path/graph-dijkstra.md) | The cheapest cost from one source to every reachable node, measured by a numeric relationship property. |

## [Centrality procedures](./centrality/README.md)

Two different answers to "which nodes matter". Degree counts relationships. PageRank weighs an endorsement by the standing of whoever gave it. They disagree often, and where they disagree is usually the interesting part.

| Page | Summary |
| --- | --- |
| [`graph.degree`](./centrality/graph-degree.md) | Relationship counts per node, split by direction. |
| [`graph.pagerank`](./centrality/graph-pagerank.md) | Recursive importance: a score that weighs who points at you, not how many. |

## [Community and component procedures](./community/README.md)

Three ways of cutting a graph into groups. Two are structural facts about connectivity; the third is an optimisation whose answer depends on the algorithm as much as on the data.

| Page | Summary |
| --- | --- |
| [`graph.wcc`](./community/graph-wcc.md) | Weakly connected components: the islands of the graph, ignoring direction. |
| [`graph.scc`](./community/graph-scc.md) | Strongly connected components: groups where every node can reach every other, following direction. |
| [`graph.louvain`](./community/graph-louvain.md) | Communities found by modularity optimisation: groups that are denser inside than the graph is on average. |

## [Local structure procedures](./structure/README.md)

How tightly knit the graph is, measured three ways: closed triangles across the whole graph, the same idea per node, and how deep into a densely connected core each node survives.

| Page | Summary |
| --- | --- |
| [`graph.trianglecount`](./structure/graph-trianglecount.md) | One number: how many closed triangles the whole graph contains. |
| [`graph.clusteringcoefficient`](./structure/graph-clusteringcoefficient.md) | Per node, the share of its neighbours that are connected to each other. |
| [`graph.kcore`](./structure/graph-kcore.md) | How deep into the graph's densely connected interior each node survives. |
