# Community and component procedures

Three ways of cutting a graph into groups. Two are structural facts about connectivity; the third is an optimisation whose answer depends on the algorithm as much as on the data.

| Page | Summary | Standard |
| --- | --- | --- |
| [`graph.wcc`](./graph-wcc.md) | Weakly connected components: the islands of the graph, ignoring direction. | extension |
| [`graph.scc`](./graph-scc.md) | Strongly connected components: groups where every node can reach every other, following direction. | extension |
| [`graph.louvain`](./graph-louvain.md) | Communities found by modularity optimisation: groups that are denser inside than the graph is on average. | extension |
