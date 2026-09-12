# Centrality procedures

Two different answers to "which nodes matter". Degree counts relationships. PageRank weighs an endorsement by the standing of whoever gave it. They disagree often, and where they disagree is usually the interesting part.

| Page | Summary | Standard |
| --- | --- | --- |
| [`graph.degree`](./graph-degree.md) | Relationship counts per node, split by direction. | extension |
| [`graph.pagerank`](./graph-pagerank.md) | Recursive importance: a score that weighs who points at you, not how many. | extension |
