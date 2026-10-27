# Cypher reference

This is the reference for Cypher as IronGraph implements it: the clauses, functions and procedures
you can run, what each one does, and where each one differs from Cypher as you may know it from
elsewhere.

Every page follows the same shape, and every page answers the same three questions in its header
table: what kind of thing this is, whether it is standard Cypher or an IronGraph addition, and which
dataset its examples run against.

## Run the examples

Reference pages carry simple and advanced examples. Dataset-backed pages state which project to
load. Self-contained pages use an existing disposable project and construct their own values or graph
records. Read prerequisites before running a write example. Each execution submits one statement.

Result tables show example executions. Identities, revisions, clocks, random values, timings and
graph-dependent output vary with the current project and host. Expected-result prose states the
behavior to check rather than promising a fixed value for a changing graph.

## The datasets

Nine open datasets, chosen so that every capability has somewhere real to be demonstrated. Load
them once and every example on every page becomes runnable.

| Dataset | Nodes | Relationships | What it is here for |
| --- | ---: | ---: | --- |
| [`flights`](./datasets.md#flights) | 13,859 | 66,770 | Weighted routing: shortest paths, Dijkstra, traversal, numeric and string functions. |
| [`trust`](./datasets.md#trust) | 5,881 | 35,592 | The temporal reference dataset: real 2010-2016 event times, declared temporal properties, HISTORY, AT TIME, WINDOW, rollups and every aggregate. |
| [`epinions`](./datasets.md#epinions) | 75,879 | 508,837 | Centrality and component structure: PageRank, WCC, SCC, k-core, degree. |
| [`dblp`](./datasets.md#dblp) | 322,080 | 1,162,094 | Community detection at scale, compared against published communities. |
| [`citations`](./datasets.md#citations) | 58,572 | 352,807 | Text and document work: text indexes, string functions, dates, documents. |
| [`library`](./datasets.md#library) | 8 | 0 | A compact paper collection for text and semantic-search examples. |
| [`social`](./datasets.md#social) | 4,039 | 88,234 | Dense undirected neighbourhoods: triangle counting and clustering coefficient. |
| [`email`](./datasets.md#email) | 1,005 | 24,929 | Small labelled graph with 42 known departments for checking community output. |
| [`overflow`](./datasets.md#overflow) | 24,818 | 506,550 | Large timestamped interaction stream for windowed time analysis. |

Full descriptions, sources, licences and graph models are on the [datasets page](./datasets.md),
including how to load them.

## How to read this reference

- **[Clauses](./clauses/README.md)** — graph patterns, reads and writes, projections, procedure calls,
  time, layers and similarity search.
- **[Statements](./statements/README.md)** — projects, indexes, constraints and temporal schema.
  Administration is Cypher; there is no second language for it.
- **[Functions](./functions/README.md)** — numeric, string, list, entity, conversion, temporal,
  aggregate and vector functions.
- **[Procedures](./procedures/README.md)** — the twelve built-in graph algorithms: traversal,
  routing, centrality, community detection and local structure.

See [expressions and value semantics](./expressions.md) for operators, nulls, parameters,
comprehensions, CASE and temporal fields. Use [automatic semantic scope](./statements/indexes/alter-automatic-semantic.md)
when a project's structural relationships should stay outside semantic search.

## Reading the header table

Each page states its relationship to standard Cypher in one of three ways.

- **Standard Cypher** — uses the named Cypher construct with the supported behavior described on
  that page. This label does not promise every form or option available in another implementation.
- **Standard Cypher, extended by IronGraph** — the name is familiar, but the behaviour or the
  available forms go further than the standard describes. Read the page before assuming.
- **IronGraph extension** — no equivalent in standard Cypher. Everything under Procedures is an
  extension, as is everything to do with time, layers, projects and vectors.
