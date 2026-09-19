# Features

IronGraph brings graph storage, Cypher, analytics, vector search, and compatible event protocols
into one GPU-first, single-node database. This page highlights the supported capabilities and the
operating boundaries that accompany them.

## Cypher-native graph development

Use Cypher for both data queries and administration. IronGraph supports property-graph patterns,
parameters, filtering, aggregation, ordering, variable-length paths, writes, transactions, and
graph procedures without introducing a second query language for database operations.

```cypher
USE fraud
MATCH path = (account:Account)-[:TRANSFERRED_TO*1..4]->(destination:Account)
WHERE account.id = 'acct-100'
RETURN path
```

Projects are explicit, so an application cannot fall through to an accidental default graph.

## First-class graph layers

Every project can distinguish:

- `OBSERVED` facts captured from source activity;
- `KNOWLEDGE` facts that represent curated understanding; and
- `WORKSPACE` data used for provisional or application working state.

Queries can read one layer or a combined view and can select the write layer. The default read view
includes `OBSERVED` and `KNOWLEDGE`; `WORKSPACE` remains explicit.

## GPU-first execution with a CPU reference

IronGraph selects one execution device per process. Metal is the primary local accelerator, CUDA is
an optional build target, and CPU is the complete reference backend.

GPU-backed processes keep each admitted project graph and its derived indexes resident. If the graph
does not fit, IronGraph reports admission failure instead of silently paging or truncating canonical
rows. This makes device capacity an explicit deployment decision.

## Indexes and constraints

Declared graph indexes support common lookup and retrieval patterns:

- equality indexes for exact property lookup;
- range indexes for ordered property comparison;
- text indexes for text retrieval;
- vector indexes for similarity search; and
- temporal indexes for time-aware property access.

Index state is visible through Cypher:

```cypher
USE catalog SHOW INDEXES
```

Unique constraints and index administration share the same project and durability boundaries as
graph data.

## Temporal graph data

Temporal properties let applications query current and historical values through Cypher. Temporal
declarations belong to node or relationship properties and include a retention policy. Time-aware
queries remain part of the graph rather than moving history into a separate database.

## Graph algorithms

Built-in graph procedures cover frequently used structural analysis, including:

- breadth-first and depth-first traversal;
- unweighted and weighted shortest paths;
- weakly and strongly connected components;
- PageRank;
- triangle counting and clustering coefficient;
- k-core decomposition; and
- Louvain community detection.

Run algorithms in the context of an explicit project and selected graph layers. Results remain query
results; scores and processing intermediates do not become canonical graph entities unless the
application deliberately writes domain data.

## Vectors, text embedding, and documents

Nodes and relationships contribute meaningful content to automatic semantic search when the local
embedding model is enabled. This includes documents, emails, tables, people, calendar plans, and
tasks represented in the graph. Relationship meaning includes the relationship type, its content,
and identifying names from its endpoints. Operational metadata, identifiers, and source URLs are
excluded by field-selection rules.

Existing content is embedded when the project is prepared; later writes maintain its searchable
representation automatically. Source text remains complete. Derived vectors do not create extra
graph entities. You can also declare an embedding index for a specific node text property or a
vector index for vectors you supply.

```cypher
USE knowledge
SEARCH entity IN (EMBEDDING INDEX graph_semantic
                  FOR TEXT 'plans for the product launch' LIMIT 10)
  SCORE AS score
RETURN entity, score
```

The query returns up to ten nodes and relationships ordered by descending similarity score.
Vector retrieval runs on the selected execution device; GPU-backed instances use their selected
GPU, and CPU instances use the CPU backend.

A document is an ordinary graph node:

```cypher
USE knowledge
CREATE (:Document {
  title: 'Device handbook',
  body: 'Complete source text remains on this node.'
})
```

The document follows the normal transaction, WAL, and snapshot path. There is no document-specific
REST endpoint or second document store. Source text remains complete on its owning node.

The verified local text encoder installs, loads, binds to the selected device, and warms at startup
when embedding support is enabled. Encoder memory must be included in device-capacity planning.

## Transactions, recovery, and snapshots

IronGraph provides transactions, write-ahead-log recovery, periodic snapshots, and asynchronous WAL
durability. The embedded and standalone forms use the same canonical persistence model. Applications
choose one database directory and should protect and back up that directory as a unit.

## Application access

| Client | Embedded | Query API | Bolt |
| --- | :---: | :---: | :---: |
| Rust | Yes | Yes | Yes |
| Python | Yes | Yes | Yes |
| Node.js | Yes | Yes | Yes |
| Browser and React | No | Yes | No |
| Supported Bolt driver | No | No | Yes |

The Query API is available at `POST /api/query`. It is the only browser-facing data endpoint and
streams typed query results. The built-in web application provides Query, Streams, Documents,
Training, Docs, and Settings below `/web/`. Graph is the Plot result view inside Query.

## Kafka-compatible Streams

Applications can use Kafka-compatible topic operations for producers and consumers. Create,
inspect, clear, and delete topics with Cypher, and inspect consumer progress with `SHOW CONSUMER
LAG`.

```cypher
USE events CREATE TOPIC activity PARTITIONS 12 RETENTION 30 DAYS
```

Change the age-based retention window without recreating the topic:

```cypher
USE events ALTER TOPIC activity RETENTION 90 DAYS
```

Kafka wire compatibility supports the client and protocol versions identified with each IronGraph
release; it does not turn the single-node database into a distributed Kafka cluster.

## AMQP-compatible Queues

Applications can use AMQP-compatible queues, exchanges, and bindings with the client and protocol
versions identified with each IronGraph release. Cypher manages classic and stream queues, direct,
fanout, and topic exchanges, bindings, purging, and deletion.

```cypher
USE jobs
CREATE EXCHANGE routing TYPE TOPIC
```

```cypher
USE jobs
CREATE QUEUE image_processing STREAM RETENTION 14 DAYS
```

Change the queue retention window with the same day-based form:

```cypher
USE jobs ALTER QUEUE image_processing RETENTION 30 DAYS
```

```cypher
USE jobs
BIND QUEUE image_processing TO EXCHANGE routing KEY images
```

Remote Streams and Queues connections require mutual TLS, like remote Query API and Bolt access.

## Built-in developer interface

The web application is intentionally focused:

- **Query** runs Cypher and displays typed results.
- **Graph** explores returned graph data through the Plot result view in Query.
- **Streams** administers and monitors topics and queues through Cypher.
- **Documents** creates, lists, searches, edits, and removes ordinary `:Document` nodes through
  Cypher.
- **Training** guides you through the open reference datasets and runs explained investigations.
- **Docs** provides a searchable reader for the bundled public documentation and runs its Cypher
  examples.
- **Settings** controls console preferences and local AI integrations.

These views share the same database and public query surface. They do not introduce a parallel data
model or administration API.

## Operating boundaries

IronGraph is designed around explicit constraints:

- one process on one node;
- one selected CPU, Metal, or CUDA device;
- one canonical WAL and snapshot path;
- no implicit default project;
- no silent GPU paging or canonical-row truncation; and
- mutual TLS for every remote Query API, Bolt, Streams, or Queues connection.

These boundaries are part of the product model and should be reflected in application lifecycle,
capacity planning, and deployment design.
