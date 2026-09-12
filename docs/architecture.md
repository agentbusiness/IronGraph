# Architecture

IronGraph is a standalone, GPU-first, single-node graph database. It combines a property graph,
Cypher, transactions, durable storage, graph analytics, vector search, and compatible streaming
protocols in one database process.

This page describes the public architecture and operating guarantees. It is intended for developers
and platform engineers choosing a deployment shape or planning capacity.

## Architecture at a glance

```text
Application
  |-- Embedded Rust, Python, or Node.js
  |-- Query API client, browser/React client, or Bolt driver
  |-- Kafka-compatible producer/consumer or AMQP-compatible client
                         |
                  IronGraph process
                  |-- Explicit projects
                  |-- OBSERVED, KNOWLEDGE, and WORKSPACE layers
                  |-- Cypher query and administration surface
                  |-- CPU, Metal, or CUDA execution device
                  |-- Graph, temporal, text, and vector indexes
                  `-- Canonical WAL and periodic snapshots
                         |
                Caller-selected data directory
```

The standalone and embedded forms share the same database semantics. Embedding removes the network
hop; it does not create a reduced or secondary storage mode.

## Deployment model

IronGraph runs as one process on one host. It does not form a cluster and does not distribute a
project across machines. Choose between two deployment shapes:

| Mode | Best for | Connection boundary |
| --- | --- | --- |
| Embedded | Desktop software, local-first tools, services that own their database lifecycle | In-process Rust, Python, or Node.js calls |
| Standalone | Multiple clients, language-neutral access, browser applications, and protocol clients | Query API, Bolt, Kafka-compatible Streams, or AMQP-compatible Queues |

An embedded host chooses the database directory and owns startup and clean shutdown. Only one
embedded IronGraph instance can be active in a process. A standalone deployment owns the listeners
and can serve multiple supported clients while remaining a single database process.

## Projects and graph layers

A project is a named, isolated graph with its own data and derived indexes. There is no implicit
default project. Queries identify the project with a `USE` prefix or through the project context
provided by a client.

Each project contains three graph layers:

| Layer | Intended role | Default read behavior |
| --- | --- | --- |
| `OBSERVED` | Facts captured directly from a source or event | Included |
| `KNOWLEDGE` | Curated or consolidated facts | Included |
| `WORKSPACE` | Provisional, session-oriented, or application working data | Excluded unless requested |

An unprefixed layer selection reads `OBSERVED` and `KNOWLEDGE`. Include `WORKSPACE` explicitly when
the query needs working data:

```cypher
USE recommendations USE LAYER OBSERVED, KNOWLEDGE, WORKSPACE
MATCH (n)
RETURN n
```

Choose the write layer explicitly when writing outside `OBSERVED`:

```cypher
USE recommendations USE LAYER WORKSPACE WRITE LAYER WORKSPACE
CREATE (:Draft {name: 'candidate'})
```

Layers are database semantics, not separate databases or client-side tags. A query can read across
the selected layers as one graph view.

## Query and administration model

Cypher is the only query and administration language. The same surface creates graph data, declares
indexes and temporal properties, manages projects, and administers topics and queues. This keeps
automation and operational access consistent with application queries.

The only browser-facing data endpoint is `POST /api/query`. It returns a streamed query result. Bolt
remains available for supported drivers; exact protocol and client versions are supplied with each
IronGraph release. The web application is served below `/web/` and provides
three surfaces:

- **Query** for writing and running Cypher;
- **Graph**, presented as the Plot result view within Query, for visual exploration; and
- **Streams** for topics, queues, exchanges, bindings, and lag.

The browser interface and SDK clients use the database's supported query surface; they do not own a
second copy of graph data.

## Execution and residency

Every process selects exactly one execution device:

- **CPU** is the reference backend and the portable choice.
- **Metal** is the primary local accelerator on supported Apple hardware.
- **CUDA** is an optional build target for supported NVIDIA environments.

In a GPU-backed process, each admitted project's canonical graph rows and derived indexes remain
resident on the selected device. IronGraph reports an admission failure when the available device
capacity cannot hold the project. It does not silently page or truncate canonical graph rows.

Plan GPU capacity for the combined resident graph, its indexes, and the enabled text encoder. If a
workload cannot satisfy that requirement, use a larger device, reduce the admitted data set, or use
the CPU backend.

## Transactions and durability

Graph writes, index declarations, temporal declarations, and supported administration changes use
the database transaction and durability path. Committed changes are recorded in the canonical
write-ahead log. Periodic snapshots bound recovery work, and clean embedded shutdown creates a
snapshot before releasing the database.

WAL durability is asynchronous. Applications that coordinate dependent work should retain and use
the bookmark returned with a result where the client surface supports bookmarks. Treat the database
directory as one unit for storage, backup, and access control; do not share it between processes.

Documents are normal graph nodes, for example `(:Document {body: ...})`. Their source text remains
on the owning node and follows the same WAL and snapshot path as other graph data. There is no
document-specific store or REST endpoint.

## Derived indexes and text embedding

Range, text, vector, and temporal indexes accelerate graph operations while remaining rebuildable
from canonical graph data. Text embedding and vector search use declared graph indexes. When local
embedding is enabled, the verified encoder loads on the selected device and warms during startup.

Encoder startup and index residency contribute to startup time and device-memory requirements. A
production readiness check should therefore include encoder availability, warm-up completion, index
status, and device admission—not only listener availability.

## Streams and queues

Kafka-compatible topics and AMQP-compatible queues, exchanges, and bindings are protocol surfaces of
the same process. Use client and protocol versions listed as supported for your IronGraph release.
Their wire fields exist for client compatibility. Cypher remains the administration surface:

```cypher
USE recommendations SHOW TOPICS
```

```cypher
USE recommendations SHOW QUEUES
```

Use `SHOW CONSUMER LAG` to inspect consumption progress. Stream and queue data uses the same
single-node operating boundary and durability path; it is not an external broker cluster.

## Security boundaries

Plain listeners are for local development and bind only to loopback. Remote Query API, Bolt,
Kafka-compatible Streams, and AMQP-compatible Queues access requires mutual TLS. Both the client and
server present certificates, and the configured trust roots authenticate the connection.

A trusted certificate establishes identity; it does not grant unrestricted access. Before a remote
client connects, provision its certificate through the documented administration workflow and bind it
to the intended project, protocol, permitted operations, and graph layers. Remote credentials cannot
administer the project catalog. Obtain the assigned project context from the local administrator.

Keep certificate private keys and other credentials outside graph properties, query parameters that
may be logged by an application, result sets, and the database directory. Browser applications are
remote-only and rely on browser-managed client certificates for remote mutual TLS.
