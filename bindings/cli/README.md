# IronGraph

**Embeddable Cypher graph database with sub-µs point reads, built-in streams, queues, and local text embeddings. Standalone launcher.**

Use Cypher to query graph structure, search text and vectors, inspect property history, and
administer the database. Built-in Kafka-compatible Streams and AMQP-compatible Queues handle event
and message workloads. The `irongraph` CLI starts and manages a standalone database with its
built-in web console. IronGraph is open-source software released under the Apache License 2.0.

## Database capabilities

| Capability | What you can do |
| --- | --- |
| **Graph queries and analytics** | Match relationships and paths with Cypher; run shortest paths, PageRank, connected components, and community detection. |
| **Temporal properties and analysis** | Retain property history, read values with `AT TIME`, inspect samples with `HISTORY`, and aggregate with time windows and maintained rollups. |
| **Text documents and search** | Store the full text of articles, notes, manuals, and other written content on graph nodes; search that text with text and vector indexes and automatic local embeddings. |
| **Built-in Kafka-compatible Streams** | Produce and consume events through supported Kafka clients; manage topics, partitions, retention, and consumer-lag monitoring. |
| **Built-in AMQP-compatible Queues** | Use classic and stream queues, direct/fanout/topic exchanges, bindings, and retention with supported AMQP clients. |

Cypher handles graph operations, temporal analysis, search, and stream/queue administration.
Message producers and consumers use the supported Kafka or AMQP protocols through the instance's
configured listeners. The SDK query interface does not replace those protocol clients.

Explicit projects and the `OBSERVED`, `KNOWLEDGE`, and `WORKSPACE` layers organize data.
Indexes and unique constraints, transactions, asynchronous write-ahead logging, recovery, and
periodic snapshots are part of the database.

The standalone instance serves its web console at
[http://127.0.0.1:18484/web/](http://127.0.0.1:18484/web/) by default and also supports local MCP
access. Start the official standalone distribution before opening that address. Opening an
embedded database through Python, Node.js, or Rust does not start a web server or expose the
console. For a configured remote instance, use its HTTPS address followed by `/web/`, with
the required browser-managed client certificate.

Temporal reads apply to declared properties within their retention window. Graph topology and
ordinary properties are read in their current state.

A project is an isolated graph with its own schema and indexes. `OBSERVED` holds source facts,
`KNOWLEDGE` holds curated facts, and `WORKSPACE` holds provisional working data. There is no
implicit default project.

Graph queries, algorithms, indexes, and vector retrieval use one shared CPU graph. Text inference
runs asynchronously on an independently selected CPU, Metal, or CUDA device. Parallel readers do
not wait for writers and may observe mixed values during a multi-record write.

## Choose your package

| Application | Package | Access |
| --- | --- | --- |
| Standalone database and web console | `irongraph` on npm, or native archive | Local process and remote listeners |
| Python | `irongraph` | Embedded and remote |
| Node.js | `@irongraph/node` | Embedded and remote |
| Rust / Cargo | `irongraph-sdk` | Embedded and remote |
| Browser / React | `@irongraph/client`, with `/react` exports | Remote Query API |

Install from the official package source or artifact bundle supplied with your release. An
installation command requires the matching release to be available in that source. Native
release targets are macOS 15+ ARM64 and Linux with glibc 2.28+ on ARM64 or AMD64. These are the
package build baselines; deployment support depends on the qualification results supplied with
your release. CUDA requires a package built with CUDA support.

## Start with a text document in Python

You need Python 3.9 or later, a matching official wheel, and a writable directory owned by one
application process. Default startup downloads, loads, and warms the local Bekko a8m embedding
model; allow network access and space for the model on first use.

After configuring your official package source, install the binary package:

```sh
python -m pip install --only-binary=:all: irongraph
```

Save the following as `example.py`:

```python
from irongraph import EmbeddedDatabase

with EmbeddedDatabase("./irongraph-data", device="cpu") as database:
    database.query("CREATE PROJECT IF NOT EXISTS notes")
    result = database.query(
        """USE notes
        MERGE (document:Document {id: $id})
        SET document.body = $body
        RETURN document.body AS body""",
        parameters={"id": "welcome", "body": "Graphs connect facts."},
    )
    print(result["rows"][0][0]["value"])
```

Run it from the directory containing the file:

```sh
python example.py
```

Expected output: `Graphs connect facts.` The text document persists in the selected directory. Run
the example again to update and read the same text document. Next, declare an embedding index on its
text property and use `MATCH … SEARCH … RETURN` to combine retrieval with graph context.

The example selects CPU so the same code works across native release targets, while automatic
text embedding stays enabled. Select `embedding_device="metal"` on a supported Mac to accelerate text inference.

## Deploy deliberately

Each database process uses one shared CPU graph and independently selects its text inference device. One
embedded instance owns its directory exclusively. Keep it open for your application's lifetime
and close it during orderly shutdown.

The local embedding model processes text; IronGraph does not host or invoke generative language
models. Text documents use the same storage, transactions, and recovery as other graph data.

Plain listeners bind only to loopback. Remote Query API, Bolt, Kafka-compatible Streams, and
AMQP-compatible Queues require mutual TLS. Browser certificates are managed by the browser.
The web console provides Query with a Plot view, Streams, Documents, Training, Docs, and Settings.
Local MCP exposes the Query API through stdio or loopback HTTP.

## License

IronGraph is licensed under the Apache License 2.0. Third-party components and the embedding model
retain their own license terms.
