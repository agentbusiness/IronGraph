# IronGraph for Python

**Python access to a GPU-first temporal graph database with built-in streaming and queues.**

Work with graph relationships, text document and vector search, temporal property history, graph
analytics, and built-in Streams and Queues. The `irongraph` package provides native embedded
access and remote Query API/Bolt clients with the same Cypher language. IronGraph is open-source
software released under the Apache License 2.0.

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

## Install

You need Python 3.9 or later and an official wheel. Native release targets are macOS 15+ ARM64
and Linux with glibc 2.28+ on ARM64 or AMD64. These are package build baselines; check your
release's qualification results before deploying. After configuring the official package source
supplied with your release:

```sh
python -m pip install --only-binary=:all: irongraph
```

The matching wheel must be published to that source. Installation uses a compiled database
binary and does not require a Rust compiler.

## Store and read a text document

Use a writable directory exclusively owned by your application process. Default startup
downloads and verifies the local embedding model if absent, then loads and warms it. Allow
network access and sufficient disk space on first use.

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

Expected output: `Graphs connect facts.` The context manager closes the database and the
text document persists. Run the example again to update the same text document.

The example selects CPU across native release targets and keeps automatic text embedding
enabled. Select `device="metal"` on a supported Mac to use Metal acceleration.

Next, declare an embedding index on the text document body and use `MATCH … SEARCH … RETURN` to
combine vector retrieval with graph filters. The same `query` method supports text document updates,
indexes, temporal data, graph algorithms, and topic, queue, exchange, and binding administration.
Text documents are ordinary graph nodes; their complete text remains on the owning node.

## Operate the database

Choose projects explicitly with `USE` or `project_id`; there is no implicit default. Each project
has `OBSERVED`, `KNOWLEDGE`, and `WORKSPACE` layers for source facts, curated facts, and working
data. Transactions, asynchronous write-ahead logging, recovery, and snapshots apply to embedded
data as they do in a standalone instance.

One process uses one execution device and one active embedded instance. CPU is the reference
backend, Metal is the primary local accelerator, and CUDA requires a CUDA-enabled release.
GPU admission rejects project graphs and derived indexes that do not fit in device memory.
Keep the database open for the application's lifetime. IronGraph embeds text locally and does
not run generative language models.

For remote access, use `Client.api_mtls` or `Client.bolt_mtls` with a client certificate,
private-key path, and trusted certificate authority. Plain connections are loopback-only;
remote access requires mutual TLS and a credential authorized for the project and operations.

## License

IronGraph is licensed under the Apache License 2.0. Third-party components and the embedding model
retain their own terms.
