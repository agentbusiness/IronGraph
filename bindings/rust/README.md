# IronGraph for Rust

**Rust access to a GPU-first temporal graph database with built-in streaming and queues.**

Work with graph relationships, text document and vector search, temporal property history, graph
analytics, and built-in Streams and Queues. `irongraph-sdk` provides a Rust interface to the
native database and remote Query API/Bolt clients. IronGraph is open-source software released
under the Apache License 2.0.

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

You need Rust 1.94 or later, a native linker, and an official release. Native release targets
are macOS 15+ ARM64 and Linux with glibc 2.28+ on ARM64 or AMD64. These are package build
baselines; check your release's qualification results before deploying. Once the matching SDK
and native artifacts are available from the official distribution channel:

```sh
cargo add irongraph-sdk
```

The SDK automatically downloads a version-specific native archive and verifies its checksum.
Allow network access during the first build. Your application compiles the Rust interface and
links the precompiled database; the database implementation is supplied only as a binary.

For disconnected builds, obtain the official native archive for your exact SDK version and
target while online. Verify it against the release checksums, then copy it into an absolute
directory with the filename `libirongraph_ffi.a`. Set `IRONGRAPH_NATIVE_DIR` to that directory
before building. Cargo dependencies must also be cached or vendored before using `cargo --offline`.
The automatic download cache alone does not support every fresh offline build. Offline database
startup additionally requires the embedding model to be installed before disconnecting.

## Store and read a text document

Use a writable directory exclusively owned by your application process. Default startup
downloads and verifies the local embedding model if absent, then loads and warms it. Allow
network access and sufficient disk space on first use.

```rust
use irongraph_sdk::{EmbeddedDatabase, EmbeddedOptions, ExecutionDevice, Query};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new("./irongraph-data").with_execution_device(ExecutionDevice::Cpu),
    )?;
    database.query(Query::new("CREATE PROJECT IF NOT EXISTS notes"))?;
    let result = database.query(Query::new(
        "USE notes
         MERGE (document:Document {id: 'welcome'})
         SET document.body = 'Graphs connect facts.'
         RETURN document.body AS body",
    ))?;
    println!("{}", result.rows[0][0]["value"].as_str().unwrap());
    database.close()?;
    Ok(())
}
```

Expected output: `Graphs connect facts.` The text document persists; running the example again
updates the same text document. Use `Query::with_parameter` for application-supplied values.
Next, declare an embedding index on the text property and use `MATCH … SEARCH … RETURN`
to combine retrieval with graph filters.

The example selects CPU across native release targets and keeps automatic text embedding
enabled. Select `ExecutionDevice::Metal(0)` on a supported Mac to use Metal acceleration.

## Use the database capabilities

The same `query` method supports text document updates, indexes, temporal data, graph algorithms,
and topic, queue, exchange, and binding administration. Text documents are ordinary graph nodes;
their complete text persists through the same transactions, asynchronous write-ahead logging,
recovery, and snapshots as other data.

Select projects explicitly with `USE` or `Query::with_project`; there is no implicit default.
Each project has `OBSERVED`, `KNOWLEDGE`, and `WORKSPACE` layers for source facts, curated facts,
and working data. Query results retain typed values, including vectors and temporal values.

One embedded instance owns its directory; one process selects one device. CPU is the reference
backend, Metal is the primary local accelerator, and CUDA requires a CUDA-enabled release.
GPU admission rejects project graphs and derived indexes that do not fit. Keep the database
open for the application's lifetime and call `close` during orderly shutdown. IronGraph
embeds text locally and does not run generative language models.

Use `RemoteClient::api_mtls` or `RemoteClient::bolt_mtls` with `MutualTls` for remote access.
Remote connections require mutual TLS and project authorization. Plain connections are
loopback-only.

## License

IronGraph is licensed under the Apache License 2.0. Third-party components and the embedding model
retain their own terms.
