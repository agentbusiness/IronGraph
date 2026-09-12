# IronGraph for Node.js

**Node.js access to a GPU-first temporal graph database with built-in streaming and queues.**

Work with graph relationships, text document and vector search, temporal property history, graph
analytics, and built-in Streams and Queues. `@irongraph/node` includes native embedded access,
remote Query API/Bolt clients, and TypeScript declarations. IronGraph is open-source software
released under the Apache License 2.0.

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

You need Node.js 20.17 or later and an official package. Native release targets are macOS 15+
ARM64 and Linux with glibc 2.28+ on ARM64 or AMD64. These are package build baselines; check your
release's qualification results before deploying. After configuring the official npm package
source supplied with your release:

```sh
npm install @irongraph/node
```

The matching release and its platform package must be available in that source. Keep npm's
optional dependencies enabled: they select the compiled database binary for your platform.
Installation does not require a Rust compiler.

## Store and read a text document

Choose a writable directory exclusively owned by your application process. Default startup
downloads and verifies the local embedding model if absent, then loads and warms it. Allow
network access and sufficient disk space on first use.

```javascript
const { EmbeddedDatabase } = require('@irongraph/node')

async function main() {
  const database = await EmbeddedDatabase.open('./irongraph-data', 'cpu')
  try {
    await database.query('CREATE PROJECT IF NOT EXISTS notes')
    const result = await database.query(
      `USE notes
       MERGE (document:Document {id: $id})
       SET document.body = $body
       RETURN document.body AS body`,
      null,
      { id: 'welcome', body: 'Graphs connect facts.' },
    )
    console.log(result.rows[0][0].value)
  } finally {
    await database.close()
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
```

Expected output: `Graphs connect facts.` The text document persists; running the example again
updates the same text document. Next, declare an embedding index on the text property and use
`MATCH … SEARCH … RETURN` to combine retrieval with graph filters.

The example selects CPU across native release targets and keeps automatic text embedding
enabled. Pass `'metal'` instead of `'cpu'` on a supported Mac to use Metal acceleration.

## Use the database capabilities

The same `query` method runs text document updates, index and temporal statements, graph algorithms,
and topic, queue, exchange, and binding administration. Text documents are ordinary graph nodes.
Their complete text persists through the same transactions, asynchronous write-ahead logging,
recovery, and snapshots as other graph data.

Select projects explicitly with `USE` or the `projectId` argument. Each project has `OBSERVED`,
`KNOWLEDGE`, and `WORKSPACE` layers for source facts, curated facts, and working data. There is
no implicit default project.

One embedded instance owns its directory; one process selects one device. CPU is the reference
backend, Metal is the primary local accelerator, and CUDA requires a CUDA-enabled release.
GPU admission rejects project graphs and indexes that do not fit. IronGraph embeds text
locally and does not run generative language models.

Use `Client.apiMtls` or `Client.boltMtls` for remote access with certificate, private-key, and
certificate-authority paths. Remote connections require mutual TLS and project authorization;
plain connections are loopback-only. For a browser or React application, use
`@irongraph/client` to connect to a running instance.

## License

IronGraph is licensed under the Apache License 2.0. Third-party components and the embedding model
retain their own terms.
