# IronGraph for JavaScript and React

**Connect JavaScript and React to a GPU-first temporal graph database with built-in streaming and queues.**

Query graph relationships, text documents, text and vectors, temporal property history, and graph
analytics; administer Streams and Queues through Cypher. `@irongraph/client` provides a typed
Query API client and React hooks for a running IronGraph instance. The browser package
is remote-only; the database runs in a separate process. IronGraph is open-source software
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

You need a browser with Fetch and streaming response support, an ES module application, and
React 18 or later if using the React exports. After configuring the official npm package source
supplied with your release:

```sh
npm install @irongraph/client
```

The matching release must be available in that source. React is supplied by your application;
the plain JavaScript client does not require it.

## Query a local instance

For this example, run an official IronGraph instance on loopback port `18484`. In its Query
console, create a project with `CREATE PROJECT IF NOT EXISTS notes`. Serve the browser
application from the instance's origin so the request has same-origin access.

```javascript
import { Client } from '@irongraph/client'

const client = new Client('http://127.0.0.1:18484')
const result = await client.query({
  cypher: "USE notes RETURN 'Connected to IronGraph' AS message",
})
console.log(result.rows[0][0].value)
```

Expected output: `Connected to IronGraph`. Next, query your project's text document nodes using
`MATCH (document:Document) RETURN document.body` after `USE notes`.

## Use React

With the same local prerequisites, render this component in your application:

```jsx
import { IronGraphProvider, useIronGraphQuery } from '@irongraph/client/react'

function Connection() {
  const { result, error, loading } = useIronGraphQuery({
    cypher: "USE notes RETURN 'Connected to IronGraph' AS message",
  })
  if (loading) return <p>Connecting…</p>
  if (error) return <p>{String(error)}</p>
  return <pre>{JSON.stringify(result?.rows)}</pre>
}

export default function App() {
  return (
    <IronGraphProvider baseUrl="http://127.0.0.1:18484">
      <Connection />
    </IronGraphProvider>
  )
}
```

Expected result: the typed row containing `Connected to IronGraph` appears. The hook exposes
local loading and error states and cancels the previous request when its inputs change or the
component unmounts. Next, replace the statement with your application's graph query.

## Query the graph, text documents, and search

Cypher provides text document updates, vector search, indexes, temporal data, algorithms, and stream
and queue administration through `POST /api/query`, subject to your credential's permissions.
Text documents are ordinary graph nodes whose complete text participates in graph relationships and
declared embedding indexes. Select a project explicitly; there is no implicit default.
`OBSERVED`, `KNOWLEDGE`, and `WORKSPACE` distinguish source facts, curated facts, and working data.

Remote deployments require HTTPS and browser-managed mutual TLS certificates authorized for
the required project and operations. Plain HTTP is loopback-only. The database process selects
one CPU, Metal, or optional CUDA device; GPU admission rejects graphs and indexes that do not
fit. Embedding runs in the database process, with no generative language model hosted by
IronGraph.

Native database release targets are macOS 15+ ARM64 and Linux with glibc 2.28+ on ARM64 or
AMD64. These package build baselines apply to the database host; check its release qualification
results before deploying. The browser client does not contain a native database binary.

## License

IronGraph is licensed under the Apache License 2.0.
