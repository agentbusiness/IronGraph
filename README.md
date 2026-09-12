# IronGraph

**An embeddable, GPU-first temporal graph database with automatic local embeddings, streaming,
and queues.**

Run IronGraph inside a Python, Node.js, or Rust application, or connect to it as a standalone
single-node database. Store relationships and complete text documents in the same graph. Declare
text and vector indexes, and IronGraph automatically installs, verifies, loads, and warms its local
embedding model on the selected execution device.

Use Cypher to query graph structure, search text and vectors, inspect property history, and
administer the database. Built-in Kafka-compatible Streams and AMQP-compatible Queues handle event
and message workloads. The browser package provides a typed client and React hooks. IronGraph is
open-source software released under the Apache License 2.0.

## Start the database and open the web console

Use this path to run IronGraph as a standalone database with its built-in web console.
Python, Node.js, and Rust embedding are separate options described below.

### 1. Check the prerequisites

You need Node.js 20.17 or later with npm, and an official `irongraph` release available on npm
for your platform: macOS 15+ ARM64, or Linux ARM64/AMD64 with glibc 2.28+. Before the first
official publication, these npm commands are not an available installation channel.

Use a writable local data directory owned exclusively by this database process. First startup
also needs network access and several gigabytes of free storage for the automatically installed
local text embedding model. The launcher selects a compiled database binary with the web console
included. You do not need Python, a Rust compiler, or a separate frontend server.

### 2. Start IronGraph

Run this command to install the matching package as needed and start in the background:

```sh
npx irongraph start --background
```

Your data lives in `~/.irongraph/data`. The database keeps running after you close the terminal.
Wait for startup and the embedding model's installation, loading, and warm-up to finish. Check
progress and readiness with:

```sh
npx irongraph logs
npx irongraph status
```

The process chooses its execution backend automatically. To explicitly use CPU, add
`--execution-backend cpu`; on a supported Mac, use `--execution-backend metal`. To run in the
foreground with live terminal output, use `npx irongraph start` without `--background`.

### 3. Open the web console

When status reports **ready**, open **[http://127.0.0.1:18484/web/](http://127.0.0.1:18484/web/)**
in a browser on the same computer.
This address belongs to your running local instance, not a public project page.

The console provides Query, Streams, Documents (for text documents), Training, Docs, and
Settings. The graph visualization is the Plot result view inside Query.

### 4. Create your first project

Click **New** in the top bar, enter `notes`, and click **Create project**. The console selects
your new project. In **Query**, replace the editor contents with:

```cypher
USE notes
RETURN 'IronGraph is ready' AS status
```

Click **Run**, then select the **Table** result view. Expected result: a `status` column
containing `IronGraph is ready`. Every graph operation needs
an explicit project; IronGraph has no implicit default project. Next, open **Documents** to work
with text documents in your project, or continue in Query to create nodes and relationships.

### 5. Stop and restart

For a background instance, run:

```sh
npx irongraph stop
```

For a foreground instance, press `Ctrl+C` and wait for the process to exit. Start again with
`npx irongraph start --background` to reopen the same data directory. Your data persists between
runs. Background mode does not install a system service or restart the database after a reboot.

### Configuration and direct binary use

Choose another directory with `--data-dir /absolute/path`. Pass the same directory to `status`,
`logs`, and `stop`. A second start cannot open a second managed instance on that directory.

Local listener defaults are consistent across development and releases:

| Surface | Loopback port |
| --- | --- |
| Web console and Query API | `18484` |
| Bolt | `18485` |
| Kafka-compatible Streams | `18486` |
| AMQP-compatible Queues | `18487` |
| Local MCP HTTP | `18488` |

The frontend development server uses `18489`; the standalone console is always served by the
database's HTTP listener. Embedded libraries do not start these listeners.

If the console does not open, check that the process is still running and startup completed.
If port `18484` is already occupied, start with `--http-addr 127.0.0.1:19484` and open
`http://127.0.0.1:19484/web/` instead. Keep plain listeners on loopback; access from another
computer requires a configured remote listener with mutual TLS.

You can also extract the matching official standalone archive and run its executable directly,
without Node.js or npm:

```sh
./bin/irongraph --data-dir "$HOME/.irongraph/data" --http-addr 127.0.0.1:18484
```

Direct execution runs in the foreground. Stop it with `Ctrl+C`; launcher commands manage only
instances started through the launcher. The archive also includes `bin/irongraph-mcp` for local
stdio MCP hosts.

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

CPU is the reference backend, Metal is the primary local accelerator, and CUDA is an optional
build target. GPU execution keeps admitted project graphs and their derived indexes resident
on the selected device.

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
application process. Default startup installs, verifies, loads, and warms the local embedding
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
text embedding stays enabled. Select `device="metal"` on a supported Mac to use Metal acceleration.

## Deploy deliberately

Each database process selects one execution device. GPU admission fails explicitly when project
data and indexes do not fit; canonical graph data is never silently paged or truncated. One
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
