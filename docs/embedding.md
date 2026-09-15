# Embed IronGraph

Embedding runs IronGraph in the same process as your application. It is designed for software that
owns the database lifecycle and benefits from local calls without a network service.

The embedded database has the same project, layer, Cypher, transaction, WAL, snapshot, index, text
embedding, and execution-device semantics as the standalone instance.

Native queries and stream append/fetch execute inside your process. Opening an embedded database
does not launch a database executable or open an HTTP, Bolt, Kafka, or AMQP listener. Automatic
text-model installation requires network access when the verified model is not already installed.
For graph and stream applications that do not use text embeddings, explicitly disable model loading.

## Decide whether embedding fits

Embed IronGraph when:

- one application process owns the database;
- the application can choose and protect a local data directory;
- in-process startup and shutdown fit the application's lifecycle; and
- other processes do not need to open the same database directory.

Run a standalone instance when multiple processes need concurrent access, a browser application is the
primary client, or protocol-level integration is required.

Only one embedded IronGraph instance can be active in a process. The selected data directory must be
exclusive to that process.

## Native streams and query controls

Create projects and topics with Cypher, then select the immutable project UUID for native stream
calls. Native streams use the same topics, records, offsets, and persistence as Kafka clients.
The following Python example requires an installed native package and no standalone application:

```python
from irongraph import EmbeddedDatabase

with EmbeddedDatabase("./data/streams", device="cpu", load_embeddings=False,
                      budgets={"worker_threads": 2, "max_concurrent_operations": 8}) as db:
    db.query("CREATE PROJECT IF NOT EXISTS streams")
    project = db.query("USE streams RETURN 1")["catalog"]["project_id"]
    db.query("CREATE TOPIC IF NOT EXISTS events PARTITIONS 1", project_id=project)
    ack = db.stream_append({
        "project_id": project, "topic": "events", "partition": 0,
        "records": [{"key": None, "headers": {}, "value": list(b"hello"),
                     "create_time_ms": None}],
    })
    page = db.stream_fetch({
        "project_id": project, "topic": "events", "partition": 0,
        "offset": ack["first_offset"], "max_records": 1, "max_bytes": 4096,
    })
    assert bytes(page["records"][0][1]["payload"]) == b"hello"
```

The expected payload is `hello`. Closing and reopening the same directory preserves it. Next,
use `next_offset` for the next page and check `high_watermark` and `truncated` to determine the
prefix covered by the read. A high watermark is an exclusive end offset captured for that page;
it does not promise that no later records will arrive. A first record that exceeds `max_bytes`
returns an explicit budget error. Retained or unreadable records return errors rather than EOF.

Rust exposes `stream_append`, `stream_fetch`, `flush`, `status`, and `cancel` on `EmbeddedDatabase`.
Node.js exposes `streamAppend`, `streamFetch`, `flush`, `status`, and `cancel`. Binary values use byte
arrays; null and empty values are distinct. Returned records include their persisted message
identity and ingress metadata. Topic administration continues to use Cypher.

Python's `query_options` and Node.js's fourth query argument accept `bookmark`, `consistency`,
and `limits`. Rust carries these fields on `Query`. `CHECK READ ONLY` accepts candidate text
as the `$statement` parameter through the same query method.

Use `operation_options` in Python, the fifth query argument in Node.js, or
`query_with_options` in Rust to supply `operation_id` and `timeout_ms`. Stream calls accept the
same operation options. Cancellation IDs identify active calls, not durable idempotency keys;
`cancel` returns false when the call has not started or has already completed. Query cancellation
is cooperative. A stream append cancelled before dispatch is rejected; after dispatch, it waits
for the authoritative result or deadline. A dispatched write that times out has an uncertain
outcome and must not be retried automatically.

Acknowledgements mean the mutation has been published in memory. WAL persistence is asynchronous;
an abrupt process or host failure can lose recently acknowledged mutations that have not reached
durable storage. Call `flush` to make earlier acknowledged writes durable while keeping the handle
open. Explicit `close` drains and joins the WAL workers, establishes the final durable
boundary, and releases directory ownership. Independent query and stream calls are not one atomic
transaction. Always observe errors from explicit close.

Resource options include `max_write_bytes`, `max_concurrent_operations`, `worker_threads`,
`request_timeout_ms`, `startup_timeout_ms`, `snapshot_interval_ms`, and the device memory limits.
These are separate request, worker, and device budgets; they are not a total host-process RSS limit.
`status` reports readiness, the resolved data directory, and active operations. One handle can
serve concurrent projects up to its configured operation budget.

## Install an SDK

Install from the official package source or artifact bundle provided with your IronGraph
distribution. The supported package identities are:

| Runtime | Package | Mode |
| --- | --- | --- |
| Python 3.9+ | `irongraph` | Embedded and remote |
| Node.js 20.17+ | `@irongraph/node` | Embedded and remote |
| Rust 1.94 | IronGraph Rust libraries supplied with the distribution | Embedded and remote |
| Browser or React | `@irongraph/client` | Remote only |

For example, after you configure the official package source:

```sh
python -m pip install irongraph
```

```sh
npm install @irongraph/node
```

These commands identify the packages; availability and authentication are controlled by the
configured official distribution channel.

## Embed in Python

```python
from irongraph import EmbeddedDatabase

with EmbeddedDatabase(
    "./data/catalog",
    device="auto",
    load_embeddings=True,
) as database:
    database.query("CREATE PROJECT IF NOT EXISTS catalog")

    database.query(
        """
        USE catalog
        MERGE (product:Product {sku: $sku})
        SET product.name = $name
        """,
        parameters={"sku": "A-100", "name": "Trail camera"},
    )

    result = database.query(
        """
        USE catalog
        MATCH (product:Product {sku: $sku})
        RETURN product.name AS name
        """,
        parameters={"sku": "A-100"},
    )
    print(result["rows"])
```

Expected output:

```text
[[{'type': 'string', 'value': 'Trail camera'}]]
```

The context manager closes the database cleanly even when application code raises an exception.
Keep the database open for the lifetime of the owning application rather than opening it for each
query.

## Embed in Node.js

```javascript
const { EmbeddedDatabase } = require('@irongraph/node')

async function main() {
  const database = await EmbeddedDatabase.open('./data/catalog', 'auto', 0, true)

  try {
    await database.query('CREATE PROJECT IF NOT EXISTS catalog')

    await database.query(
      `USE catalog
       MERGE (product:Product {sku: $sku})
       SET product.name = $name`,
      null,
      { sku: 'A-100', name: 'Trail camera' },
    )

    const result = await database.query(
      `USE catalog
       MATCH (product:Product {sku: $sku})
       RETURN product.name AS name`,
      null,
      { sku: 'A-100' },
    )
    console.log(result.rows)
  } finally {
    await database.close()
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
```

The optional second argument is a project identifier. These examples use the Cypher `USE` prefix,
so they pass `null` and keep the project selection visible in the statement.

## Embed in Rust

Use the version-matched Rust dependency entries supplied with your IronGraph release. A minimal
embedded lifecycle is:

```rust
use irongraph::client::Query;
use irongraph::embedded::{EmbeddedDatabase, EmbeddedOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let database = EmbeddedDatabase::open(EmbeddedOptions::new("./data/catalog"))?;

    database.query(Query::new("CREATE PROJECT IF NOT EXISTS catalog"))?;
    let result = database.query(Query::new(
        "USE catalog MATCH (product:Product) RETURN product.name AS name",
    ))?;
    println!("{} row(s)", result.rows.len());

    database.close()?;
    Ok(())
}
```

Expected output:

```text
[[{ type: 'string', value: 'Trail camera' }]]
```

Call `close` during orderly shutdown so IronGraph can complete its snapshot and release the data
directory.

## Select an execution device

The portable options are `auto`, `cpu`, `metal`, and `cuda`, with an optional device ordinal for
Metal or CUDA. Automatic selection chooses the supported accelerator for the packaged build and
platform. Select `cpu` explicitly when portability or deterministic CPU behavior matters more than
acceleration.

Each process uses one device. A GPU-backed embedded database admits project data and derived indexes
only when they fit in the available device budget; admission failure is explicit.

The local text encoder loads and warms automatically by default. Set `load_embeddings=False` in
Python, or the corresponding final `false` option in Node.js, only when the application supplies
vector properties itself and does not use local text embedding. IronGraph never loads a generative
language model or falls back to an external inference service.

## Persist and close safely

- Choose a stable, absolute data directory in production.
- Keep the database object open for the owning process lifetime.
- Do not share the directory with another process or embedded instance.
- Use parameters for application values.
- Call `snapshot` when the application needs an explicit snapshot boundary.
- Close the database and wait for completion during graceful shutdown.
- Back up and restore the complete directory as one unit.

Dropping the last language-level reference attempts cleanup, but explicit close is the reliable
production lifecycle.

## Connect remotely instead

Python and Node.js also expose remote clients for the Query API and Bolt. Plain connections are
accepted only for loopback development. Use mutual TLS for any remote address:

```python
from irongraph import Client

client = Client.api_mtls(
    "https://graph.example.com:18484",
    "./certs/client.pem",
    "./certs/client-key.pem",
    "./certs/ca.pem",
)

result = client.query(
    "MATCH (product:Product {sku: $sku}) RETURN product.name AS name",
    parameters={"sku": "A-100"},
)
```

The remote credential supplies the project context. It must be provisioned by the local
administrator for that project, the Query API, the required operations, and the graph layers the
query can access. Remote credentials cannot run project-catalog administration such as `SHOW
PROJECTS`.

Browser and React packages are remote-only. On a remote deployment, the browser must be configured
to present a client certificate trusted by the IronGraph remote listener and authorized for the
required project and query scope. Use client and protocol versions listed as supported for your
IronGraph release.

## Understand query results

SDK results contain column metadata, rows, and a summary. Values use typed envelopes so integers,
temporal values, vectors, nodes, relationships, and paths retain their database type across language
and transport boundaries. For example, an integer may appear as:

```json
{"type": "integer", "value": "42"}
```

The string representation preserves the full integer range in JavaScript. Decode values according
to the `type` field instead of assuming every value is a native JSON scalar.
