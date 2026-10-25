# IronGraph documentation

IronGraph is an embeddable Cypher graph database written in Rust, with sub-µs point reads. Use Cypher for graph queries,
semantic search, temporal properties, graph algorithms, and database administration.
Run it inside Python, Node.js, or Rust, or connect to a standalone instance.

If this is your first time using IronGraph, begin with [Start here](getting-started.md). You will
create a project, write a small property graph, and run your first traversal.

## Choose your path

| Your goal | Read this |
| --- | --- |
| Create and query a graph in a few minutes | [Start here](getting-started.md) |
| Understand the deployment and data model | [Architecture](architecture.md) |
| Install and run a standalone instance | [Install IronGraph](installation.md) |
| Add IronGraph to an application process | [Embed IronGraph](embedding.md) |
| Connect an AI host through MCP | [Use IronGraph as an AI second brain](mcp.md) |
| Review the database capabilities and boundaries | [Features](features.md) |
| Look up Cypher syntax and runnable examples | [Cypher reference](cypher/README.md) |
| Try graph algorithms on bundled data | [Reference datasets](cypher/datasets.md) |

## SDK documentation

| Application | Package | Guide |
| --- | --- | --- |
| Python | `irongraph` | [Python SDK](../bindings/python/README.md) |
| Node.js | `@irongraph/node` | [Node.js SDK](../bindings/node/README.md) |
| Rust | `irongraph-sdk` | [Rust SDK](../bindings/rust/README.md) |
| Browser and React | `@irongraph/client` | [JavaScript and React SDK](../bindings/javascript/README.md) |

## Core concepts

**Project**  
A named, isolated graph. Every graph query runs against an explicit project; IronGraph does not
create or select an implicit default project.

**Property graph**  
Data represented as nodes, relationships, labels, relationship types, and properties. Cypher lets
you describe graph patterns instead of writing traversal code.

**Graph layer**  
One of three semantic layers within a project: `OBSERVED`, `KNOWLEDGE`, or `WORKSPACE`. The first two
form the default read view. `WORKSPACE` is included only when a query requests it explicitly.

**Inference device**

The CPU, Metal device, or CUDA device selected for local text embedding. Graph storage and execution
always use one shared CPU graph; inference is selected independently.

## Supported access modes

- Embedded Rust, Python, and Node.js
- Remote Rust, Python, and Node.js through the Query API or Bolt
- Browser and React applications through the Query API
- AI hosts through local MCP over stdio or Streamable HTTP
- The built-in web application at `/web/` and supported Bolt drivers

Plain local listeners bind to loopback. Remote Query API, Bolt, Kafka-compatible Streams, and
AMQP-compatible Queues connections require mutual TLS and a provisioned credential for the intended
project and operation scope.
