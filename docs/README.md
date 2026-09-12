# IronGraph documentation

IronGraph is a GPU-first, single-node graph database for applications that need a local graph,
Cypher access, and predictable ownership of their data. Run it as a standalone database or embed it
in a Rust, Python, or Node.js process.

If this is your first time using IronGraph, begin with [Start here](getting-started.md). You will
create a project, write a small property graph, and run your first traversal.

## Choose your path

| Your goal | Read this |
| --- | --- |
| Create and query a graph in a few minutes | [Start here](getting-started.md) |
| Understand the deployment and data model | [Architecture](architecture.md) |
| Install and run a standalone node | [Install IronGraph](installation.md) |
| Add IronGraph to an application process | [Embed IronGraph](embedding.md) |
| Connect an AI host through MCP | [Use IronGraph as an AI second brain](mcp.md) |
| Review the database capabilities and boundaries | [Features](features.md) |

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

**Execution device**  
The CPU, Metal device, or CUDA device selected for the database process. A process uses one device.
GPU-backed deployments keep admitted project data and derived indexes resident on that device.

## Supported access modes

- Embedded Rust, Python, and Node.js
- Remote Rust, Python, and Node.js through the Query API or Bolt
- Browser and React applications through the Query API
- AI hosts through the local MCP stdio server
- Supported Bolt drivers; consult the compatibility information supplied with your IronGraph release
- The built-in web application at `/web/`

Plain local listeners bind to loopback. Remote Query API, Bolt, Kafka-compatible Streams, and
AMQP-compatible Queues connections require mutual TLS and a provisioned credential for the intended
project and operation scope.
