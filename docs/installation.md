# Install IronGraph

This guide starts a standalone IronGraph instance for local development and outlines the production
connection boundary. Use [Embed IronGraph](embedding.md) when the database should live inside your
application process.

## Choose a distribution

IronGraph is open-source software released under the Apache License 2.0. Build it from source or use
a published package that matches your platform. Do not assume an unpublished package is available.

For the npm launcher, install Node.js 20.17 or later with npm. The official `irongraph` package and
its matching platform package must be published on npm before using the commands below. Supported
build targets are macOS 15+ ARM64 and Linux ARM64/AMD64 with glibc 2.28+. The launcher supplies a
compiled database with the web console included; it does not compile the engine on your computer.
The launcher also requires the `ps` system utility for process identification; on Linux it is
provided by the `procps` or `procps-ng` package.

You can alternatively obtain the matching standalone archive from the official binary release.
Extract it and run `./bin/irongraph` from the extracted directory. Direct execution does not require
Node.js, Python, a Rust compiler, or a separate frontend server. SDK packages are separate from
the standalone launcher and executable. Reserve storage for the database, snapshots, and model.

An embedding-enabled first start also needs access to the official artifact source for your IronGraph
release, or encoder artifacts preprovisioned by your administrator. Reserve the release-specific
multi-gigabyte storage budget before startup. Artifact installation, verification, device loading,
and warm-up make the first start longer than later starts.

Python embedding requires Python 3.9 or later. The native Node.js SDK requires Node.js 20.17 or
later. Available operating-system and processor builds depend on the current distribution.

## Start a local instance

Start a database that keeps running after you close the terminal:

```sh
npx irongraph start --background
```

The local defaults are:

| Surface | Address |
| --- | --- |
| Web application and Query API | `127.0.0.1:18484` |
| Bolt | `127.0.0.1:18485` |
| Kafka-compatible Streams | `127.0.0.1:18486` |
| AMQP-compatible Queues | `127.0.0.1:18487` |
| Local MCP HTTP | `127.0.0.1:18488` |

Data persists in `$HOME/.irongraph/data`. Check startup progress and readiness:

```sh
npx irongraph logs
npx irongraph status
```

Open [http://127.0.0.1:18484/web/](http://127.0.0.1:18484/web/) after status reports that the
instance is ready. The first embedding-enabled start can take longer while the verified local text
encoder is installed, loaded, bound to the selected device, and warmed.

The address is served by your local database;
the public project page is not your database console.
To stop, run `npx irongraph stop`. Restart with the same start command to reopen your persisted
data. Background mode does not install an operating-system service or start after a reboot.

Use `npx irongraph start` without `--background` for foreground execution. Keep that terminal
open, and press `Ctrl+C` to stop cleanly. Add `--execution-backend cpu` to select CPU explicitly,
or `--execution-backend metal` to select Metal on a supported Mac.

## Choose the local data directory

Set an absolute directory before starting IronGraph:

```sh
npx irongraph start --background --data-dir "$HOME/.local/share/irongraph/dev"
```

The directory contains the canonical database state and snapshots. Give the IronGraph process
exclusive access, place it on persistent local storage, and include the whole directory in your
backup policy. Do not open the same directory from another standalone or embedded process.
Pass the same `--data-dir` to `status`, `logs`, and `stop` when managing this instance.

Launcher state, logs, and runtime binaries live outside npm's package cache. Set
`IRONGRAPH_CLI_HOME` to change their parent directory; this does not move your database or the
embedding model cache. Use the same setting for every launcher command managing that instance.

## Configure local listener addresses

Override a local address only when the default port conflicts with another process:

```sh
IRONGRAPH_BOLT_ADDR=127.0.0.1:19485 \
npx irongraph start --background --http-addr 127.0.0.1:19484
```

Keep plain listeners on `127.0.0.1` or `::1`. A non-loopback client must use a separately configured
remote listener with mutual TLS.

This example serves the console at `http://127.0.0.1:19484/web/`. Configure other local listeners
with `IRONGRAPH_STREAM_ADDR`, `IRONGRAPH_QUEUE_ADDR`, and `IRONGRAPH_MCP_ADDR`. An occupied port
produces an error; IronGraph does not silently choose another port. Development uses the same
database ports, with a separate frontend development server on loopback port `18489`.

The default port block was unassigned in the [IANA registry](https://www.iana.org/assignments/service-names-port-numbers/)
when checked on September 5, 2026, and is below the Linux default temporary-port range of
32768–60999. This reduces expected collisions but cannot guarantee availability on your host.
[Linux networking documentation](https://docs.kernel.org/networking/ip-sysctl.html#ip-variables)

## Verify the installation

In the Query view, run:

```cypher
CREATE PROJECT IF NOT EXISTS installation_check
```

Then run:

```cypher
USE installation_check
RETURN 'IronGraph is ready' AS status
```

Expected result:

| status |
| --- |
| IronGraph is ready |

If the web application opens but the query does not complete, check the startup log for device
admission or encoder warm-up errors before changing listener settings.

## Configure a production connection boundary

Production clients outside the host use mutual TLS. Configure the remote listener set with a server
certificate, its private key, and the certificate authority used to trust client certificates. Then
provision each client certificate for its project, protocol, permitted operations, and graph layers
through the documented administration workflow. A valid trust chain authenticates identity but does
not grant authorization. The same requirement applies independently to remote Query API, Bolt,
Streams, and Queues listeners.

Do not expose a plain local listener through a proxy or bind it to a public interface. Mutual TLS is
part of the remote protocol boundary, not an optional application convention.

Use the following readiness sequence in production:

1. Confirm that the database directory is writable and on persistent storage.
2. Confirm that the selected CPU, Metal, or CUDA backend is available.
3. Wait for project and index admission to complete.
4. Wait for the enabled local text encoder to load and warm.
5. Verify a Cypher query over the intended mutually authenticated client path.
6. Verify that a clean shutdown completes before replacing or stopping the host.

## Upgrade and recovery discipline

Use only version-matched database and SDK artifacts from the same IronGraph release. Before an
upgrade, stop writers, perform a clean shutdown, and preserve a backup of the complete database
directory. Never copy individual WAL or snapshot files between active installations.

If startup recovery reports an error, preserve the original directory and startup log. Do not edit
database files manually. Restore a known-good full-directory backup or follow the recovery procedure
provided with your IronGraph release.

## Troubleshoot startup and connections

| Symptom | Check |
| --- | --- |
| The web address does not open | Confirm that startup completed, the configured HTTP address is correct, and another process is not using the port. |
| Startup cannot prepare text embedding | Confirm access to the official artifact source or preprovisioned encoder artifacts, available storage, and selected-device capacity. |
| A project fails device admission | Confirm that the selected device can hold the project, its indexes, and the enabled text encoder together; otherwise use a larger device or the CPU backend. |
| The data directory is rejected | Use an absolute, writable directory that is not open in another IronGraph process. |
| Mutual TLS succeeds but a query is denied | Confirm that the certificate is provisioned for the requested project, protocol, operation, and graph layers. |
| A remote client version cannot connect | Use a protocol and client version listed as supported in the IronGraph release information. |

Preserve the startup log and exact client error when escalating a problem. Do not include private
keys, client certificates, or graph data in a diagnostic bundle unless your security process
explicitly authorizes it.
