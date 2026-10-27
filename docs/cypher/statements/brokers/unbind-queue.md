# `UNBIND QUEUE`

Remove an exchange-to-queue binding. This page is for applications operating IronGraph Streams and Queues.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `UNBIND QUEUE <queue> TO EXCHANGE <exchange> KEY <routing-key>` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | Broker resources in an existing project |

## What it does

Remove an exchange-to-queue binding.

## How it behaves

Use the same queue, exchange and routing key as the binding to remove. Existing queue messages remain; future routing no longer uses that binding.

Resources belong to the selected project. These statements operate on broker state through Cypher;
they do not create a second query language or convert message-processing intermediates into nodes.
Standalone SHOW statements cannot be composed with YIELD, WITH or WHERE.

## Prerequisites

Use an existing disposable project named `worked_example`. Resource-changing examples assume the
named topics, queues, exchanges and bindings exist where the statement requires them. Follow
[the complete setup sequence](./README.md) first. Recreate a resource after an example drops it.
Local wire listeners bind only to loopback; remote Kafka-compatible and AMQP-compatible clients
require mutual TLS. The statement itself uses your existing Cypher connection.

## Simple example

```cypher
USE worked_example
UNBIND QUEUE reference_queue TO EXCHANGE reference_exchange KEY created
```

Expected result: the operation completes without graph rows.

## Advanced example

```cypher
USE worked_example
UNBIND QUEUE reference_stream TO EXCHANGE reference_exchange KEY `events.created`
```

Expected result: the named operation applies to the second example resource.

## Limitations and trade-offs

Destructive administration is explicit. Inspect the relevant SHOW statement and confirm the intended
consequence before clearing, purging or dropping resources. Retention can remove older messages;
applications must account for consumer offsets and delivery behavior.

## See also

- [Streams and Queues statements](./README.md)
- [SHOW CONSUMER LAG](./show-consumer-lag.md)

