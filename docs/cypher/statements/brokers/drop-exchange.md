# `DROP EXCHANGE`

Remove an exchange declaration and its bindings. This page is for applications operating IronGraph Streams and Queues.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `DROP EXCHANGE <name>` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | Broker resources in an existing project |

## What it does

Remove an exchange declaration and its bindings.

## How it behaves

Destructive: routing through this exchange stops. Dropping the exchange does not itself drop its queues.

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
DROP EXCHANGE reference_exchange
```

Expected result: the operation completes without graph rows.

## Advanced example

```cypher
USE worked_example
DROP EXCHANGE reference_fanout
```

Expected result: the named operation applies to the second example resource.

## Limitations and trade-offs

Destructive administration is explicit. Inspect the relevant SHOW statement and confirm the intended
consequence before clearing, purging or dropping resources. Retention can remove older messages;
applications must account for consumer offsets and delivery behavior.

## See also

- [Streams and Queues statements](./README.md)
- [SHOW CONSUMER LAG](./show-consumer-lag.md)

