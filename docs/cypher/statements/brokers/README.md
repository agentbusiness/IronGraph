# Streams and Queues statements

Create and operate project-scoped Kafka-compatible topics and AMQP-compatible queues, exchanges and
bindings through Cypher. Remote wire connections require mutual TLS; local plaintext listeners bind
only to loopback.

## Setup sequence

Create a disposable project with `CREATE PROJECT IF NOT EXISTS worked_example`. Execute each block
separately in order. Names are examples, not required application names.

```cypher
USE worked_example
CREATE TOPIC reference_topic PARTITIONS 1
```

```cypher
USE worked_example
CREATE QUEUE reference_queue CLASSIC
```

```cypher
USE worked_example
CREATE EXCHANGE reference_exchange TYPE direct
```

```cypher
USE worked_example
BIND QUEUE reference_queue TO EXCHANGE reference_exchange KEY created
```

Expected result: a topic, queue, exchange and binding in the selected project. Use the corresponding
SHOW statements to inspect them. Create `reference_topic_partitioned`, `reference_stream` and
`reference_fanout` with their creation-page examples before using advanced examples on those names.
Run resource creation only once per name unless the resource has been dropped.

## Reference

| Statement | Purpose |
| --- | --- |
| [`CREATE TOPIC`](./create-topic.md) | Create a project-scoped Kafka-compatible topic. |
| [`ALTER TOPIC`](./alter-topic.md) | Change a topic's retention period. |
| [`SHOW TOPICS`](./show-topics.md) | Inspect topic partitions and retained records. |
| [`CLEAR TOPIC`](./clear-topic.md) | Remove retained records from a topic while retaining its declaration. |
| [`DROP TOPIC`](./drop-topic.md) | Remove a topic and its retained records. |
| [`CREATE QUEUE`](./create-queue.md) | Create a project-scoped AMQP-compatible classic queue or stream queue. |
| [`ALTER QUEUE`](./alter-queue.md) | Change queue retention in whole days. |
| [`SHOW QUEUES`](./show-queues.md) | Inspect queue kind, retained messages and availability. |
| [`PURGE QUEUE`](./purge-queue.md) | Remove queued messages while retaining the queue declaration. |
| [`DROP QUEUE`](./drop-queue.md) | Remove a queue and its retained messages. |
| [`CREATE EXCHANGE`](./create-exchange.md) | Create a project-scoped AMQP-compatible exchange. |
| [`SHOW EXCHANGES`](./show-exchanges.md) | Inspect exchange declarations and binding counts. |
| [`BIND QUEUE`](./bind-queue.md) | Bind a queue to an exchange with a routing key. |
| [`UNBIND QUEUE`](./unbind-queue.md) | Remove an exchange-to-queue binding. |
| [`DROP EXCHANGE`](./drop-exchange.md) | Remove an exchange declaration and its bindings. |
| [`SHOW CONSUMER LAG`](./show-consumer-lag.md) | Inspect Kafka-compatible consumer-group progress. |

Next, run [SHOW TOPICS](./show-topics.md), [SHOW QUEUES](./show-queues.md) and
[SHOW EXCHANGES](./show-exchanges.md) to inspect your resources before changing them.

