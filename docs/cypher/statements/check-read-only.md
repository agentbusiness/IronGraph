# `CHECK READ ONLY`

Validate that a parameterized Cypher statement is read-only without executing it. Use this check when
a host needs to accept a query but reject graph or schema mutations.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `CHECK READ ONLY` with string request parameter `statement` |
| Relationship to standard Cypher | IronGraph extension |

## What it does

IronGraph parses and binds the `statement` parameter with write and schema permissions disabled.
A read-only candidate succeeds without running its graph read. A write, unavailable project or
invalid candidate reports an error. This check does not grant access or establish a security
boundary for a later independently submitted request.

## Simple example

Prerequisite: an existing project named `worked_example`. Pass these request parameters using your
Query API or SDK's supported parameter argument:

```json
{"statement": "USE worked_example MATCH (node) RETURN count(node) AS nodes"}
```

```cypher
CHECK READ ONLY
```

Expected result: success without graph rows and without executing the candidate's count.

## Advanced example

The same check accepts a candidate with a parameter to be supplied when it is later executed:

```json
{"statement": "USE worked_example MATCH (node) WHERE node.name = $name RETURN node"}
```

```cypher
CHECK READ ONLY
```

Expected result: successful read-only validation. Replacing the candidate with a CREATE, SET, DELETE
or schema-changing statement produces an error. Missing or non-string `statement` is also an error.

## Limitations and trade-offs

Validation does not return candidate results, estimate cost, reserve graph state or promise that a
later query sees the same schema. Continue to enforce the host's authorization for actual execution.

Next, execute the validated read through your normal Cypher connection when appropriate.
