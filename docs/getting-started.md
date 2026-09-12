# Start here

In this guide, you will create a project, add a small graph, and find a path with Cypher. The example
uses the built-in Query view and takes about five minutes after IronGraph is running.

## Before you begin

You need:

- an official IronGraph distribution installed on your machine;
- a running standalone IronGraph instance; and
- a modern browser with access to the node's local web address.

If IronGraph is not running yet, follow [Install IronGraph](installation.md).

## 1. Open Query

Open [http://127.0.0.1:18484/web/](http://127.0.0.1:18484/web/) and select **Query**.

The default address is local to your machine. If your administrator configured a different address,
use the URL they provided.

## 2. Create a project

Run:

```cypher
CREATE PROJECT recommendations
```

A project is the isolation boundary for a graph. IronGraph requires every data query to name a
project, which prevents an application from writing to an accidental default database.

You can list available projects at any time:

```cypher
SHOW PROJECTS
```

The result contains the project's stable identifier and display name.

## 3. Create a graph

Run the following statement as one query:

```cypher
USE recommendations
CREATE
  (ada:Developer {name: 'Ada', language: 'Rust'}),
  (lin:Developer {name: 'Lin', language: 'Python'}),
  (mira:Developer {name: 'Mira', language: 'TypeScript'}),
  (graph:Technology {name: 'Graph databases'}),
  (ada)-[:RECOMMENDS {score: 10}]->(graph),
  (lin)-[:FOLLOWS]->(ada),
  (mira)-[:FOLLOWS]->(lin)
```

This creates four nodes and three directed relationships. Labels such as `Developer` describe a
node's role. Relationship types such as `FOLLOWS` describe how two nodes are connected. Properties
store values on either kind of graph entity.

## 4. Query a pattern

Find developers who are one or more `FOLLOWS` hops away from someone who recommends graph
databases:

```cypher
USE recommendations
MATCH (developer:Developer)-[:FOLLOWS*1..]->(:Developer)-[:RECOMMENDS]->(topic:Technology)
RETURN developer.name AS developer, topic.name AS recommendation
ORDER BY developer
```

Expected result:

| developer | recommendation |
| --- | --- |
| Lin | Graph databases |
| Mira | Graph databases |

The variable-length pattern `[:FOLLOWS*1..]` is the key graph operation: it follows a relationship
for one or more hops without requiring your application to implement a traversal loop.

## 5. Inspect the graph result

In **Query**, run:

```cypher
USE recommendations
MATCH (source)-[relationship]->(target)
RETURN source, relationship, target
```

In the result panel, select **Plot**. The graph visualization contains four nodes and three
relationships. This view visualizes the same data returned by Cypher; it is not a separate store.

## Use the same example from Python

Embedded applications run the same Cypher and use a directory chosen by the caller:

```python
from irongraph import EmbeddedDatabase

with EmbeddedDatabase("./data/quickstart") as database:
    database.query("CREATE PROJECT IF NOT EXISTS recommendations")
    database.query(
        """
        USE recommendations
        MERGE (ada:Developer {name: $name})
        SET ada.language = $language
        """,
        parameters={"name": "Ada", "language": "Rust"},
    )

    result = database.query(
        """
        USE recommendations
        MATCH (developer:Developer)
        RETURN developer.name AS name
        ORDER BY name
        """
    )
    print(result["rows"])
```

Expected output for this embedded example:

```text
[[{'type': 'string', 'value': 'Ada'}]]
```

Use parameters for application values. This keeps data separate from the Cypher statement and
avoids constructing queries through string interpolation.

## Where to go next

- Read [Architecture](architecture.md) to understand projects, layers, residency, and durability.
- Read [Embed IronGraph](embedding.md) for lifecycle and deployment guidance.
- Read [Features](features.md) for indexes, temporal data, vectors, algorithms, and streams.
