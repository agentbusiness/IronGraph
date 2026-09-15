import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from irongraph import Client, EmbeddedDatabase


class QueryHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        assert self.path == "/api/query"
        length = int(self.headers["content-length"])
        request = json.loads(self.rfile.read(length))
        events = [
            {
                "type": "schema",
                "request_id": request["request_id"],
                "columns": [
                    {"name": "answer", "value_type": "INTEGER", "nullable": False}
                ],
            },
            {
                "type": "batch",
                "request_id": request["request_id"],
                "sequence": 0,
                "row_count": 1,
                "columns": [
                    {
                        "name": "answer",
                        "value_type": "INTEGER",
                        "values": [{"type": "integer", "value": "42"}],
                    }
                ],
            },
            {
                "type": "summary",
                "request_id": request["request_id"],
                "bookmark": {"term": 1, "index": 1},
                "statistics": {
                    "elapsed_ms": 0,
                    "elapsed_us": 0,
                    "rows": 1,
                    "nodes": 0,
                    "edges": 0,
                    "updates": 0,
                },
                "truncated": False,
                "truncation_reason": None,
            },
        ]
        body = ("\n".join(json.dumps(event) for event in events) + "\n").encode()
        self.send_response(200)
        self.send_header("content-type", "application/x-ndjson")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def embedded_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="irongraph-python-") as temporary:
        data_dir = Path(temporary) / "chosen-database-directory"
        with EmbeddedDatabase(data_dir, device="cpu", load_embeddings=False) as database:
            database.query("CREATE PROJECT app")
            database.query("USE app CREATE (:Item {value: 42})")
            database.query(
                "USE app CREATE (:Document {id: 'guide', body: $body, embedding: [1.0, 0.0]})",
                parameters={"body": "Full document body — searchable text"},
            )
            database.query("USE app CREATE TEXT INDEX document_body FOR (d:Document) ON (d.body)")
            database.query("USE app MATCH (d:Document {id: 'guide'}) SET d.body = $body",
                           parameters={"body": "Updated complete document body"})
            ranked = database.query(
                "USE app MATCH (d:Document) WHERE d.body CONTAINS 'complete' "
                "RETURN d.id, vector.cosine(d.embedding, [1.0, 0.0]) AS score"
            )
            assert ranked["rows"][0][0]["value"] == "guide"
            assert abs(ranked["rows"][0][1]["value"] - 1.0) < 1e-6
            database.query("USE app USE LAYER WORKSPACE WRITE LAYER WORKSPACE CREATE (:Draft {id: 'draft'})")
            assert not database.query("USE app MATCH (d:Draft) RETURN d")["rows"]
            assert len(database.query("USE app USE LAYER WORKSPACE MATCH (d:Draft) RETURN d")["rows"]) == 1
            for statement in (
                "CREATE TOPIC activity PARTITIONS 2", "CREATE EXCHANGE routing TYPE TOPIC",
                "CREATE QUEUE jobs STREAM", "BIND QUEUE jobs TO EXCHANGE routing KEY documents",
            ):
                database.query(f"USE app {statement}")
            for statement in ("SHOW TOPICS", "SHOW QUEUES", "SHOW EXCHANGES", "SHOW INDEXES"):
                assert database.query(f"USE app {statement}")["rows"], statement
            project_id = database.query("USE app RETURN 1")["catalog"]["project_id"]
            assert database.status()["ready"]
            ack = database.stream_append({"project_id": project_id, "topic": "activity", "partition": 0,
                "records": [{"key": [0, 255], "headers": {"source": [80]}, "value": [1, 2, 3], "create_time_ms": 1234}]})
            assert ack["first_offset"] == 0
            fetch = {"project_id": project_id, "topic": "activity", "partition": 0, "offset": 0, "max_records": 1, "max_bytes": 4096}
            page = database.stream_fetch(fetch)
            assert page["high_watermark"] == 1
            assert page["records"][0][1]["payload"] == [1, 2, 3]
            bounded = database.query("UNWIND [1,2,3] AS n RETURN n", project_id=project_id,
                                     query_options={"bookmark": ack["bookmark"], "limits": {"rows": 3}})
            assert len(bounded["rows"]) == 3
            try:
                database.query("UNWIND [1,2,3] AS n RETURN n", project_id=project_id,
                               query_options={"limits": {"rows": 1}})
            except RuntimeError as error:
                assert "ResultBudgetExceeded" in str(error)
            else:
                raise AssertionError("row limit was ignored")
            database.snapshot()
            database.flush()
        with EmbeddedDatabase(data_dir, device="cpu", load_embeddings=False) as reopened:
            assert reopened.stream_fetch(fetch)["records"] == page["records"]
            result = reopened.query(
                "USE app MATCH (n:Item) RETURN n.value AS value"
            )
            assert result["rows"][0][0] == {"type": "integer", "value": "42"}
            document = reopened.query("USE app MATCH (d:Document {id: 'guide'}) RETURN d.body")
            assert document["rows"][0][0]["value"] == "Updated complete document body"
            reopened.query("USE app MATCH (d:Document {id: 'guide'}) DELETE d")
        with EmbeddedDatabase(data_dir, device="cpu", load_embeddings=False) as reopened:
            assert not reopened.query("USE app MATCH (d:Document) RETURN d")["rows"]


def embedding_qualification() -> None:
    """Release gate: runs the real model on the selected release backend."""
    device = os.environ.get("IRONGRAPH_QUALIFY_DEVICE", "cpu")
    with tempfile.TemporaryDirectory(prefix="irongraph-python-embedding-") as temporary:
        with EmbeddedDatabase(temporary, device=device, load_embeddings=True) as database:
            database.query("CREATE PROJECT semantic")
            database.query(
                "USE semantic CREATE (:Document {id: 'graph', body: $body, embedding: [0.0]})",
                parameters={"body": "Graph databases store nodes and relationships for connected data."},
            )
            database.query(
                "USE semantic CREATE EMBEDDING INDEX document_semantic FOR (d:Document) "
                "FROM d.body INTO d.embedding USING MODEL default SIMILARITY COSINE"
            )
            search = (
                "USE semantic MATCH (d:Document) SEARCH d IN (EMBEDDING INDEX document_semantic "
                "FOR TEXT $text LIMIT 1) SCORE AS score RETURN d.id, d.body, score"
            )
            result = database.query(search, parameters={"text": "connected graph data"})
            assert len(result["rows"]) == 1
            assert result["rows"][0][0]["value"] == "graph"
            assert result["rows"][0][2]["type"] == "float"
            vector_result = database.query(
                "USE semantic MATCH (d:Document) SEARCH d IN (EMBEDDING INDEX document_semantic "
                "FOR VECTOR vector.normalize($vector) LIMIT 1) SCORE AS score RETURN d.id, score",
                parameters={"vector": [1.0] + [0.0] * 383},
            )
            assert len(vector_result["rows"]) == 1
            assert vector_result["rows"][0][0]["value"] == "graph"
            database.query("USE semantic MATCH (d:Document) SET d.body = $body",
                           parameters={"body": "Documents remain complete graph records after editing."})
            result = database.query(search, parameters={"text": "editing complete documents"})
            assert result["rows"][0][1]["value"] == "Documents remain complete graph records after editing."
            database.snapshot()
        with EmbeddedDatabase(temporary, device=device, load_embeddings=True) as reopened:
            assert len(reopened.query(search, parameters={"text": "complete documents"})["rows"]) == 1
            reopened.query("USE semantic MATCH (d:Document) DELETE d")
            assert not reopened.query(search, parameters={"text": "complete documents"})["rows"]


def remote_smoke() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), QueryHandler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        client = Client.api(f"http://127.0.0.1:{server.server_port}")
        result = client.query("RETURN 42 AS answer")
        assert result["rows"][0][0] == {"type": "integer", "value": "42"}
    finally:
        server.shutdown()
        server.server_close()
        serving.join()


if __name__ == "__main__":
    embedded_smoke()
    remote_smoke()
    if os.environ.get("IRONGRAPH_QUALIFY_EMBEDDINGS") == "1":
        embedding_qualification()
