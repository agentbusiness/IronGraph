#!/usr/bin/env python3
"""Exercise IronGraph through real HTTP and stock Neo4j Bolt clients."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import neo4j
from neo4j import GraphDatabase


def typed_value(value: dict[str, Any]) -> Any:
    kind, raw = value.get("type"), value.get("value")
    if kind == "integer":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "boolean":
        return bool(raw)
    if kind == "null":
        return None
    return raw


def event_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "batch":
            continue
        columns = event.get("columns", [])
        for row_index in range(int(event.get("row_count", 0))):
            rows.append({column["name"]: typed_value(column["values"][row_index]) for column in columns})
    return rows


def http_query(base_url: str, project_id: str | None, query: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps({
        "request_id": str(uuid.uuid4()), "project_id": project_id, "query": query,
        "parameters": parameters or {}, "consistency": "PUBLISHED", "bookmark": None, "limits": {},
    }).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/query", data=body, method="POST",
        headers={"Accept": "application/x-ndjson", "Content-Type": "application/json"},
    )
    started = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=130) as response:
        response_body, status = response.read().decode(), response.status
    wall_us = (time.perf_counter_ns() - started) / 1_000
    events = [json.loads(line) for line in response_body.splitlines() if line]
    failure = next((event for event in events if event.get("type") == "error"), None)
    if failure:
        raise RuntimeError(f"HTTP query failed: {failure.get('code')}: {failure.get('message')}")
    summary = next((event for event in reversed(events) if event.get("type") == "summary"), {})
    return {"wall_us": wall_us, "server_us": int(summary.get("statistics", {}).get("elapsed_us", 0)), "http_status": status, "rows": event_rows(events)}


def record_http(operations: list[dict[str, Any]], base_url: str, project_id: str, operation: str, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    result = http_query(base_url, project_id, query, parameters)
    operations.append({"transport": "http_query", "operation": operation, "wall_us": result["wall_us"], "server_us": result["server_us"], "rows": result["rows"]})
    return result["rows"]


def record_bolt(operations: list[dict[str, Any]], session: Any, operation: str, query: str, **parameters: Any) -> list[dict[str, Any]]:
    started = time.perf_counter_ns()
    result = session.run(query, **parameters)
    rows = [record.data() for record in result]
    summary = result.consume()
    operations.append({
        "transport": "neo4j_driver_bolt", "operation": operation,
        "wall_us": (time.perf_counter_ns() - started) / 1_000,
        "server_available_after_us": summary.result_available_after,
        "server_consumed_after_us": summary.result_consumed_after, "rows": rows,
    })
    return rows


def operation_family(name: str) -> str:
    if name.startswith("write_nodes_"):
        return "write_nodes"
    if name.startswith("write_edges_"):
        return "write_edges"
    return name


def verify(report_path: Path) -> None:
    report = json.loads(report_path.read_text())
    operations = report.get("operations", [])
    transports = {operation.get("transport") for operation in operations}
    assert transports == {"http_query", "neo4j_driver_bolt"}, f"unexpected transports: {transports}"
    required = {"write_nodes", "write_edges", "point_read", "traversal", "aggregate", "louvain"}
    for transport in transports:
        present = {operation_family(operation["operation"]) for operation in operations if operation.get("transport") == transport}
        assert required <= present, f"{transport} missing {sorted(required - present)}"
    assert all(operation.get("wall_us", 0) > 0 for operation in operations)
    assert report.get("validation") == {"node_count": 20_000, "edge_count": 19_998}
    icij = report.get("icij_louvain", {})
    assert icij.get("assignments") == 2_017_662
    assert icij.get("samples") and max(icij["samples"]) < 120_000_000
    print(json.dumps({"verified": True, "operations": len(operations), "transports": sorted(transports), "validation": report["validation"]}))


def batched_rows(start: int, count: int) -> list[dict[str, int]]:
    return [{"id": index, "value": index % 97} for index in range(start, start + count)]


def batched_edges(start: int, count: int) -> list[dict[str, int]]:
    return [{"source": index, "target": index + 1, "weight": index % 11 + 1} for index in range(start, start + count)]


def run(base_url: str, bolt_uri: str, icij_project: str, output: Path) -> None:
    icij_samples = []
    icij_rows: list[dict[str, Any]] = []
    for _ in range(3):
        icij_result = http_query(base_url, icij_project, "CALL graph.louvain() YIELD community RETURN count(*) AS assignments, count(DISTINCT community) AS communities")
        icij_samples.append(icij_result["server_us"])
        icij_rows = icij_result["rows"]
    run_id = uuid.uuid4().hex[:12]
    project_name = f"route_probe_{run_id}"
    http_query(base_url, None, f"CREATE PROJECT `{project_name}`")
    projects = http_query(base_url, None, "SHOW PROJECTS")["rows"]
    project_id = next(str(project["project_id"]) for project in projects if project.get("display_name") == project_name)
    operations: list[dict[str, Any]] = []
    node_query = "UNWIND $rows AS row CREATE (:RouteNode {route_id: row.id, value: row.value})"
    edge_query = "UNWIND $rows AS row MATCH (a:RouteNode {route_id: row.source}), (b:RouteNode {route_id: row.target}) CREATE (a)-[:ROUTE_LINK {weight: row.weight}]->(b)"

    offset = 0
    for batch_size in (100, 1_000, 5_000, 3_900):
        record_http(operations, base_url, project_id, f"write_nodes_{batch_size}", node_query, {"rows": batched_rows(offset, batch_size)})
        offset += batch_size
    record_http(operations, base_url, project_id, "create_index", "CREATE INDEX route_node_id FOR (n:RouteNode) ON (n.route_id)")
    offset = 0
    for batch_size in (100, 1_000, 5_000, 3_899):
        record_http(operations, base_url, project_id, f"write_edges_{batch_size}", edge_query, {"rows": batched_edges(offset, batch_size)})
        offset += batch_size
    record_http(operations, base_url, project_id, "point_read", "MATCH (n:RouteNode {route_id: $id}) RETURN n.value AS value", {"id": 7_777})
    record_http(operations, base_url, project_id, "traversal", "MATCH (a:RouteNode {route_id: $id})-[:ROUTE_LINK*1..3]->(b) RETURN count(DISTINCT b) AS reached", {"id": 5_000})
    record_http(operations, base_url, project_id, "aggregate", "MATCH (n:RouteNode) RETURN count(n) AS nodes, sum(n.value) AS value_sum")
    record_http(operations, base_url, project_id, "louvain", "CALL graph.louvain() YIELD community RETURN count(*) AS assignments, count(DISTINCT community) AS communities")

    driver = GraphDatabase.driver(bolt_uri, auth=None)
    driver.verify_connectivity()
    with driver.session(database=project_id) as session:
        offset = 10_000
        for batch_size in (100, 1_000, 5_000, 3_900):
            record_bolt(operations, session, f"write_nodes_{batch_size}", node_query, rows=batched_rows(offset, batch_size))
            offset += batch_size
        offset = 10_000
        for batch_size in (100, 1_000, 5_000, 3_899):
            record_bolt(operations, session, f"write_edges_{batch_size}", edge_query, rows=batched_edges(offset, batch_size))
            offset += batch_size
        record_bolt(operations, session, "point_read", "MATCH (n:RouteNode {route_id: $id}) RETURN n.value AS value", id=17_777)
        record_bolt(operations, session, "traversal", "MATCH (a:RouteNode {route_id: $id})-[:ROUTE_LINK*1..3]->(b) RETURN count(DISTINCT b) AS reached", id=15_000)
        record_bolt(operations, session, "aggregate", "MATCH (n:RouteNode) RETURN count(n) AS nodes, sum(n.value) AS value_sum")
        record_bolt(operations, session, "louvain", "CALL graph.louvain() YIELD community RETURN count(*) AS assignments, count(DISTINCT community) AS communities")
        validation = record_bolt(operations, session, "validation", "MATCH (n:RouteNode) WITH count(n) AS node_count MATCH ()-[r:ROUTE_LINK]->() RETURN node_count, count(r) AS edge_count")[0]
    driver.close()

    report = {
        "schema": "irongraph-real-route-v1", "generated_unix_ms": int(time.time() * 1_000),
        "project_id": project_id, "project_name": project_name, "http_url": base_url,
        "bolt_uri": bolt_uri, "neo4j_driver_version": neo4j.__version__, "operations": operations,
        "validation": validation,
        "icij_louvain": {"project_id": icij_project, "nodes": 2_017_662, "edges": 1_006_150, "assignments": icij_rows[0]["assignments"], "communities": icij_rows[0]["communities"], "samples": icij_samples, "p50_us": statistics.median(icij_samples)},
        "median_http_wall_us": statistics.median(operation["wall_us"] for operation in operations if operation["transport"] == "http_query"),
        "median_bolt_wall_us": statistics.median(operation["wall_us"] for operation in operations if operation["transport"] == "neo4j_driver_bolt"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    verify(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", default="http://127.0.0.1:18484")
    parser.add_argument("--bolt", default="bolt://127.0.0.1:18485")
    parser.add_argument("--icij-project", default="a6c1dd5b-a162-8867-8848-0594ec59bfc0")
    parser.add_argument("--output", type=Path, default=Path("performance-results/real-graph-datasets/route-final.json"))
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    verify(args.verify) if args.verify else run(args.http, args.bolt, args.icij_project, args.output)


if __name__ == "__main__":
    main()
