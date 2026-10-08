#!/usr/bin/env python3
"""Measure complete results through the supported Python SDK, without an IPC timer."""
import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from irongraph import Client, EmbeddedDatabase


def checkpoint(stage):
    # The parent samples both processes while this runner waits outside query timing.
    print(json.dumps({"event": "checkpoint", "stage": stage, "pid": os.getpid()}), flush=True)
    if sys.stdin.readline().strip() != "continue":
        raise RuntimeError("Memory observer disconnected")


def complete(result, rows, expected=None):
    assert len(result["rows"]) == rows, (len(result["rows"]), rows)
    assert not result["summary"]["truncated"]
    assert all(len(row) == len(result["columns"]) for row in result["rows"])
    if expected is not None:
        assert float(result["rows"][0][0]["value"]) == expected


parser = argparse.ArgumentParser()
parser.add_argument("--transport", choices=["embedded", "api", "bolt"], required=True)
parser.add_argument("--endpoint")
parser.add_argument("--data-dir")
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--samples", type=int, default=5)
parser.add_argument("--warmups", type=int, default=2)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
nodes = manifest["nodes"]
database = (EmbeddedDatabase(args.data_dir, device="cpu") if args.transport == "embedded"
            else getattr(Client, args.transport)(args.endpoint))
administration = Client.api(manifest["http_endpoint"]) if args.transport == "bolt" else database
projects = {}


def query(statement, parameters=None, project="public_bench"):
    return database.query(f"USE {project} {statement}", parameters=parameters or {},
                          **({"project_id": projects[project]} if args.transport == "bolt" else {}))


def measure_workloads(workloads):
    for item in workloads:
        name, statement, rows, *expect = item
        project = "flights" if name.startswith("flights_") else "public_bench"
        checkpoint(name + ":before")
        durations = []
        result = None
        for iteration in range(args.warmups + args.samples):
            result = None
            start = time.perf_counter_ns()
            result = query(statement, project=project)
            elapsed = (time.perf_counter_ns() - start) / 1000
            complete(result, rows, expect[0] if expect else None)
            if iteration >= args.warmups:
                durations.append(elapsed)
            checkpoint(name + ":iteration:" + str(iteration))
        checkpoint(name + ":after")
        print(json.dumps({"event": "measurement", "operation": name, "nodes": nodes,
                          "p50": sorted(durations)[len(durations) // 2], "durations_us": durations,
                          "result_rows": len(result["rows"]), "status": "ok"}), flush=True)
        result = None


def select_project(name):
    result = administration.query(f"USE {name} RETURN 1")
    projects[name] = result["catalog"]["project_id"]


try:
    checkpoint("empty_graph_encoder_ready")
    administration.query("CREATE PROJECT public_bench")
    select_project("public_bench")
    stride = max(nodes // 100, 1)
    for name, statement, multiplier in [
        ("initial_node_ingest", 'UNWIND range($start,$end) AS row CREATE (:Node {value:row % 1000,bucket:row % 64,body:CASE WHEN row % $stride = 0 THEN toString(row) + ":" + $body ELSE null END}) RETURN count(*)', 1),
        ("initial_edge_ingest", "UNWIND range($start,$end) AS row UNWIND range(1,4) AS step MATCH (source:Node) WHERE id(source)=row+1 MATCH (target:Node) WHERE id(target)=((row+step*7919)%$nodes)+1 CREATE (source)-[:R]->(target) RETURN count(*)", 4),
    ]:
        checkpoint(name + ":before")
        start = time.perf_counter_ns()
        for offset in range(0, nodes, 8192):
            end = min(nodes - 1, offset + 8191)
            result = query(statement, {"start": offset, "end": end, "stride": stride,
                                       "body": "x" * 2048, "nodes": nodes})
            complete(result, 1, (end - offset + 1) * multiplier)
        elapsed = (time.perf_counter_ns() - start) / 1000
        checkpoint(name + ":after")
        print(json.dumps({"event": "measurement", "operation": name, "nodes": nodes,
                          "p50": elapsed, "durations_us": [elapsed], "result_rows": len(result["rows"]),
                          "status": "ok"}), flush=True)
        result = None
    workloads = manifest["workloads"]
    measure_workloads(workloads)
    for name in ["parallel_full_count_queries", "canonical_batch_insert", "reader_during_real_write"]:
        checkpoint(name + ":before")
        durations = []
        overlap = None
        for iteration in range(args.warmups + args.samples):
            with ThreadPoolExecutor(max_workers=8) as pool:
                start = time.perf_counter_ns()
                if name == "parallel_full_count_queries":
                    def read_many(_):
                        for _ in range(100):
                            complete(query("MATCH (n:Node) RETURN count(n)"), 1, nodes)
                    list(pool.map(read_many, range(8)))
                    rows = 800
                else:
                    items = list(range(256 if name == "canonical_batch_insert" else 4096))
                    finished = threading.Event()
                    def write():
                        result = query("UNWIND $items AS value CREATE (:BenchWrite {value:value}) RETURN count(*)", {"items": items})
                        complete(result, 1, len(items))
                        finished.set()
                    writing = pool.submit(write)
                    if name == "reader_during_real_write":
                        complete(query("MATCH (n:Node) RETURN count(n)"), 1, nodes)
                        overlap = not finished.is_set()
                    else:
                        writing.result()
                    rows = 1
                elapsed = (time.perf_counter_ns() - start) / 1000
                if name != "parallel_full_count_queries":
                    writing.result()
                    query("MATCH (n:BenchWrite) DELETE n")
            if iteration >= args.warmups:
                durations.append(elapsed)
            checkpoint(name + ":iteration:" + str(iteration))
        checkpoint(name + ":after")
        print(json.dumps({"event": "measurement", "operation": name, "nodes": nodes,
                          "p50": sorted(durations)[len(durations) // 2], "durations_us": durations,
                          "read_overlapped_write": overlap, "result_rows": rows, "status": "ok"}), flush=True)
    if manifest.get("flights"):
        administration.query("IMPORT DATASET flights")
        select_project("flights")
        measure_workloads(manifest["flight_workloads"])
    checkpoint("before_close_and_durable_snapshot")
finally:
    if args.transport == "embedded":
        database.close()
print(json.dumps({"event": "complete"}), flush=True)
