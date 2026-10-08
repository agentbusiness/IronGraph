#!/usr/bin/env python3
"""Reject incomplete public-language query or memory qualification reports."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("report", type=Path)
parser.add_argument("--sizes", default="100,10000,100000")
parser.add_argument("--flights", action="store_true")
args = parser.parse_args()
report = json.loads(args.report.read_text())
assert report["schema"] == "irongraph.public-sdk-performance.v1"
assert "complete decoded" in report["timing_boundary"]
assert not any(row.get("status") == "error" for row in report["measurements"])
assert not any(row.get("status") == "sampling_error" for row in report["memory_samples"])
for value in report["provenance"].values():
    assert len(value) == 64
prior = json.loads(Path("performance-results/cpu-recovery/current/results.json").read_text())
rename = {"canonical_metadata_capture": "public_metadata", "canonical_scalar_point": "public_scalar_point",
          "reader_while_writer_paused": "reader_during_real_write"}
expected = {rename.get(row["operation"], row["operation"]) for row in prior["measurements"]}
assert len(expected) == 33
language = report["config"]["language"]
roles = ({f"{language}_client", "database"} if report["config"]["transport"] in ["api", "bolt"]
         else {f"{language}_database_and_client"})
for nodes in map(int, args.sizes.split(",")):
    rows = [row for row in report["measurements"] if row.get("nodes") == nodes and not row.get("operation", "").startswith("flights_")]
    assert {row["operation"] for row in rows} == expected
    stages = {row["stage"] for row in report["physical_footprint_samples"] if row["nodes"] == nodes}
    assert {"empty_graph_encoder_ready", "before_close_and_durable_snapshot"} <= stages
    lifecycle = [row for row in report["lifecycle_memory"] if row["nodes"] == nodes]
    assert len(lifecycle) == 1 and lifecycle[0]["samples"]
    for row in rows:
        assert math.isfinite(row["p50"]) and row["p50"] > 0
        assert row["result_rows"] > 0 and row["status"] == "ok"
        assert {row["operation"] + ":before", row["operation"] + ":after"} <= stages
        assert {memory["role"] for memory in row["memory"]} == roles
        for memory in row["memory"]:
            assert memory["samples"] > 0
            assert memory["peak_rss_bytes"] >= max(memory["before_rss_bytes"], memory["after_rss_bytes"]) > 0
        if row["operation"] == "reader_during_real_write":
            assert row["read_overlapped_write"]
for footprint in report["physical_footprint_samples"]:
    assert "Physical footprint" in footprint["raw"]
if args.flights:
    expected_flights = {"flights_nodes": 13859, "flights_route_ids": 66770, "flights_full_routes": 66770}
    rows = [row for row in report["measurements"] if row.get("operation") in expected_flights]
    assert len(rows) == 3
    for row in rows:
        assert row["result_rows"] == expected_flights[row["operation"]] and row["status"] == "ok"
        assert {memory["role"] for memory in row["memory"]} == roles
print("PUBLIC SDK QUERY COVERAGE AND MEMORY EVIDENCE PASS")
