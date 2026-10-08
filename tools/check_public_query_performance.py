#!/usr/bin/env python3
"""Check public benchmark coverage, provenance, complete results and memory evidence."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("report", type=Path)
parser.add_argument("--sizes", default="100,10000,100000,1000000,2000000")
parser.add_argument("--flights", action="store_true")
args = parser.parse_args()
report = json.loads(args.report.read_text())
prior = json.loads(Path("performance-results/cpu-recovery/current/results.json").read_text())
renames = {
    "canonical_metadata_capture": "public_metadata",
    "canonical_scalar_point": "public_scalar_point",
    "reader_while_writer_paused": "reader_during_real_write",
}
expected = {renames.get(row["operation"], row["operation"]) for row in prior["measurements"]}
assert len(expected) == 33, "Historical workload inventory changed; inspect it"
assert report["schema"] == "irongraph.public-query-performance.v1"
assert "complete decoded" in report["timing_boundary"]
for key in ("executable_sha256", "harness_sha256", "javascript_client_sha256"):
    assert len(report["provenance"][key]) == 64, key
assert report["memory_samples"], "No process memory evidence"
assert not any(row.get("status") == "sampling_error" for row in report["memory_samples"])
roles = {"embedded_database_and_client"} if report["config"]["transport"] == "embedded" else {"database", "bolt_client" if report["config"]["transport"] == "bolt" else "api_client"}
footprints = report["physical_footprint_samples"]
assert footprints, "No physical footprint evidence, including GPU-owned memory"
for footprint in footprints:
    assert footprint["peak_physical_footprint_bytes_rounded"] >= footprint["physical_footprint_bytes_rounded"] > 0
    assert "Physical footprint" in footprint["raw"]
for nodes in map(int, args.sizes.split(",")):
    stages = {row["stage"] for row in footprints if row["nodes"] == nodes}
    assert {"empty_graph_encoder_ready", "before_close_and_durable_snapshot"} <= stages
    lifecycle = [row for row in report["lifecycle_memory"] if row["nodes"] == nodes]
    assert len(lifecycle) == 1 and lifecycle[0]["stage"] == "close_and_durable_snapshot"
    assert {item["role"] for item in lifecycle[0]["memory"]} == roles
    controls = [row for row in report["measurements"] if row["nodes"] == nodes and row["operation"] == "memory_sampler_control"]
    if report["config"].get("memory_control"):
        assert len(controls) == 4 and {row["iteration"] for row in controls} == set(range(4))
    else:
        assert not controls
    for control in controls:
        assert control["status"] == "ok" and control["samples"] == 500 and control["warmups"] == 20
        assert control["periodic_memory_sampling"] is (control["iteration"] % 2 == 0)
        assert math.isfinite(control["p50"]) and control["p95"] >= control["p50"] > 0
        assert control["server_reported_elapsed_us_p50"] >= 0
        assert "complete decoded" in control["timing_boundary"]
    rows = [row for row in report["measurements"] if row["nodes"] == nodes and not row["operation"].startswith("flights_") and row["operation"] != "memory_sampler_control"]
    assert {row["operation"] for row in rows} == expected, (nodes, "Missing or additional workload")
    for row in rows:
        assert row["status"] == "ok", row
        assert math.isfinite(row["p50"]) and row["p50"] > 0, row
        assert row["result_rows"] > 0, row
        assert {row["operation"] + ":before", row["operation"] + ":after"} <= stages
        assert {item["role"] for item in row["memory"]} == roles, row
        for memory in row["memory"]:
            assert memory["samples"] > 0
            assert memory["peak_rss_bytes"] >= max(memory["before_rss_bytes"], memory["after_rss_bytes"]) > 0
        if row["operation"] == "reader_during_real_write":
            assert row["read_overlapped_write"], "Read did not finish during the write"
if args.flights:
    counts = {"flights_nodes": 13859, "flights_route_ids": 66770, "flights_full_routes": 66770}
    rows = [row for row in report["measurements"] if row["operation"] in counts]
    assert len(rows) == 3
    for row in rows:
        assert row["status"] == "ok" and row["result_rows"] == counts[row["operation"]]
        assert row["dataset_nodes"] == 13859 and row["dataset_edges"] == 66770
        assert {item["role"] for item in row["memory"]} == roles
print("PUBLIC QUERY COVERAGE AND MEMORY EVIDENCE PASS")
